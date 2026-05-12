"""
Market-Making Backtest Engine with ML Signal Integration
=========================================================
Simulates a market maker that adjusts bid/ask spreads based on
ML predictions vs a baseline fixed-spread market maker.

The simulation models:
  - Order placement at bid/ask prices
  - Fill probability based on price crossing the order level
  - Inventory risk and PnL tracking
  - Transaction costs (exchange fees)
"""

import json
import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.feather as pf

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_CAL_AVAILABLE = True
except Exception:
    IsotonicRegression = None
    SKLEARN_CAL_AVAILABLE = False

from shared.hb_config import normalize_exchange_id, read_simple_yaml_scalar
from shared.overlays import apply_adverse_size_cap
from shared.paths import BACKTEST_RESULTS_FILE, HB_FEE_OVERRIDES_FILE, HB_SCRIPT_CONFIG_FILE
from config import (
    BINANCE_DATA_DIR, CONFIDENCE_THRESHOLD, DATA_DIR, LATEST_MODEL_DIR,
    BASE_SPREAD_PCT, MIN_SPREAD_PCT, MAX_SPREAD_PCT,
    DIRECTION_WEIGHT, VOLATILITY_WEIGHT, ORDER_AMOUNT_USD,
    MIN_ORDER_AMOUNT_USD, MAX_ORDER_AMOUNT_USD,
    ORDER_LEVELS, ORDER_LEVEL_SPREAD_STEP_PCT,
    load_models, load_multi_horizon_models, load_neural_model,
    dynamic_confidence_threshold, compute_spreads as _compute_spreads,
)
from execution_learning import load_or_build_execution_features
from meta_ensemble import load_meta_ensemble, predict_direction_series
from orderbook_features import load_or_build_orderbook_features
from features import compute_features, compute_labels, get_feature_columns
from regime_model import apply_regime_model, load_regime_model


def _print_progress(done: int, total: int, t_start: float, best_label: str = "") -> None:
    """Print an in-place progress bar with ETA and current best to stdout."""
    elapsed = time.time() - t_start
    pct = done / total
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "#" * filled + "." * (bar_len - filled)
    eta_str = "--:--"
    if done > 0:
        remaining = elapsed / done * (total - done)
        m, s = divmod(int(remaining), 60)
        eta_str = f"{m:02d}:{s:02d}"
    line = f"  [{bar}] {done:>5}/{total} ({pct*100:5.1f}%)  elapsed {int(elapsed):>3}s  ETA {eta_str}"
    if best_label:
        line += f"  | best: {best_label}"
    sys.stdout.write("\r" + line)
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _write_json_report(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Report saved to: {path}")
    except PermissionError as exc:
        print(f"WARNING: could not write report {path}: {exc}")


def _fill_probability(order_price: float, candle_extreme: float, candle_range: float) -> float:
    if candle_range < 1e-10:
        return 0.0
    penetration = abs(candle_extreme - order_price) / candle_range
    return min(0.30 + 0.55 * penetration, 0.85)


def _default_backtest_fee_pct() -> float:
    strategy_config = HB_SCRIPT_CONFIG_FILE
    fee_overrides = HB_FEE_OVERRIDES_FILE
    exchange_name = normalize_exchange_id(read_simple_yaml_scalar(strategy_config, "exchange"))
    candidate_keys: list[str] = []
    if exchange_name:
        candidate_keys.append(f"{exchange_name}_maker_percent_fee")
    candidate_keys.append("kraken_maker_percent_fee")

    for key in candidate_keys:
        raw_value = read_simple_yaml_scalar(fee_overrides, key)
        if raw_value is None:
            continue
        try:
            fee_pct = float(raw_value)
        except ValueError:
            continue
        if fee_pct > 0:
            return fee_pct

    return 0.16


def _default_backtest_min_spread_pct() -> float:
    """Derive the backtest spread floor from the Hummingbot min_spread_floor config.

    Hummingbot enforces this floor on every order — the backtest must match it,
    otherwise the simulation generates fills at spreads too narrow to cover fees,
    producing an artificially negative Sharpe ratio.

    Falls back to fee_pct + 6 bps adverse-selection buffer if the config is missing.
    """
    strategy_config = HB_SCRIPT_CONFIG_FILE
    raw = read_simple_yaml_scalar(strategy_config, "min_spread_floor")
    if raw is not None:
        try:
            floor_frac = float(raw)
            if floor_frac > 0:
                # min_spread_floor is a fraction (e.g. 0.0022 = 22 bps).
                # MMConfig.min_spread_pct is a percentage (e.g. 0.22 = 22 bps).
                return floor_frac * 100
        except ValueError:
            pass
    # Fallback: fee per side + 6 bps (0.06%) adverse-selection buffer
    fee = _default_backtest_fee_pct()
    return fee + 0.06


# ── Simulation Parameters ──────────────────────────────────────────
@dataclass
class MMConfig:
    """Market-making simulation configuration."""
    base_spread_pct: float = field(default=None)  # Base spread each side (%) — auto-set from fee if None
    min_spread_pct: float = field(default_factory=_default_backtest_min_spread_pct)  # Fee-aware floor from Hummingbot min_spread_floor
    max_spread_pct: float = MAX_SPREAD_PCT     # Maximum spread cap (%) — from config.py
    order_amount_usd: float = ORDER_AMOUNT_USD  # Order size in USD — from config.py
    max_inventory: float = 500.0               # Max one-sided inventory (USD)
    fee_pct: float = field(default_factory=_default_backtest_fee_pct)  # Active maker fee (%), sourced from Hummingbot config
    # Synthetic live-book spread fed into compute_spreads so backtest mirrors the
    # book-anchored behavior of the signal server. ETH-USDT/BTC-USDT books on
    # kraken/binance are ~1-2 bps, so the book_half_spread*1.2 term is always
    # below the Hummingbot min_spread_floor and the floor dominates — matching
    # what the live signal server actually produces. Without this the backtest
    # uses BASE_SPREAD_PCT (30 bps) and then vol-amplifies to 50-75 bps while
    # live runs at 22 bps, causing systematic "zero trades" retrain failures.
    synthetic_book_spread_bps: float = 2.0

    def __post_init__(self):
        # Ensure base spread is at least the fee-aware floor
        if self.base_spread_pct is None:
            self.base_spread_pct = max(BASE_SPREAD_PCT, self.min_spread_pct)
        if self.base_spread_pct < self.min_spread_pct:
            self.base_spread_pct = self.min_spread_pct
    refresh_candles: int = 1                   # Order refresh interval (candles)
    inventory_skew_factor: float = 0.5         # How much inventory skews spread
    # ML-specific parameters — kept in sync with config.py live constants
    direction_weight: float = DIRECTION_WEIGHT      # How much direction shifts spread
    volatility_weight: float = VOLATILITY_WEIGHT    # How volatility scales spread
    confidence_threshold: float = CONFIDENCE_THRESHOLD  # Min confidence to act on direction
    # A/B harness options
    inventory_target_base_pct: float = 0.5
    inventory_target_tolerance_pct: float = 0.12
    inventory_target_skew_strength: float = 0.25
    order_levels: int = ORDER_LEVELS
    order_level_spread_step_pct: float = ORDER_LEVEL_SPREAD_STEP_PCT
    min_order_amount_usd: float = MIN_ORDER_AMOUNT_USD
    max_order_amount_usd: float = MAX_ORDER_AMOUNT_USD
    # Execution-fidelity parameters. These mirror live deployment risk controls
    # while staying conservative when live fill calibration data is unavailable.
    partial_fill_alpha: float = 2.0
    partial_fill_beta: float = 3.0
    late_fill_volatility_penalty: float = 0.35
    queue_volume_fraction: float = 1.0
    adverse_selection_penalty_alpha: float = 0.35
    latency_penalty_weight: float = 0.20
    as_inventory_risk_aversion: float = 0.50
    as_inventory_max_shift: float = 0.0050
    max_session_drawdown_pct: float = 0.012
    dd_size_taper_start_ratio: float = 0.50
    dd_size_floor: float = 0.40
    # Adaptive regime gate: when historical avg signed return for the current
    # regime is below `regime_gate_threshold_bps` and we have at least
    # `regime_gate_min_trades` prior fills in that regime, scale order size by
    # `regime_gate_size_suppression`. Default OFF (suppression=1.0) to preserve
    # existing behavior; set suppression=0.0 to skip quoting in losing regimes.
    regime_gate_enabled: bool = False
    regime_gate_min_trades: int = 15
    regime_gate_threshold_bps: float = -2.0
    regime_gate_size_suppression: float = 0.0


def apply_regime_gate_env_override(config: "MMConfig", *, verbose: bool = False) -> None:
    """Apply ML_BACKTEST_REGIME_GATE / ML_REGIME_GATE_* env-var overrides in place.

    Used by both `run_backtest` and the walk-forward path so the gate can be
    A/B-tested across the full validation pipeline without code edits.
    """
    if os.environ.get("ML_BACKTEST_REGIME_GATE", "").strip() not in {"1", "true", "on", "yes"}:
        return
    config.regime_gate_enabled = True
    try:
        config.regime_gate_min_trades = int(
            os.environ.get("ML_REGIME_GATE_MIN_TRADES", config.regime_gate_min_trades)
        )
    except ValueError:
        pass
    try:
        config.regime_gate_threshold_bps = float(
            os.environ.get("ML_REGIME_GATE_THRESHOLD_BPS", config.regime_gate_threshold_bps)
        )
    except ValueError:
        pass
    try:
        config.regime_gate_size_suppression = float(
            os.environ.get("ML_REGIME_GATE_SUPPRESSION", config.regime_gate_size_suppression)
        )
    except ValueError:
        pass
    if verbose:
        print(
            f"[regime-gate] enabled "
            f"min_trades={config.regime_gate_min_trades} "
            f"threshold_bps={config.regime_gate_threshold_bps} "
            f"suppression={config.regime_gate_size_suppression}"
        )


@dataclass
class Position:
    """Track market-maker state."""
    inventory_base: float = 0.0       # Amount of base asset held
    cash: float = 10000.0             # USD cash
    initial_cash: float = 10000.0
    total_fees: float = 0.0
    total_execution_penalty: float = 0.0
    trades: int = 0
    buys: int = 0
    sells: int = 0
    fill_events: int = 0
    fill_ratio_sum: float = 0.0
    pnl_history: list = field(default_factory=list)
    inventory_history: list = field(default_factory=list)


def compute_ml_spreads(
    dir_prob,
    vol_pred,
    config: MMConfig,
    current_inventory_pct: float,
    book_spread_bps: float | None = None,
):
    """
    Compute asymmetric bid/ask spreads based on ML predictions.

    Delegates to shared ``config.compute_spreads`` with per-backtest overrides.
    """
    conf_threshold = dynamic_confidence_threshold(config.confidence_threshold, vol_pred)
    return _compute_spreads(
        dir_prob,
        vol_pred,
        inventory_pct=current_inventory_pct,
        conf_threshold=conf_threshold,
        base_spread_pct=config.base_spread_pct,
        min_spread_pct=config.min_spread_pct,
        max_spread_pct=config.max_spread_pct,
        direction_weight=config.direction_weight,
        volatility_weight=config.volatility_weight,
        inventory_skew_factor=config.inventory_skew_factor,
        book_spread_bps=(
            config.synthetic_book_spread_bps
            if book_spread_bps is None
            else float(book_spread_bps)
        ),
    )


def compute_fixed_spreads(
    config: MMConfig,
    current_inventory_pct: float,
    book_spread_bps: float | None = None,
):
    """Baseline: fixed symmetric spread with inventory skew only.

    Uses the same book-anchored base as compute_ml_spreads so the Fixed vs ML
    comparison is structurally fair.  Both arms quote from the same floor;
    only the ML direction/volatility tilts differ.
    """
    spread_bps = config.synthetic_book_spread_bps if book_spread_bps is None else float(book_spread_bps)
    book_half_spread = (spread_bps / 2) / 10_000
    base = max(book_half_spread * 1.2, config.min_spread_pct / 100)
    inv_shift = current_inventory_pct * config.inventory_skew_factor * base

    bid_spread = base + inv_shift
    ask_spread = base - inv_shift

    min_s = config.min_spread_pct / 100
    max_s = config.max_spread_pct / 100
    bid_spread = np.clip(bid_spread, min_s, max_s)
    ask_spread = np.clip(ask_spread, min_s, max_s)

    return bid_spread, ask_spread


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _has_orderbook_data(row) -> bool:
    if "ob_has_data" not in row.index:
        return False
    return bool(_safe_float(row.get("ob_has_data"), 0.0) > 0)


def _historical_book_spread_bps(row, config: MMConfig) -> float:
    if _has_orderbook_data(row):
        spread_bps = _safe_float(row.get("ob_quoted_spread_bps"), 0.0)
        if spread_bps > 0:
            return spread_bps
    return float(config.synthetic_book_spread_bps)


def _orderbook_state_from_rows(row, prev_row=None) -> dict:
    if not _has_orderbook_data(row):
        return {
            "ob_quoted_spread_bps": 0.0,
            "ob_bid_ask_size_imbalance": 0.0,
            "ob_depth_imbalance_5": 0.0,
            "ob_depth_imbalance_20": 0.0,
            "ob_book_pressure": 0.5,
            "ob_weighted_mid_offset_bps": 0.0,
            "ob_pressure_velocity": 0.0,
            "ob_depth_imb_5_velocity": 0.0,
            "ob_has_data": False,
        }

    book_pressure = _safe_float(row.get("ob_book_pressure"), 0.5)
    depth_imb_5 = _safe_float(row.get("ob_depth_imbalance_5"), 0.0)
    prev_pressure = book_pressure
    prev_depth_imb_5 = depth_imb_5
    if prev_row is not None and _has_orderbook_data(prev_row):
        prev_pressure = _safe_float(prev_row.get("ob_book_pressure"), book_pressure)
        prev_depth_imb_5 = _safe_float(prev_row.get("ob_depth_imbalance_5"), depth_imb_5)

    return {
        "ob_quoted_spread_bps": _safe_float(row.get("ob_quoted_spread_bps"), 0.0),
        "ob_bid_ask_size_imbalance": _safe_float(row.get("ob_bid_ask_size_imbalance"), 0.0),
        "ob_depth_imbalance_5": depth_imb_5,
        "ob_depth_imbalance_20": _safe_float(row.get("ob_depth_imbalance_20"), 0.0),
        "ob_book_pressure": book_pressure,
        "ob_weighted_mid_offset_bps": float(np.clip(_safe_float(row.get("ob_weighted_mid_offset_bps"), 0.0), -20.0, 20.0)),
        "ob_pressure_velocity": float(np.clip(book_pressure - prev_pressure, -0.5, 0.5)),
        "ob_depth_imb_5_velocity": float(np.clip(depth_imb_5 - prev_depth_imb_5, -1.0, 1.0)),
        "ob_has_data": True,
    }


def _mtf_context_at(close: np.ndarray, i: int) -> dict:
    def _ret(lag: int) -> float:
        if i < lag or close[i - lag] == 0:
            return 0.0
        return float(close[i] / close[i - lag] - 1.0)

    ret_5m = _ret(1)
    ret_15m = _ret(3)
    ret_1h = _ret(12)
    signs = np.array([ret_5m, ret_15m, ret_1h], dtype=float)
    nonzero = np.abs(signs) > 1e-9
    if nonzero.any():
        pos = int(np.sum(signs[nonzero] > 0))
        neg = int(np.sum(signs[nonzero] < 0))
        trend_alignment = (max(pos, neg) - min(pos, neg)) / max(1, int(np.sum(nonzero)))
    else:
        trend_alignment = 0.0
    return {
        "ret_5m": ret_5m,
        "ret_15m": ret_15m,
        "ret_1h": ret_1h,
        "trend_alignment": float(trend_alignment),
    }


def _backtest_adverse_score(
    dir_proba: float,
    vol_pred: float,
    ob_state: dict,
    mtf_ctx: dict,
) -> float:
    direction = 1.0 if dir_proba >= 0.5 else -1.0
    depth_imb_5 = float(ob_state.get("ob_depth_imbalance_5", 0.0))
    book_pressure = float(ob_state.get("ob_book_pressure", 0.5))
    signed_flow = float(np.clip((book_pressure - 0.5) * 2.0, -1.0, 1.0))
    signed_depth = float(np.clip(depth_imb_5, -1.0, 1.0))
    flow_against = max(0.0, -direction * signed_flow)
    depth_against = max(0.0, -direction * signed_depth)
    market_toxicity = float(np.clip(0.55 * flow_against + 0.45 * depth_against, 0.0, 1.0))

    ret_5m = float(mtf_ctx.get("ret_5m", 0.0))
    ret_15m = float(mtf_ctx.get("ret_15m", 0.0))
    ret_1h = float(mtf_ctx.get("ret_1h", 0.0))
    trend_alignment = float(np.clip(mtf_ctx.get("trend_alignment", 0.0), 0.0, 1.0))
    trend_against = 1.0 if direction * np.sign(ret_15m + ret_1h) < 0 else 0.0
    micro_reversal = 1.0 if direction * ret_5m < 0 else 0.0
    trend_toxicity = float(np.clip(0.55 * trend_against + 0.45 * micro_reversal, 0.0, 1.0))

    spread_bps = float(ob_state.get("ob_quoted_spread_bps", 0.0))
    spread_stress = float(np.clip((spread_bps - 8.0) / 30.0, 0.0, 1.0))
    vol_ratio = float(np.clip(vol_pred / 0.005, 0.5, 3.0))
    vol_stress = float(np.clip((vol_ratio - 1.0) / 1.5, 0.0, 1.0))

    # No live execution table exists inside historical simulation, so the score
    # uses market/trend/vol components and leaves exec toxicity neutral.
    exec_toxicity = 0.0
    score = (
        0.32 * exec_toxicity
        + 0.26 * market_toxicity
        + 0.18 * spread_stress
        + 0.14 * trend_alignment * trend_toxicity
        + 0.10 * vol_stress
    )
    return float(np.clip(score, 0.0, 1.0))


def _apply_orderbook_overlays_fast(
    bid_spread: float,
    ask_spread: float,
    *,
    has_data: bool,
    spread_bps: float,
    depth_imb_5: float,
    book_pressure: float,
    wmid_offset_bps: float,
    pressure_velocity: float,
    min_spread_pct: float,
    max_spread_pct: float,
) -> tuple[float, float]:
    if not has_data:
        return bid_spread, ask_spread

    min_spread = min_spread_pct / 100
    max_spread = max_spread_pct / 100

    if spread_bps > 0:
        market_half_spread = (spread_bps / 10_000) / 2
        bid_spread = float(max(bid_spread, market_half_spread * 0.9))
        ask_spread = float(max(ask_spread, market_half_spread * 0.9))

    imb_shift = float(np.clip(depth_imb_5 * 0.10, -0.08, 0.08))
    bid_spread = float(np.clip(bid_spread * (1.0 - imb_shift), min_spread, max_spread))
    ask_spread = float(np.clip(ask_spread * (1.0 + imb_shift), min_spread, max_spread))

    pressure_shift = float(np.clip((book_pressure - 0.5) * 0.06, -0.04, 0.04))
    bid_spread = float(np.clip(bid_spread * (1.0 - pressure_shift), min_spread, max_spread))
    ask_spread = float(np.clip(ask_spread * (1.0 + pressure_shift), min_spread, max_spread))

    wmid_shift = float(np.clip(wmid_offset_bps / 200.0, -0.05, 0.05))
    bid_spread = float(np.clip(bid_spread * (1.0 - wmid_shift), min_spread, max_spread))
    ask_spread = float(np.clip(ask_spread * (1.0 + wmid_shift), min_spread, max_spread))

    if pressure_velocity > 0.05:
        widen = float(np.clip(pressure_velocity * 0.40, 0.0, 0.20))
        ask_spread = float(np.clip(ask_spread * (1.0 + widen), min_spread, max_spread))
    elif pressure_velocity < -0.05:
        widen = float(np.clip(-pressure_velocity * 0.40, 0.0, 0.20))
        bid_spread = float(np.clip(bid_spread * (1.0 + widen), min_spread, max_spread))

    return bid_spread, ask_spread


def _backtest_adverse_score_fast(
    dir_proba: float,
    vol_pred: float,
    *,
    spread_bps: float,
    depth_imb_5: float,
    book_pressure: float,
    ret_5m: float,
    ret_15m: float,
    ret_1h: float,
    trend_alignment: float,
) -> float:
    direction = 1.0 if dir_proba >= 0.5 else -1.0
    signed_flow = float(np.clip((book_pressure - 0.5) * 2.0, -1.0, 1.0))
    signed_depth = float(np.clip(depth_imb_5, -1.0, 1.0))
    flow_against = max(0.0, -direction * signed_flow)
    depth_against = max(0.0, -direction * signed_depth)
    market_toxicity = float(np.clip(0.55 * flow_against + 0.45 * depth_against, 0.0, 1.0))

    trend_against = 1.0 if direction * np.sign(ret_15m + ret_1h) < 0 else 0.0
    micro_reversal = 1.0 if direction * ret_5m < 0 else 0.0
    trend_toxicity = float(np.clip(0.55 * trend_against + 0.45 * micro_reversal, 0.0, 1.0))
    spread_stress = float(np.clip((spread_bps - 8.0) / 30.0, 0.0, 1.0))
    vol_ratio = float(np.clip(vol_pred / 0.005, 0.5, 3.0))
    vol_stress = float(np.clip((vol_ratio - 1.0) / 1.5, 0.0, 1.0))

    score = (
        0.26 * market_toxicity
        + 0.18 * spread_stress
        + 0.14 * float(np.clip(trend_alignment, 0.0, 1.0)) * trend_toxicity
        + 0.10 * vol_stress
    )
    return float(np.clip(score, 0.0, 1.0))


def _apply_as_inventory_skew(
    bid_spread: float,
    ask_spread: float,
    *,
    inventory_usd: float,
    vol_pred: float,
    config: MMConfig,
) -> tuple[float, float]:
    if config.max_inventory <= 0 or config.as_inventory_risk_aversion <= 0:
        return bid_spread, ask_spread
    q_signed = float(np.clip(inventory_usd / config.max_inventory, -1.0, 1.0))
    if abs(q_signed) < 1e-12:
        return bid_spread, ask_spread
    vol = float(np.clip(vol_pred, 0.001, 0.05))
    shift = float(np.clip(
        q_signed * config.as_inventory_risk_aversion * vol,
        -config.as_inventory_max_shift,
        config.as_inventory_max_shift,
    ))
    min_s = config.min_spread_pct / 100
    max_s = config.max_spread_pct / 100
    return (
        float(np.clip(bid_spread + shift, min_s, max_s)),
        float(np.clip(ask_spread - shift, min_s, max_s)),
    )


def _drawdown_size_scaler(pos: Position, config: MMConfig) -> float:
    if not pos.pnl_history:
        return 1.0
    peak = max(pos.pnl_history)
    if peak <= 0:
        return 1.0
    current = pos.pnl_history[-1]
    drawdown = max(0.0, (peak - current) / peak)
    max_dd = max(float(config.max_session_drawdown_pct), 0.0)
    if max_dd <= 0:
        return 1.0
    taper_start = max_dd * float(np.clip(config.dd_size_taper_start_ratio, 0.0, 1.0))
    if drawdown <= taper_start:
        return 1.0
    floor = float(np.clip(config.dd_size_floor, 0.0, 1.0))
    if drawdown >= max_dd or max_dd <= taper_start:
        return floor
    progress = (drawdown - taper_start) / (max_dd - taper_start)
    return float(np.clip(1.0 - progress * (1.0 - floor), floor, 1.0))


def _sample_fill_fraction(
    rng: np.random.Generator,
    *,
    order_price: float,
    candle_extreme: float,
    candle_range: float,
    mid_price: float,
    candle_volume: float,
    order_qty_base: float,
    vol_pred: float,
    config: MMConfig,
) -> float:
    if candle_range < 1e-10 or mid_price <= 0 or order_qty_base <= 0:
        return 0.0
    penetration = float(np.clip(abs(candle_extreme - order_price) / candle_range, 0.0, 1.0))
    if penetration <= 0:
        return 0.0

    volume_base = max(float(candle_volume or 0.0), order_qty_base)
    candle_vol_pct = candle_range / mid_price
    vol_ref = max(abs(float(vol_pred or 0.005)), 0.0005)
    late_haircut = 1.0 - (
        config.late_fill_volatility_penalty
        * max(0.0, candle_vol_pct / vol_ref - 1.0)
        * (1.0 - penetration)
    )
    late_haircut = float(np.clip(late_haircut, 0.35, 1.0))

    marketable_volume = volume_base * penetration * late_haircut * config.queue_volume_fraction
    queue_position = rng.uniform(0.0, volume_base)
    if marketable_volume <= queue_position:
        return 0.0

    fillable_fraction = float(np.clip((marketable_volume - queue_position) / order_qty_base, 0.0, 1.0))
    if fillable_fraction <= 0:
        return 0.0

    alpha = max(float(config.partial_fill_alpha), 0.1)
    beta = max(float(config.partial_fill_beta), 0.1)
    partial_fraction = float(np.clip(rng.beta(alpha, beta), 0.02, 1.0))
    return float(np.clip(min(fillable_fraction, partial_fraction), 0.0, 1.0))


def _execution_penalty_fraction(
    *,
    side: int,
    future_return: float,
    latency_return: float,
    config: MMConfig,
) -> float:
    adverse_penalty = config.adverse_selection_penalty_alpha * max(0.0, -side * future_return)
    latency_penalty = config.latency_penalty_weight * abs(latency_return)
    return float(np.clip(adverse_penalty + latency_penalty, 0.0, 0.01))


def _position_metrics(pos: Position, label: str) -> tuple[dict, np.ndarray, float]:
    pnl = np.array(pos.pnl_history)
    returns = np.diff(pnl) / pnl[:-1]
    returns = returns[np.isfinite(returns)]

    total_return = (pnl[-1] / pos.initial_cash - 1) * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(288 * 365)
    max_dd = np.min(pnl / np.maximum.accumulate(pnl) - 1) * 100
    avg_inventory = np.mean(np.abs(pos.inventory_history))
    avg_fill_ratio = pos.fill_ratio_sum / pos.fill_events if pos.fill_events > 0 else 0.0

    metrics = {
        "label": label,
        "total_return_pct": round(total_return, 3),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 3),
        "total_trades": pos.trades,
        "buys": pos.buys,
        "sells": pos.sells,
        "total_fees_usd": round(pos.total_fees, 2),
        "execution_penalty_usd": round(pos.total_execution_penalty, 2),
        "avg_fill_ratio": round(avg_fill_ratio, 3),
        "final_portfolio_usd": round(pnl[-1], 2),
        "avg_inventory_usd": round(avg_inventory, 2),
    }
    return metrics, pnl, avg_inventory


def _load_reference_frame() -> pd.DataFrame | None:
    for ref_file in (
        BINANCE_DATA_DIR / "ETH_USD-5m.feather",
        BINANCE_DATA_DIR / "ETH_USDT-5m.feather",
    ):
        if ref_file.exists():
            return pf.read_feather(str(ref_file), memory_map=False)
    return None


def _load_prediction_stack() -> dict:
    dir_model, dir3_model, vol_model, calibrator, metadata = load_models()
    feature_cols = metadata["feature_columns"]
    nn_model, nn_meta = load_neural_model(len(feature_cols), LATEST_MODEL_DIR)
    dir_model_fast, dir_model_mid = load_multi_horizon_models(LATEST_MODEL_DIR)
    meta_bundle = load_meta_ensemble(LATEST_MODEL_DIR)
    regime_bundle = load_regime_model(LATEST_MODEL_DIR)
    return {
        "dir_model": dir_model,
        "dir3_model": dir3_model,
        "vol_model": vol_model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "nn_model": nn_model,
        "nn_meta": nn_meta,
        "dir_model_fast": dir_model_fast,
        "dir_model_mid": dir_model_mid,
        "meta_bundle": meta_bundle,
        "regime_bundle": regime_bundle,
    }


def _predict_with_stack(feat_df: pd.DataFrame, stack: dict) -> pd.DataFrame:
    feat_df = apply_regime_model(feat_df, stack.get("regime_bundle"))
    return predict_direction_series(
        feat_df,
        stack["feature_cols"],
        stack["dir_model"],
        stack["dir3_model"],
        stack["vol_model"],
        calibrator=stack["calibrator"],
        nn_model=stack["nn_model"],
        nn_meta=stack["nn_meta"],
        dir_model_fast=stack["dir_model_fast"],
        dir_model_mid=stack["dir_model_mid"],
        meta_bundle=stack["meta_bundle"],
    )


def simulate_market_making(df, spreads_func, config: MMConfig, label="Strategy", seed: int = 42):
    """
    Run market-making simulation on historical data.

    Parameters
    ----------
    df : DataFrame with columns [close, high, low, bid_spread, ask_spread]
    spreads_func : callable returning (bid_spread, ask_spread) series or None if pre-computed
    config : MMConfig

    Returns
    -------
    dict with performance metrics
    """
    rng = np.random.default_rng(seed)
    pos = Position()
    mid_prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values if "volume" in df.columns else np.zeros(len(df))
    vol_preds = df["predicted_volatility"].values if "predicted_volatility" in df.columns else np.full(len(df), 0.005)
    bid_spreads = df["bid_spread"].values
    ask_spreads = df["ask_spread"].values
    future_returns = np.zeros(len(df))
    if len(df) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            future_returns[:-1] = np.where(mid_prices[:-1] > 0, mid_prices[1:] / mid_prices[:-1] - 1.0, 0.0)
    future_returns = np.nan_to_num(future_returns, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(len(df)):
        mid = mid_prices[i]
        h = highs[i]
        l = lows[i]

        bid_price = mid * (1 - bid_spreads[i])
        ask_price = mid * (1 + ask_spreads[i])

        inventory_usd = pos.inventory_base * mid
        inventory_pct = inventory_usd / config.max_inventory if config.max_inventory > 0 else 0

        # Fill probability model: price touching the level doesn't guarantee fill.
        # Probability scales with how far price penetrated past the order level.
        # This models queue priority — deeper penetration = more likely filled.
        candle_range = h - l if h > l else 1e-10

        # Check if bid order would fill (price went low enough)
        can_buy = inventory_usd < config.max_inventory and pos.cash >= config.order_amount_usd
        if can_buy and l <= bid_price:
            requested_qty = config.order_amount_usd / bid_price
            fill_ratio = _sample_fill_fraction(
                rng,
                order_price=bid_price,
                candle_extreme=l,
                candle_range=candle_range,
                mid_price=mid,
                candle_volume=volumes[i],
                order_qty_base=requested_qty,
                vol_pred=vol_preds[i],
                config=config,
            )
            if fill_ratio > 0:
                notional = config.order_amount_usd * fill_ratio
                qty = notional / bid_price
                fee = notional * config.fee_pct / 100
                penalty = notional * _execution_penalty_fraction(
                    side=1,
                    future_return=float(future_returns[i]),
                    latency_return=float(future_returns[i]),
                    config=config,
                )
                pos.inventory_base += qty
                pos.cash -= notional + fee + penalty
                pos.total_fees += fee
                pos.total_execution_penalty += penalty
                pos.trades += 1
                pos.buys += 1
                pos.fill_events += 1
                pos.fill_ratio_sum += fill_ratio

        # Check if ask order would fill (price went high enough)
        can_sell = pos.inventory_base * mid > config.order_amount_usd * 0.5
        if can_sell and h >= ask_price:
            requested_qty = min(config.order_amount_usd / ask_price, pos.inventory_base)
            fill_ratio = _sample_fill_fraction(
                rng,
                order_price=ask_price,
                candle_extreme=h,
                candle_range=candle_range,
                mid_price=mid,
                candle_volume=volumes[i],
                order_qty_base=requested_qty,
                vol_pred=vol_preds[i],
                config=config,
            )
            if fill_ratio > 0:
                qty = config.order_amount_usd * fill_ratio / ask_price
                qty = min(qty, pos.inventory_base)
                revenue = qty * ask_price
                fee = revenue * config.fee_pct / 100
                penalty = revenue * _execution_penalty_fraction(
                    side=-1,
                    future_return=float(future_returns[i]),
                    latency_return=float(future_returns[i]),
                    config=config,
                )
                pos.inventory_base -= qty
                pos.cash += revenue - fee - penalty
                pos.total_fees += fee
                pos.total_execution_penalty += penalty
                pos.trades += 1
                pos.sells += 1
                pos.fill_events += 1
                pos.fill_ratio_sum += min(fill_ratio, 1.0)

        # Track portfolio value
        portfolio_value = pos.cash + pos.inventory_base * mid
        pos.pnl_history.append(portfolio_value)
        pos.inventory_history.append(pos.inventory_base * mid)

    metrics, pnl, _ = _position_metrics(pos, label)
    return metrics, pnl, pos.inventory_history


def _apply_backtest_mtf_gate(
    dir_proba: np.ndarray, test_df: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    """Mirror signal_server._mtf_trend_gate onto a historical candle frame.

    The live signal server neutralizes the model's direction probability to 0.5
    whenever (a) 5m/15m/1h returns all agree on a direction, (b) that agreed
    direction opposes the model, and (c) either the 5m return also opposes the
    model OR orderbook toxicity is elevated. This helper rebuilds the same
    context from the backtest's `test_df` and applies the gate candle-by-candle
    so validation backtests report spreads/fills comparable to production.

    Expects `test_df` to have `close` and optionally `ob_depth_imbalance_5`,
    `ob_book_pressure`. Any missing OB columns default to neutral (0.0 / 0.5),
    which effectively disables the market-toxicity confirmation path — the
    micro-reversal path still gates predictions.
    """
    n = len(test_df)
    gated = dir_proba.copy().astype(float)
    if n < 13:  # need at least 12 lags for ret_1h
        return gated, {"fired": 0, "total": n}

    close = test_df["close"].astype(float).to_numpy()
    # Safe rolling pct-change: (close[i] / close[i-lag]) - 1, NaN where undefined
    def _pct_change(lag: int) -> np.ndarray:
        out = np.full(n, np.nan, dtype=float)
        if lag >= n:
            return out
        prev = close[:-lag]
        cur = close[lag:]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lag:] = np.where(prev != 0, (cur / prev) - 1.0, np.nan)
        return out

    ret_5m = _pct_change(1)
    ret_15m = _pct_change(3)
    ret_1h = _pct_change(12)

    # Orderbook columns are optional — fall back to neutral values.
    if "ob_depth_imbalance_5" in test_df.columns:
        depth_imb_5 = test_df["ob_depth_imbalance_5"].astype(float).fillna(0.0).to_numpy()
    else:
        depth_imb_5 = np.zeros(n, dtype=float)
    if "ob_book_pressure" in test_df.columns:
        book_pressure = test_df["ob_book_pressure"].astype(float).fillna(0.5).to_numpy()
    else:
        book_pressure = np.full(n, 0.5, dtype=float)

    fired = 0
    for i in range(n):
        r5, r15, r1h = ret_5m[i], ret_15m[i], ret_1h[i]
        if np.isnan(r5) or np.isnan(r15) or np.isnan(r1h):
            continue
        signs = np.array([r5, r15, r1h], dtype=float)
        nonzero = np.abs(signs) > 1e-9
        if not nonzero.any():
            continue
        pos = int(np.sum(signs[nonzero] > 0))
        neg = int(np.sum(signs[nonzero] < 0))
        alignment = (max(pos, neg) - min(pos, neg)) / max(1, int(np.sum(nonzero)))
        if alignment < 1.0:
            continue
        mtf_dir = 1.0 if (r15 + r1h) > 0 else -1.0
        model_dir = 1.0 if gated[i] >= 0.5 else -1.0
        if model_dir == mtf_dir:
            continue
        # Model disagrees with aligned trend — require secondary confirmation
        signed_flow = float(np.clip((book_pressure[i] - 0.5) * 2.0, -1.0, 1.0))
        signed_depth = float(np.clip(depth_imb_5[i], -1.0, 1.0))
        flow_against = max(0.0, -model_dir * signed_flow)
        depth_against = max(0.0, -model_dir * signed_depth)
        market_toxicity = float(np.clip(0.55 * flow_against + 0.45 * depth_against, 0.0, 1.0))
        micro_reversal = (model_dir * r5) < 0
        if micro_reversal or market_toxicity > 0.35:
            gated[i] = 0.5
            fired += 1

    return gated, {"fired": fired, "total": n}


def _apply_backtest_conformal_gate(
    dir_proba: np.ndarray,
    pred_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Neutralize predictions whose conformal band still straddles 0.5."""
    if "conformal_enabled" not in pred_frame.columns:
        return dir_proba, {"fired": 0, "total": int(len(dir_proba)), "enabled": False}
    if float(pd.to_numeric(pred_frame["conformal_enabled"], errors="coerce").fillna(0.0).max()) <= 0.0:
        return dir_proba, {"fired": 0, "total": int(len(dir_proba)), "enabled": False}
    if "direction_lower_bound" not in pred_frame.columns or "direction_upper_bound" not in pred_frame.columns:
        return dir_proba, {"fired": 0, "total": int(len(dir_proba)), "enabled": False}

    lower = pd.to_numeric(pred_frame["direction_lower_bound"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    upper = pd.to_numeric(pred_frame["direction_upper_bound"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    gated = np.asarray(dir_proba, dtype=float).copy()
    mask = (lower <= 0.5) & (upper >= 0.5)
    gated[mask] = 0.5
    return gated, {"fired": int(mask.sum()), "total": int(len(gated)), "enabled": True}


def run_backtest():
    """Full backtest comparing ML market maker vs fixed-spread baseline."""
    print("=" * 70)
    print("MARKET-MAKING BACKTEST: ML-Enhanced vs Fixed Spread")
    print("=" * 70)

    # Load data
    df = pf.read_feather(str(DATA_DIR / "ETH_USDT-5m.feather"), memory_map=False)
    btc_df = pf.read_feather(str(DATA_DIR / "BTC_USDT-5m.feather"), memory_map=False)
    sol_file = DATA_DIR / "SOL_USDT-5m.feather"
    sol_df = pf.read_feather(str(sol_file), memory_map=False) if sol_file.exists() else None

    print(f"Data: {len(df)} rows")

    # Load models via shared loader (validates feature counts)
    print("Loading ML models...")
    stack = _load_prediction_stack()
    feature_cols = stack["feature_cols"]

    # Compute features
    print("Computing features...")
    execution_df = load_or_build_execution_features(candle_dates=df["date"], prefer_cached=True)
    orderbook_df = load_or_build_orderbook_features(candle_dates=df["date"], prefer_cached=True)
    feat_df = compute_features(
        df,
        btc_df,
        sol_df=sol_df,
        execution_df=execution_df,
        orderbook_df=orderbook_df,
        binance_df=_load_reference_frame(),
    )
    feat_df = compute_labels(feat_df, horizon=6)
    feat_df = apply_regime_model(feat_df, stack.get("regime_bundle"))
    feat_df = feat_df.dropna(subset=feature_cols + ["direction"]).reset_index(drop=True)

    # Use only test period (last 15% of data)
    n = len(feat_df)
    test_start = int(n * 0.85)
    test_df = feat_df.iloc[test_start:].copy().reset_index(drop=True)
    print(f"Test period: {len(test_df)} candles ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")
    # Generate ML predictions
    print("Generating ML predictions...")
    pred_frame = _predict_with_stack(test_df, stack)
    dir_proba = pred_frame["direction_probability"].to_numpy()
    vol_pred = pred_frame["predicted_volatility"].to_numpy()
    test_df["predicted_volatility"] = vol_pred
    model_post_calibrated = bool(
        "post_calibrated" in pred_frame.columns
        and float(pred_frame["post_calibrated"].max()) > 0.0
    )
    print(f"Prediction mode: {pred_frame['ensemble_mode'].iloc[-1]}")

    # Apply the live runtime MTF trend gate to the raw predictions so backtest
    # validation reflects production filtering. Without this, backtest Sharpe
    # dramatically overstates live performance because live neutralizes any
    # high-confidence prediction that disagrees with an aligned MTF trend.
    dir_proba, mtf_gate_stats = _apply_backtest_mtf_gate(dir_proba, test_df)
    if mtf_gate_stats["total"] > 0:
        gate_pct = 100.0 * mtf_gate_stats["fired"] / mtf_gate_stats["total"]
        print(
            f"MTF trend gate (backtest parity): "
            f"fired on {mtf_gate_stats['fired']}/{mtf_gate_stats['total']} "
            f"({gate_pct:.1f}%) candles — neutralized to 0.5"
        )

    dir_proba, conformal_gate_stats = _apply_backtest_conformal_gate(dir_proba, pred_frame)
    if conformal_gate_stats["enabled"] and conformal_gate_stats["total"] > 0:
        gate_pct = 100.0 * conformal_gate_stats["fired"] / conformal_gate_stats["total"]
        print(
            f"Conformal gate (backtest parity): "
            f"fired on {conformal_gate_stats['fired']}/{conformal_gate_stats['total']} "
            f"({gate_pct:.1f}%) candles â€” neutralized to 0.5"
        )

    config = MMConfig()
    apply_regime_gate_env_override(config, verbose=True)

    print(f"Fee: {config.fee_pct:.2f}%/side ({config.fee_pct * 2:.2f}% round-trip)")
    print(f"Spread floor: {config.min_spread_pct:.2f}% ({config.min_spread_pct * 100:.0f} bps)")
    print(f"Base spread: {config.base_spread_pct:.2f}% ({config.base_spread_pct * 100:.0f} bps)")

    ml_metrics, ml_pnl, ml_inv, ml_spread_stats = simulate_ml_variant(
        test_df,
        dir_proba,
        vol_pred,
        config,
        "ML Market Maker",
        use_inventory_target=False,
        use_multilevel=config.order_levels > 1,
        use_confidence_sizing=True,
    )

    # === Baseline fixed spread ===
    fixed_bid_spreads = np.zeros(len(test_df))
    fixed_ask_spreads = np.zeros(len(test_df))

    for i in range(len(test_df)):
        book_spread_bps = _historical_book_spread_bps(test_df.iloc[i], config)
        bid_s, ask_s = compute_fixed_spreads(config, 0.0, book_spread_bps)
        fixed_bid_spreads[i] = bid_s
        fixed_ask_spreads[i] = ask_s

    test_df["bid_spread"] = fixed_bid_spreads
    test_df["ask_spread"] = fixed_ask_spreads

    fixed_metrics, fixed_pnl, fixed_inv = simulate_market_making(test_df, None, config, "Fixed Spread")

    # === Print results ===
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    header = f"{'Metric':<30} {'ML Market Maker':>18} {'Fixed Spread':>18}"
    print(header)
    print("-" * 70)

    for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades",
                 "buys", "sells", "total_fees_usd", "execution_penalty_usd", "avg_fill_ratio",
                 "final_portfolio_usd", "avg_inventory_usd"]:
        ml_val = ml_metrics[key]
        fixed_val = fixed_metrics[key]
        print(f"{key:<30} {ml_val:>18} {fixed_val:>18}")

    print("=" * 70)

    # ML prediction accuracy on test set
    actual_dir = np.asarray(test_df["direction"].to_numpy(), dtype=int)
    pred_dir = np.asarray((dir_proba > 0.5).astype(int), dtype=int)
    from sklearn.metrics import accuracy_score, roc_auc_score
    print(f"\nML Direction Accuracy (test): {accuracy_score(actual_dir, pred_dir):.4f}")
    print(f"ML Direction AUC (test):      {roc_auc_score(actual_dir, dir_proba):.4f}")

    # Spread statistics
    print(f"\nML Avg Bid Spread: {ml_spread_stats['avg_bid']:.4f}%")
    print(f"ML Avg Ask Spread: {ml_spread_stats['avg_ask']:.4f}%")
    print(f"Fixed Spread:      {config.base_spread_pct:.4f}%")

    # Per-regime trade economics (diagnostic for regime-gate calibration)
    regime_breakdown = ml_metrics.get("regime_breakdown") or {}
    if regime_breakdown:
        print("\nPer-regime trade economics (ML arm):")
        print(
            f"  {'rid':>3}  {'candles':>7}  {'stress':>6}  {'trend':>6}  "
            f"{'trades':>6}  {'B/S':>7}  {'rate':>6}  {'acc':>5}  {'ret_bps':>9}"
        )
        for rid in sorted(regime_breakdown, key=lambda k: int(k)):
            r = regime_breakdown[rid]
            acc = r["model_accuracy"]
            acc_str = f"{acc:.3f}" if acc is not None else " n/a "
            print(
                f"  {r['regime_id']:>3}  {r['candles']:>7}  "
                f"{r['mean_stress']:>6.2f}  {r['mean_trend_score']:>6.2f}  "
                f"{r['trades']:>6}  {r['buys']:>3}/{r['sells']:<3}  "
                f"{r['trade_rate']:>6.4f}  {acc_str:>5}  "
                f"{r['avg_signed_return_bps']:>9.2f}"
            )

    # Save results
    results = {
        "ml_metrics": ml_metrics,
        "fixed_metrics": fixed_metrics,
        "test_period": {
            "start": str(test_df["date"].iloc[0]),
            "end": str(test_df["date"].iloc[-1]),
            "candles": len(test_df),
        },
        "ml_spread_stats": ml_spread_stats,
        "execution_profile": {
            "order_levels": int(config.order_levels),
            "order_level_spread_step_pct": float(config.order_level_spread_step_pct),
            "min_order_amount_usd": float(config.min_order_amount_usd),
            "max_order_amount_usd": float(config.max_order_amount_usd),
            "uses_confidence_sizing": True,
            "model_post_calibrated": model_post_calibrated,
        },
    }

    results_path = BACKTEST_RESULTS_FILE
    _write_json_report(results_path, results)

    return results


def compute_ml_spreads_with_target(
    dir_prob,
    vol_pred,
    config: MMConfig,
    current_base_pct: float,
    book_spread_bps: float | None = None,
):
    """Compute spreads with a softened inventory-target skew around a no-action tolerance band."""
    target = float(np.clip(config.inventory_target_base_pct, 0.0, 1.0))
    tol = float(np.clip(config.inventory_target_tolerance_pct, 0.0, 0.45))

    deviation = current_base_pct - target
    if abs(deviation) <= tol:
        inv_signal = 0.0
    else:
        # Scale only the excess deviation beyond tolerance into [-1, 1].
        remaining = max(1e-6, 1.0 - tol)
        inv_signal = np.sign(deviation) * min(1.0, (abs(deviation) - tol) / remaining)

    inv_signal *= float(np.clip(config.inventory_target_skew_strength, 0.0, 1.0))
    return compute_ml_spreads(dir_prob, vol_pred, config, inv_signal, book_spread_bps)


def calibrate_direction_probabilities(raw_val, y_val, raw_test):
    """Calibrate direction probabilities using isotonic regression on validation predictions."""
    if not SKLEARN_CAL_AVAILABLE or IsotonicRegression is None:
        return raw_test, False
    if len(np.unique(y_val)) < 2:
        return raw_test, False

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)
    return calibrator.predict(raw_test), True


def confidence_to_order_sizes(dir_prob: float, config: MMConfig):
    """Map direction probability to asymmetric buy/sell order sizes in USD."""
    confidence = float(np.clip(abs(dir_prob - 0.5) * 2.0, 0.0, 1.0))
    # Stronger confidence scales one side up and the other down.
    directional_tilt = (dir_prob - 0.5) * 2.0
    base_amt = config.order_amount_usd

    buy_mult = np.clip(1.0 + 0.5 * directional_tilt + 0.25 * confidence, 0.5, 1.75)
    sell_mult = np.clip(1.0 - 0.5 * directional_tilt + 0.25 * confidence, 0.5, 1.75)

    buy_usd = float(np.clip(base_amt * buy_mult, config.min_order_amount_usd, config.max_order_amount_usd))
    sell_usd = float(np.clip(base_amt * sell_mult, config.min_order_amount_usd, config.max_order_amount_usd))
    return buy_usd, sell_usd


def simulate_ml_variant(
    df,
    dir_proba,
    vol_pred,
    config: MMConfig,
    label="ML Variant",
    use_inventory_target=False,
    use_multilevel=False,
    use_confidence_sizing=False,
    seed: int = 42,
):
    """Simulate ML strategy variants with optional target-skew, levels, and confidence sizing."""
    rng = np.random.default_rng(seed)
    pos = Position()
    mid_prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values if "volume" in df.columns else np.zeros(len(df))
    future_returns = np.zeros(len(df))
    if len(df) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            future_returns[:-1] = np.where(mid_prices[:-1] > 0, mid_prices[1:] / mid_prices[:-1] - 1.0, 0.0)
    future_returns = np.nan_to_num(future_returns, nan=0.0, posinf=0.0, neginf=0.0)

    def _array(col: str, default: float) -> np.ndarray:
        if col not in df.columns:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)

    ob_has_data_arr = _array("ob_has_data", 0.0) > 0
    ob_spread_bps_arr = _array("ob_quoted_spread_bps", 0.0)
    ob_bid_ask_imb_arr = _array("ob_bid_ask_size_imbalance", 0.0)
    ob_depth_imb_5_arr = _array("ob_depth_imbalance_5", 0.0)
    ob_depth_imb_20_arr = _array("ob_depth_imbalance_20", 0.0)
    ob_pressure_arr = _array("ob_book_pressure", 0.5)
    ob_wmid_offset_arr = np.clip(_array("ob_weighted_mid_offset_bps", 0.0), -20.0, 20.0)
    ob_pressure_velocity_arr = np.zeros(len(df), dtype=float)
    ob_depth_velocity_arr = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        valid_pressure = ob_has_data_arr[1:] & ob_has_data_arr[:-1]
        ob_pressure_velocity_arr[1:] = np.where(
            valid_pressure,
            np.clip(ob_pressure_arr[1:] - ob_pressure_arr[:-1], -0.5, 0.5),
            0.0,
        )
        ob_depth_velocity_arr[1:] = np.where(
            valid_pressure,
            np.clip(ob_depth_imb_5_arr[1:] - ob_depth_imb_5_arr[:-1], -1.0, 1.0),
            0.0,
        )
    book_spread_bps_arr = np.where(
        ob_has_data_arr & (ob_spread_bps_arr > 0),
        ob_spread_bps_arr,
        float(config.synthetic_book_spread_bps),
    )

    def _pct_change_array(lag: int) -> np.ndarray:
        out = np.zeros(len(df), dtype=float)
        if lag >= len(df):
            return out
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lag:] = np.where(mid_prices[:-lag] > 0, mid_prices[lag:] / mid_prices[:-lag] - 1.0, 0.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    ret_5m_arr = _pct_change_array(1)
    ret_15m_arr = _pct_change_array(3)
    ret_1h_arr = _pct_change_array(12)
    signs = np.vstack([ret_5m_arr, ret_15m_arr, ret_1h_arr])
    nonzero = np.abs(signs) > 1e-9
    pos_counts = np.sum((signs > 0) & nonzero, axis=0)
    neg_counts = np.sum((signs < 0) & nonzero, axis=0)
    nonzero_counts = np.maximum(1, np.sum(nonzero, axis=0))
    trend_alignment_arr = np.where(
        np.sum(nonzero, axis=0) > 0,
        (np.maximum(pos_counts, neg_counts) - np.minimum(pos_counts, neg_counts)) / nonzero_counts,
        0.0,
    )

    levels = config.order_levels if use_multilevel else 1
    levels = max(1, int(levels))
    level_step = config.order_level_spread_step_pct / 100

    top_bid_spreads = np.zeros(len(df))
    top_ask_spreads = np.zeros(len(df))

    def _regime_array(col: str, default: float) -> np.ndarray:
        if col not in df.columns:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)

    if "market_regime_id" in df.columns:
        regime_id_arr = (
            pd.to_numeric(df["market_regime_id"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )
    else:
        regime_id_arr = np.full(len(df), -1, dtype=int)
    regime_stress_arr = _regime_array("market_regime_stress", 0.0)
    regime_trend_arr = _regime_array("market_regime_trend_score", 0.0)
    regime_breakdown: dict[int, dict[str, float]] = {}

    def _regime_bucket(rid: int) -> dict:
        bucket = regime_breakdown.get(rid)
        if bucket is None:
            bucket = {
                "regime_id": int(rid),
                "candles": 0,
                "trades": 0,
                "buys": 0,
                "sells": 0,
                "notional_usd": 0.0,
                "fees_usd": 0.0,
                "penalty_usd": 0.0,
                "_signed_return_notional": 0.0,
                "_stress_sum": 0.0,
                "_trend_sum": 0.0,
                "_dir_proba_sum": 0.0,
                "_dir_correct": 0,
                "_dir_observed": 0,
            }
            regime_breakdown[rid] = bucket
        return bucket

    for i in range(len(df)):
        mid = mid_prices[i]
        h = highs[i]
        l = lows[i]
        regime_id_i = int(regime_id_arr[i])
        regime_bucket = _regime_bucket(regime_id_i)
        regime_bucket["candles"] += 1
        regime_bucket["_stress_sum"] += float(regime_stress_arr[i])
        regime_bucket["_trend_sum"] += float(regime_trend_arr[i])
        dir_proba_i = float(dir_proba[i])
        regime_bucket["_dir_proba_sum"] += dir_proba_i
        # Model "right" if dir_proba > 0.5 and price rose, or dir_proba < 0.5 and price fell.
        # Skip neutral predictions (after MTF/conformal gating, those equal 0.5 exactly).
        future_ret_i = float(future_returns[i])
        if abs(dir_proba_i - 0.5) > 1e-6 and abs(future_ret_i) > 1e-9:
            predicted_up = dir_proba_i > 0.5
            actual_up = future_ret_i > 0
            regime_bucket["_dir_observed"] += 1
            if predicted_up == actual_up:
                regime_bucket["_dir_correct"] += 1
        book_spread_bps = float(book_spread_bps_arr[i])
        has_ob = bool(ob_has_data_arr[i])
        ob_spread_bps = float(ob_spread_bps_arr[i]) if has_ob else 0.0
        ob_depth_imb_5 = float(ob_depth_imb_5_arr[i]) if has_ob else 0.0
        ob_pressure = float(ob_pressure_arr[i]) if has_ob else 0.5
        ob_wmid_offset = float(ob_wmid_offset_arr[i]) if has_ob else 0.0
        ob_pressure_velocity = float(ob_pressure_velocity_arr[i]) if has_ob else 0.0

        inventory_usd = pos.inventory_base * mid
        portfolio_value = max(pos.cash + inventory_usd, 1e-9)
        current_base_pct = float(np.clip(inventory_usd / portfolio_value, 0.0, 1.0))
        inv_pct = inventory_usd / config.max_inventory if config.max_inventory > 0 else 0

        if use_inventory_target:
            bid_spread, ask_spread = compute_ml_spreads_with_target(
                dir_proba[i], vol_pred[i], config, current_base_pct, book_spread_bps
            )
        else:
            bid_spread, ask_spread = compute_ml_spreads(
                dir_proba[i], vol_pred[i], config, inv_pct, book_spread_bps
            )
        bid_spread, ask_spread = _apply_orderbook_overlays_fast(
            bid_spread,
            ask_spread,
            has_data=has_ob,
            spread_bps=ob_spread_bps,
            depth_imb_5=ob_depth_imb_5,
            book_pressure=ob_pressure,
            wmid_offset_bps=ob_wmid_offset,
            pressure_velocity=ob_pressure_velocity,
            min_spread_pct=config.min_spread_pct,
            max_spread_pct=config.max_spread_pct,
        )
        bid_spread, ask_spread = _apply_as_inventory_skew(
            bid_spread,
            ask_spread,
            inventory_usd=inventory_usd,
            vol_pred=vol_pred[i],
            config=config,
        )

        top_bid_spreads[i] = bid_spread
        top_ask_spreads[i] = ask_spread

        if use_confidence_sizing:
            buy_order_usd, sell_order_usd = confidence_to_order_sizes(float(dir_proba[i]), config)
        else:
            buy_order_usd = float(np.clip(config.order_amount_usd, config.min_order_amount_usd, config.max_order_amount_usd))
            sell_order_usd = float(np.clip(config.order_amount_usd, config.min_order_amount_usd, config.max_order_amount_usd))

        adverse_score = _backtest_adverse_score_fast(
            float(dir_proba[i]),
            float(vol_pred[i]),
            spread_bps=ob_spread_bps,
            depth_imb_5=ob_depth_imb_5,
            book_pressure=ob_pressure,
            ret_5m=float(ret_5m_arr[i]),
            ret_15m=float(ret_15m_arr[i]),
            ret_1h=float(ret_1h_arr[i]),
            trend_alignment=float(trend_alignment_arr[i]),
        )
        size_cap = apply_adverse_size_cap(1.0, adverse_score)
        dd_scaler = _drawdown_size_scaler(pos, config)
        regime_scaler = 1.0
        if (
            config.regime_gate_enabled
            and regime_id_i >= 0
            and regime_bucket["trades"] >= config.regime_gate_min_trades
        ):
            notional_so_far = float(regime_bucket["notional_usd"])
            if notional_so_far > 1e-9:
                avg_bps = (
                    float(regime_bucket["_signed_return_notional"])
                    / notional_so_far
                    * 10_000.0
                )
                if avg_bps < config.regime_gate_threshold_bps:
                    regime_scaler = float(
                        np.clip(config.regime_gate_size_suppression, 0.0, 1.0)
                    )
        size_scaler = float(np.clip(size_cap * dd_scaler * regime_scaler, 0.0, 1.0))
        buy_order_usd *= size_scaler
        sell_order_usd *= size_scaler

        buy_per_level = buy_order_usd / levels
        sell_per_level = sell_order_usd / levels

        # Evaluate each quote level from inside-out.
        candle_range = h - l if h > l else 1e-10
        for level in range(levels):
            lvl_bid = bid_spread + level * level_step
            lvl_ask = ask_spread + level * level_step
            bid_price = mid * (1 - lvl_bid)
            ask_price = mid * (1 + lvl_ask)

            current_inv_usd = pos.inventory_base * mid
            can_buy = current_inv_usd < config.max_inventory and pos.cash >= buy_per_level
            if can_buy and l <= bid_price:
                requested_qty = buy_per_level / bid_price if bid_price > 0 else 0.0
                fill_ratio = _sample_fill_fraction(
                    rng,
                    order_price=bid_price,
                    candle_extreme=l,
                    candle_range=candle_range,
                    mid_price=mid,
                    candle_volume=volumes[i],
                    order_qty_base=requested_qty,
                    vol_pred=vol_pred[i],
                    config=config,
                )
                if fill_ratio > 0:
                    notional = buy_per_level * fill_ratio
                    qty = notional / bid_price
                    fee = notional * config.fee_pct / 100
                    penalty = notional * _execution_penalty_fraction(
                        side=1,
                        future_return=float(future_returns[i]),
                        latency_return=float(future_returns[i]),
                        config=config,
                    )
                    pos.inventory_base += qty
                    pos.cash -= notional + fee + penalty
                    pos.total_fees += fee
                    pos.total_execution_penalty += penalty
                    pos.trades += 1
                    pos.buys += 1
                    pos.fill_events += 1
                    pos.fill_ratio_sum += fill_ratio
                    regime_bucket["trades"] += 1
                    regime_bucket["buys"] += 1
                    regime_bucket["notional_usd"] += float(notional)
                    regime_bucket["fees_usd"] += float(fee)
                    regime_bucket["penalty_usd"] += float(penalty)
                    regime_bucket["_signed_return_notional"] += (
                        float(future_returns[i]) * float(notional)
                    )

            can_sell = pos.inventory_base * mid > sell_per_level * 0.5
            if can_sell and h >= ask_price:
                requested_qty = min(sell_per_level / ask_price if ask_price > 0 else 0.0, pos.inventory_base)
                fill_ratio = _sample_fill_fraction(
                    rng,
                    order_price=ask_price,
                    candle_extreme=h,
                    candle_range=candle_range,
                    mid_price=mid,
                    candle_volume=volumes[i],
                    order_qty_base=requested_qty,
                    vol_pred=vol_pred[i],
                    config=config,
                )
                if fill_ratio > 0:
                    qty = min(sell_per_level * fill_ratio / ask_price, pos.inventory_base)
                    revenue = qty * ask_price
                    fee = revenue * config.fee_pct / 100
                    penalty = revenue * _execution_penalty_fraction(
                        side=-1,
                        future_return=float(future_returns[i]),
                        latency_return=float(future_returns[i]),
                        config=config,
                    )
                    pos.inventory_base -= qty
                    pos.cash += revenue - fee - penalty
                    pos.total_fees += fee
                    pos.total_execution_penalty += penalty
                    pos.trades += 1
                    pos.sells += 1
                    pos.fill_events += 1
                    pos.fill_ratio_sum += fill_ratio
                    regime_bucket["trades"] += 1
                    regime_bucket["sells"] += 1
                    regime_bucket["notional_usd"] += float(revenue)
                    regime_bucket["fees_usd"] += float(fee)
                    regime_bucket["penalty_usd"] += float(penalty)
                    # Sells profit when price falls after the fill: side = -1
                    regime_bucket["_signed_return_notional"] += (
                        -float(future_returns[i]) * float(revenue)
                    )

        pv = pos.cash + pos.inventory_base * mid
        pos.pnl_history.append(pv)
        pos.inventory_history.append(pos.inventory_base * mid)

    metrics, pnl, _ = _position_metrics(pos, label)

    regime_summary: dict[str, dict] = {}
    for rid, bucket in regime_breakdown.items():
        candles = max(1, int(bucket["candles"]))
        notional = max(1e-9, float(bucket["notional_usd"]))
        observed = max(1, int(bucket["_dir_observed"]))
        regime_summary[str(int(rid))] = {
            "regime_id": int(rid),
            "candles": int(bucket["candles"]),
            "trades": int(bucket["trades"]),
            "buys": int(bucket["buys"]),
            "sells": int(bucket["sells"]),
            "notional_usd": round(float(bucket["notional_usd"]), 2),
            "fees_usd": round(float(bucket["fees_usd"]), 4),
            "penalty_usd": round(float(bucket["penalty_usd"]), 4),
            "trade_rate": round(int(bucket["trades"]) / candles, 4),
            "mean_stress": round(float(bucket["_stress_sum"]) / candles, 4),
            "mean_trend_score": round(float(bucket["_trend_sum"]) / candles, 4),
            "mean_dir_proba": round(float(bucket["_dir_proba_sum"]) / candles, 4),
            "model_accuracy": round(int(bucket["_dir_correct"]) / observed, 4)
            if int(bucket["_dir_observed"]) > 0
            else None,
            "model_predictions": int(bucket["_dir_observed"]),
            # Notional-weighted forward-return PnL approximation, in bps.
            # Positive = the side we filled was right on average (profitable).
            # Negative = adverse selection (filled side was wrong on average).
            "avg_signed_return_bps": round(
                float(bucket["_signed_return_notional"]) / notional * 10_000.0,
                4,
            ),
        }
    metrics["regime_breakdown"] = regime_summary

    spread_stats = {
        "avg_bid": round(top_bid_spreads.mean() * 100, 4),
        "avg_ask": round(top_ask_spreads.mean() * 100, 4),
        "std_bid": round(top_bid_spreads.std() * 100, 4),
        "std_ask": round(top_ask_spreads.std() * 100, 4),
    }

    return metrics, pnl, pos.inventory_history, spread_stats


def run_spread_param_search(test_df, dir_proba, vol_pred, drawdown_cap=-0.35, min_trades=150):
    """
    Strategy 1 – Signal parameter search.

    Sweeps the four ML spread-signal parameters while holding execution fixed at
    single-level defaults.  Looks for the combination that maximises the composite
    profit × Sharpe score subject to live-trading viability gates:
      * max_drawdown_pct >= drawdown_cap
      * total_trades    >= min_trades
      * sharpe_ratio    >= 4.0
      * total_return_pct > 0
    """
    base_spread_opts   = [0.22, 0.25, 0.30, 0.35, 0.40, 0.50]
    dir_weight_opts    = [0.30, 0.40, 0.50, 0.60, 0.70]
    vol_weight_opts    = [0.60, 0.80, 1.00, 1.20]
    conf_thresh_opts   = [0.50, 0.52, 0.55]
    order_amount_opts  = [25.0, 50.0, 75.0, 100.0]  # USD per order

    rows = []
    total = (len(base_spread_opts) * len(dir_weight_opts)
             * len(vol_weight_opts) * len(conf_thresh_opts)
             * len(order_amount_opts))
    print(f"  Signal param search: {total} candidates...")

    def _composite_score(r):
        m = r["ml_metrics"]
        roi = r.get("roi_on_capital", m["total_return_pct"])
        return round(m["total_return_pct"] * m["sharpe_ratio"] * 0.7 + roi * 0.3, 4)

    t0 = time.time()
    best_so_far_label = ""
    best_so_far_score = -9e9

    for base in base_spread_opts:
        for dw in dir_weight_opts:
            for vw in vol_weight_opts:
                for ct in conf_thresh_opts:
                    for order_usd in order_amount_opts:
                        cfg = MMConfig(
                            base_spread_pct=base,
                            direction_weight=dw,
                            volatility_weight=vw,
                            confidence_threshold=ct,
                            order_amount_usd=order_usd,
                            min_order_amount_usd=order_usd * 0.4,
                            max_order_amount_usd=order_usd * 1.6,
                        )
                        metrics, _, _, spread_stats = simulate_ml_variant(
                            test_df, dir_proba, vol_pred, cfg,
                            label="sig_candidate",
                            use_inventory_target=False,
                            use_multilevel=False,
                            use_confidence_sizing=False,
                        )
                        # Capital-efficiency: net profit per average dollar deployed
                        avg_inv = max(metrics["avg_inventory_usd"], 1.0)
                        net_profit = metrics["final_portfolio_usd"] - 10000.0
                        roi_on_capital = round(net_profit / avg_inv * 100, 4)

                        passes = bool(
                            metrics["max_drawdown_pct"] >= drawdown_cap
                            and metrics["total_trades"] >= min_trades
                            and metrics["sharpe_ratio"] >= 4.0
                            and metrics["total_return_pct"] > 0
                        )
                        row = {
                            "params": {
                                "base_spread_pct": base,
                                "direction_weight": dw,
                                "volatility_weight": vw,
                                "confidence_threshold": ct,
                                "order_amount_usd": order_usd,
                            },
                            "ml_metrics": metrics,
                            "ml_spread_stats": spread_stats,
                            "roi_on_capital": roi_on_capital,
                            "passes_live_gates": passes,
                        }
                        rows.append(row)

                        score = _composite_score(row)
                        if score > best_so_far_score:
                            best_so_far_score = score
                            best_so_far_label = (
                                f"Ret={metrics['total_return_pct']}% "
                                f"Sh={metrics['sharpe_ratio']} "
                                f"$ord={order_usd:.0f}"
                            )
                        _print_progress(len(rows), total, t0, best_so_far_label)

    def _composite(r):
        return _composite_score(r)

    eligible = [r for r in rows if r["passes_live_gates"]]
    eligible_sorted = sorted(eligible, key=_composite, reverse=True)
    all_sorted = sorted(rows, key=_composite, reverse=True)

    best = eligible_sorted[0] if eligible_sorted else all_sorted[0]
    selected_from = "eligible" if eligible_sorted else "all_fallback"

    top5 = [
        {**r, "composite_score": _composite(r)}
        for r in (eligible_sorted if eligible_sorted else all_sorted)[:5]
    ]

    return {
        "drawdown_cap_pct": drawdown_cap,
        "min_trades": min_trades,
        "candidates_tested": len(rows),
        "eligible_candidates": len(eligible),
        "selected_from": selected_from,
        "best_candidate": best,
        "top5_by_composite": top5,
    }


def run_variant_grid_search(test_df, dir_proba, vol_pred, drawdown_cap=-0.35):
    """Small constrained search over execution parameters and return best Sharpe under drawdown cap."""
    level_options = [1, 2, 3]
    step_options = [0.04, 0.06, 0.08]
    max_amt_options = [70.0, 90.0]
    inv_strength_options = [0.0, 0.15, 0.25]

    total_grid = len(level_options) * len(step_options) * len(max_amt_options) * len(inv_strength_options)
    print(f"  Exec-grid search: {total_grid} candidates...")
    t0_grid = time.time()
    best_grid_label = ""
    best_grid_sharpe = -9e9

    rows = []
    for levels in level_options:
        for step in step_options:
            for max_amt in max_amt_options:
                for inv_strength in inv_strength_options:
                    cfg = MMConfig(
                        order_levels=levels,
                        order_level_spread_step_pct=step,
                        min_order_amount_usd=20.0,
                        max_order_amount_usd=max_amt,
                        inventory_target_skew_strength=inv_strength,
                    )
                    use_inventory_target = inv_strength > 0

                    metrics, _, _, _ = simulate_ml_variant(
                        test_df,
                        dir_proba,
                        vol_pred,
                        cfg,
                        label="grid_candidate",
                        use_inventory_target=use_inventory_target,
                        use_multilevel=True,
                        use_confidence_sizing=True,
                    )

                    row = {
                        "params": {
                            "order_levels": levels,
                            "order_level_spread_step_pct": step,
                            "max_order_amount_usd": max_amt,
                            "inventory_target_skew_strength": inv_strength,
                        },
                        "ml_metrics": metrics,
                        "passes_drawdown_cap": bool(metrics["max_drawdown_pct"] >= drawdown_cap),
                    }
                    rows.append(row)

                    sh = metrics["sharpe_ratio"]
                    if sh > best_grid_sharpe:
                        best_grid_sharpe = sh
                        best_grid_label = (
                            f"Ret={metrics['total_return_pct']}% "
                            f"Sh={sh}"
                        )
                    _print_progress(len(rows), total_grid, t0_grid, best_grid_label)

    eligible = [r for r in rows if r["passes_drawdown_cap"]]
    if eligible:
        best = max(eligible, key=lambda r: (r["ml_metrics"]["sharpe_ratio"], r["ml_metrics"]["total_return_pct"]))
        selected_from = "eligible"
    else:
        best = max(rows, key=lambda r: (r["ml_metrics"]["sharpe_ratio"], r["ml_metrics"]["total_return_pct"]))
        selected_from = "all_fallback"

    # Keep top 5 by Sharpe for compact reporting.
    top5 = sorted(rows, key=lambda r: (r["ml_metrics"]["sharpe_ratio"], r["ml_metrics"]["total_return_pct"]), reverse=True)[:5]

    return {
        "drawdown_cap_pct": drawdown_cap,
        "candidates_tested": len(rows),
        "eligible_candidates": len(eligible),
        "selected_from": selected_from,
        "best_candidate": best,
        "top5_by_sharpe": top5,
    }


def run_backtest_ab():
    """Run A/B experiments and write a comparison JSON report."""
    print("=" * 70)
    print("MARKET-MAKING A/B HARNESS")
    print("=" * 70)

    df = pf.read_feather(str(DATA_DIR / "ETH_USDT-5m.feather"), memory_map=False)
    btc_df = pf.read_feather(str(DATA_DIR / "BTC_USDT-5m.feather"), memory_map=False)
    sol_file = DATA_DIR / "SOL_USDT-5m.feather"
    sol_df = pf.read_feather(str(sol_file), memory_map=False) if sol_file.exists() else None

    # Load models via shared loader (validates feature counts)
    stack = _load_prediction_stack()
    feature_cols = stack["feature_cols"]

    execution_df = load_or_build_execution_features(candle_dates=df["date"], prefer_cached=True)
    orderbook_df = load_or_build_orderbook_features(candle_dates=df["date"], prefer_cached=True)
    feat_df = compute_features(
        df,
        btc_df,
        sol_df=sol_df,
        execution_df=execution_df,
        orderbook_df=orderbook_df,
        binance_df=_load_reference_frame(),
    )
    feat_df = compute_labels(feat_df, horizon=6)
    feat_df = apply_regime_model(feat_df, stack.get("regime_bundle"))
    feat_df = feat_df.dropna(subset=feature_cols + ["direction"]).reset_index(drop=True)

    n = len(feat_df)
    val_start = int(n * 0.70)
    test_start = int(n * 0.85)
    val_df = feat_df.iloc[val_start:test_start].copy().reset_index(drop=True)
    test_df = feat_df.iloc[test_start:].copy().reset_index(drop=True)
    pred_val = _predict_with_stack(val_df, stack)
    pred_test = _predict_with_stack(test_df, stack)
    dir_proba = pred_test["direction_probability"].to_numpy()
    model_post_calibrated = bool(
        "post_calibrated" in pred_test.columns
        and float(pred_test["post_calibrated"].max()) > 0.0
    )
    # Apply the live MTF trend gate so A/B results match production.
    dir_proba, _ = _apply_backtest_mtf_gate(dir_proba, test_df)
    dir_proba, _ = _apply_backtest_conformal_gate(dir_proba, pred_test)
    if model_post_calibrated:
        cal_proba, cal_enabled = dir_proba, False
    else:
        raw_val_proba = pred_val["direction_probability"].to_numpy()
        cal_proba, cal_enabled = calibrate_direction_probabilities(
            raw_val_proba,
            val_df["direction"].values,
            dir_proba,
        )
    vol_pred = pred_test["predicted_volatility"].to_numpy()
    test_df["predicted_volatility"] = vol_pred

    base = MMConfig()

    # Keep the fixed baseline from the legacy runner for continuity.
    fixed_bid_spreads = np.zeros(len(test_df))
    fixed_ask_spreads = np.zeros(len(test_df))
    inventory_pct = 0.0
    for i in range(len(test_df)):
        book_spread_bps = _historical_book_spread_bps(test_df.iloc[i], base)
        bid_s, ask_s = compute_fixed_spreads(base, inventory_pct, book_spread_bps)
        fixed_bid_spreads[i] = bid_s
        fixed_ask_spreads[i] = ask_s
    test_df["bid_spread"] = fixed_bid_spreads
    test_df["ask_spread"] = fixed_ask_spreads
    fixed_metrics, _, _ = simulate_market_making(test_df, None, base, "Fixed Spread")

    experiments = [
        {
            "name": "control_ml",
            "label": "ML Control",
            "flags": {
                "use_inventory_target": False,
                "use_multilevel": False,
                "use_confidence_sizing": False,
            },
            "config": base,
        },
        {
            "name": "inventory_target_tuned",
            "label": "ML + Inventory Target (Tuned)",
            "flags": {
                "use_inventory_target": True,
                "use_multilevel": False,
                "use_confidence_sizing": False,
            },
            "config": MMConfig(
                inventory_target_base_pct=0.5,
                inventory_target_tolerance_pct=0.15,
                inventory_target_skew_strength=0.15,
            ),
        },
        {
            "name": "multilevel_profile",
            "label": "ML + Level Profile",
            "flags": {
                "use_inventory_target": False,
                "use_multilevel": True,
                "use_confidence_sizing": False,
            },
            "config": MMConfig(order_levels=ORDER_LEVELS, order_level_spread_step_pct=ORDER_LEVEL_SPREAD_STEP_PCT),
        },
        {
            "name": "confidence_sizing",
            "label": "ML + Confidence Sizing",
            "flags": {
                "use_inventory_target": False,
                "use_multilevel": False,
                "use_confidence_sizing": True,
            },
            "config": MMConfig(order_amount_usd=50.0, min_order_amount_usd=20.0, max_order_amount_usd=70.0),
        },
        {
            "name": "calibrated_probs",
            "label": "ML + Calibrated Probabilities",
            "flags": {
                "use_inventory_target": False,
                "use_multilevel": False,
                "use_confidence_sizing": False,
            },
            "config": base,
            "use_calibrated_probabilities": True,
        },
        {
            "name": "combined",
            "label": "ML Combined (Target+Levels+Sizing)",
            "flags": {
                "use_inventory_target": True,
                "use_multilevel": True,
                "use_confidence_sizing": True,
            },
            "config": MMConfig(
                order_levels=ORDER_LEVELS,
                order_level_spread_step_pct=ORDER_LEVEL_SPREAD_STEP_PCT,
                min_order_amount_usd=20.0,
                max_order_amount_usd=70.0,
                inventory_target_tolerance_pct=0.15,
                inventory_target_skew_strength=0.15,
            ),
            "use_calibrated_probabilities": True,
        },
    ]

    report_rows = []
    control_metrics = None

    print(f"Test period: {len(test_df)} candles")
    for exp in experiments:
        exp_probs = cal_proba if exp.get("use_calibrated_probabilities", False) and cal_enabled else dir_proba
        metrics, _, _, spread_stats = simulate_ml_variant(
            test_df,
            exp_probs,
            vol_pred,
            exp["config"],
            label=exp["label"],
            **exp["flags"],
        )
        if exp["name"] == "control_ml":
            control_metrics = metrics

        row = {
            "name": exp["name"],
            "ml_metrics": metrics,
            "ml_spread_stats": spread_stats,
            "used_calibrated_probabilities": bool(exp.get("use_calibrated_probabilities", False) and cal_enabled),
        }
        report_rows.append(row)
        print(
            f"{exp['name']:<18} ret={metrics['total_return_pct']:>7.3f}% "
            f"sharpe={metrics['sharpe_ratio']:>7.3f} trades={metrics['total_trades']:>5}"
        )

    if control_metrics is None:
        raise RuntimeError("control_ml experiment missing from report_rows")

    for row in report_rows:
        row["delta_vs_control"] = {
            "total_return_pct": round(row["ml_metrics"]["total_return_pct"] - control_metrics["total_return_pct"], 3),
            "sharpe_ratio": round(row["ml_metrics"]["sharpe_ratio"] - control_metrics["sharpe_ratio"], 3),
            "max_drawdown_pct": round(row["ml_metrics"]["max_drawdown_pct"] - control_metrics["max_drawdown_pct"], 3),
            "total_trades": row["ml_metrics"]["total_trades"] - control_metrics["total_trades"],
            "avg_inventory_usd": round(row["ml_metrics"]["avg_inventory_usd"] - control_metrics["avg_inventory_usd"], 3),
        }

    best = max(report_rows, key=lambda r: (r["ml_metrics"]["sharpe_ratio"], r["ml_metrics"]["total_return_pct"]))
    print(f"Best by Sharpe: {best['name']} (Sharpe={best['ml_metrics']['sharpe_ratio']}, Return={best['ml_metrics']['total_return_pct']}%)")

    # ── Strategy 2: Execution parameter grid search ─────────────────────
    print("Running Strategy 2: execution param grid search (drawdown cap -0.35%)...")
    grid_report = run_variant_grid_search(test_df, cal_proba if cal_enabled else dir_proba, vol_pred, drawdown_cap=-0.35)
    gb = grid_report["best_candidate"]
    print(
        "  Exec-grid best: "
        f"levels={gb['params']['order_levels']} "
        f"step={gb['params']['order_level_spread_step_pct']:.2f}% "
        f"max_amt={gb['params']['max_order_amount_usd']:.0f} "
        f"inv_skew={gb['params']['inventory_target_skew_strength']:.2f} "
        f"Sharpe={gb['ml_metrics']['sharpe_ratio']} "
        f"Ret={gb['ml_metrics']['total_return_pct']}%"
    )

    # ── Strategy 1: Signal parameter search ──────────────────────────────
    print("Running Strategy 1: signal param search (base_spread/dir_weight/vol_weight/conf)...")
    spread_report = run_spread_param_search(
        test_df, cal_proba if cal_enabled else dir_proba, vol_pred, drawdown_cap=-0.35
    )
    sb = spread_report["best_candidate"]
    print(
        "  Signal-param best: "
        f"base={sb['params']['base_spread_pct']:.2f}% "
        f"dw={sb['params']['direction_weight']:.2f} "
        f"vw={sb['params']['volatility_weight']:.2f} "
        f"ct={sb['params']['confidence_threshold']:.2f} "
        f"Sharpe={sb['ml_metrics']['sharpe_ratio']} "
        f"Ret={sb['ml_metrics']['total_return_pct']}%"
    )

    # ── Pick overall winner by composite score ────────────────────────────
    def _composite(m):
        return round(m["total_return_pct"] * m["sharpe_ratio"], 4)

    exec_score    = _composite(gb["ml_metrics"])
    signal_score  = _composite(sb["ml_metrics"])

    if signal_score >= exec_score:
        winner_strategy = "signal_param_search"
        winner_params = sb["params"]
        winner_metrics = sb["ml_metrics"]
    else:
        winner_strategy = "execution_grid_search"
        winner_params = gb["params"]
        winner_metrics = gb["ml_metrics"]

    print("\n" + "=" * 70)
    print(f"WINNER: {winner_strategy}  (composite score: {max(exec_score, signal_score):.4f})")
    print(f"  Params:  {winner_params}")
    print(f"  Return:  {winner_metrics['total_return_pct']}%  "
          f"Sharpe: {winner_metrics['sharpe_ratio']}  "
          f"DD: {winner_metrics['max_drawdown_pct']}%")
    print("=" * 70)

    report = {
        "control_metrics": control_metrics,
        "fixed_metrics": fixed_metrics,
        "test_period": {
            "start": str(test_df["date"].iloc[0]),
            "end": str(test_df["date"].iloc[-1]),
            "candles": len(test_df),
        },
        "experiments": report_rows,
        "best_experiment": best["name"],
        "calibration": {
            "enabled": cal_enabled,
            "model_post_calibrated": model_post_calibrated,
            "validation_rows": len(val_df),
        },
        "strategy2_exec_grid": grid_report,
        "strategy1_signal_search": spread_report,
        "best_live_config": {
            "strategy": winner_strategy,
            "params": winner_params,
            "metrics": winner_metrics,
            "composite_score": max(exec_score, signal_score),
        },
    }

    out_path = Path(__file__).resolve().parent / "backtest_ab_results.json"
    _write_json_report(out_path, report)

    return report


def run_as_sweep(test_df, dir_proba, vol_pred):
    """
    Adverse-selection threshold sweep.

    Applies a simplified AS filter (vol-stress + direction-confidence components,
    computable from ML predictions alone) and sweeps three tuning knobs:
      high_thresh         — score above which severity flips to "high" for extra penalty
      spread_mult_strength — controls how much the AS score widens spreads
      size_mult_strength   — controls how much the AS score reduces order size

    Returns a dict with baseline metrics, best params, and top5.
    """
    # ── Per-candle simplified AS score ────────────────────────────────────────
    median_vol = float(np.median(vol_pred))
    vol_ratio = vol_pred / (median_vol + 1e-10)
    vol_stress = np.clip((vol_ratio - 1.0) / 1.5, 0.0, 1.0)
    dir_confidence = np.clip(np.abs(dir_proba - 0.5) * 2.0, 0.0, 1.0)
    # Blend: 64% vol-stress + 36% directional confidence (proxy for adverse flow)
    as_scores = 0.64 * vol_stress + 0.36 * dir_confidence

    high_thresh_opts     = [0.55, 0.65, 0.75, 0.85]
    spread_strength_opts = [0.25, 0.40, 0.55, 0.70]
    size_strength_opts   = [0.30, 0.45, 0.60]
    conf_factor          = 0.35   # same as live detect_adverse_selection()

    base_cfg = MMConfig()

    # ── Pre-compute baseline ML spreads ────────────────────────────────────────
    ml_bid_base = np.zeros(len(test_df))
    ml_ask_base = np.zeros(len(test_df))
    for i in range(len(test_df)):
        book_spread_bps = _historical_book_spread_bps(test_df.iloc[i], base_cfg)
        bid_s, ask_s = compute_ml_spreads(dir_proba[i], vol_pred[i], base_cfg, 0.0, book_spread_bps)
        ml_bid_base[i] = bid_s
        ml_ask_base[i] = ask_s

    def _run_as_sim(bid_s, ask_s, order_mults):
        """Lightweight sim that accepts per-candle bid/ask spreads and order-size multipliers."""
        pos = Position()
        mid_prices = test_df["close"].values
        highs = test_df["high"].values
        lows = test_df["low"].values
        min_ord = base_cfg.min_order_amount_usd
        max_ord = base_cfg.max_order_amount_usd
        base_ord = base_cfg.order_amount_usd
        max_inv = base_cfg.max_inventory
        fee = base_cfg.fee_pct / 100
        for i in range(len(test_df)):
            mid = mid_prices[i]
            order_usd = float(np.clip(base_ord * order_mults[i], min_ord, max_ord))
            bid_price = mid * (1 - bid_s[i])
            ask_price = mid * (1 + ask_s[i])
            inv_usd = pos.inventory_base * mid
            if inv_usd < max_inv and pos.cash >= order_usd and lows[i] <= bid_price:
                qty = order_usd / bid_price
                pos.inventory_base += qty
                pos.cash -= order_usd * (1 + fee)
                pos.total_fees += order_usd * fee
                pos.trades += 1
                pos.buys += 1
            if pos.inventory_base * mid > order_usd * 0.5 and highs[i] >= ask_price:
                qty = min(order_usd / ask_price, pos.inventory_base)
                revenue = qty * ask_price
                pos.inventory_base -= qty
                pos.cash += revenue * (1 - fee)
                pos.total_fees += revenue * fee
                pos.trades += 1
                pos.sells += 1
            pos.pnl_history.append(pos.cash + pos.inventory_base * mid)
        pnl = np.array(pos.pnl_history)
        rets = np.diff(pnl) / pnl[:-1]
        rets = rets[np.isfinite(rets)]
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(288 * 365))
        max_dd = float(np.min(pnl / np.maximum.accumulate(pnl) - 1) * 100)
        total_ret = float((pnl[-1] / pos.initial_cash - 1) * 100)
        return {
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_return_pct": round(total_ret, 4),
            "total_trades": pos.trades,
        }

    baseline_metrics = _run_as_sim(ml_bid_base, ml_ask_base, np.ones(len(test_df)))

    rows = []
    total = len(high_thresh_opts) * len(spread_strength_opts) * len(size_strength_opts)
    print(f"  AS sweep: {total} candidates  (baseline Sharpe={baseline_metrics['sharpe_ratio']:.3f})")
    t0 = time.time()
    best_label = ""
    best_composite = -9e9

    for high_thresh in high_thresh_opts:
        for spread_strength in spread_strength_opts:
            for size_strength in size_strength_opts:
                cf = 1.0 - conf_factor * dir_confidence          # confidence damping
                spread_mults = np.clip(1.0 + spread_strength * as_scores * cf, 1.0, 1.55)
                size_mults   = np.clip(1.0 - size_strength   * as_scores * cf, 0.40, 1.0)
                # Extra δ for high-severity candles
                high_mask = as_scores >= high_thresh
                spread_mults = np.where(high_mask, np.clip(spread_mults + 0.08, 1.0, 1.55), spread_mults)
                size_mults   = np.where(high_mask, np.clip(size_mults   - 0.08, 0.40, 1.0), size_mults)

                min_s = base_cfg.min_spread_pct / 100
                max_s = base_cfg.max_spread_pct / 100
                bid_adj = np.clip(ml_bid_base * spread_mults, min_s, max_s)
                ask_adj = np.clip(ml_ask_base * spread_mults, min_s, max_s)

                m = _run_as_sim(bid_adj, ask_adj, size_mults)

                delta_sharpe = round(m["sharpe_ratio"] - baseline_metrics["sharpe_ratio"], 4)
                delta_dd     = round(m["max_drawdown_pct"] - baseline_metrics["max_drawdown_pct"], 4)
                # Composite: reward absolute Sharpe AND improvement over baseline (prevents
                # overfitting to variants that just happen to have low baseline noise).
                composite = round(m["sharpe_ratio"] * 0.6 + delta_sharpe * 0.4, 4)

                row = {
                    "params": {
                        "high_thresh": high_thresh,
                        "spread_mult_strength": spread_strength,
                        "size_mult_strength": size_strength,
                    },
                    "metrics": m,
                    "delta_sharpe": delta_sharpe,
                    "delta_drawdown": delta_dd,
                    "composite_score": composite,
                }
                rows.append(row)

                if composite > best_composite:
                    best_composite = composite
                    best_label = (
                        f"dSh={delta_sharpe:+.3f} "
                        f"ht={high_thresh:.2f} "
                        f"ss={spread_strength:.2f} "
                        f"sz={size_strength:.2f}"
                    )
                _print_progress(len(rows), total, t0, best_label)

    rows_sorted = sorted(rows, key=lambda r: r["composite_score"], reverse=True)
    best = rows_sorted[0]

    print(
        f"\n  Best AS params:  high_thresh={best['params']['high_thresh']}  "
        f"spread_strength={best['params']['spread_mult_strength']}  "
        f"size_strength={best['params']['size_mult_strength']}"
    )
    print(
        f"  dSharpe={best['delta_sharpe']:+.4f}  "
        f"dDD={best['delta_drawdown']:+.4f}  "
        f"abs Sharpe={best['metrics']['sharpe_ratio']:.4f}  "
        f"trades={best['metrics']['total_trades']}"
    )

    return {
        "baseline_metrics": baseline_metrics,
        "candidates_tested": len(rows),
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "best_delta_sharpe": best["delta_sharpe"],
        "best_delta_drawdown": best["delta_drawdown"],
        "top5_by_composite": rows_sorted[:5],
    }


def run_backtest_as_sweep():
    """Load data, build features, run AS threshold sweep, write as_tuning_results.json."""
    print("=" * 70)
    print("ADVERSE-SELECTION THRESHOLD SWEEP")
    print("=" * 70)

    df      = pf.read_feather(str(DATA_DIR / "ETH_USDT-5m.feather"), memory_map=False)
    btc_df  = pf.read_feather(str(DATA_DIR / "BTC_USDT-5m.feather"), memory_map=False)
    sol_file = DATA_DIR / "SOL_USDT-5m.feather"
    sol_df  = pf.read_feather(str(sol_file), memory_map=False) if sol_file.exists() else None

    stack = _load_prediction_stack()
    feature_cols = stack["feature_cols"]

    execution_df  = load_or_build_execution_features(candle_dates=df["date"], prefer_cached=True)
    orderbook_df  = load_or_build_orderbook_features(candle_dates=df["date"], prefer_cached=True)
    feat_df       = compute_features(
        df,
        btc_df,
        sol_df=sol_df,
        execution_df=execution_df,
        orderbook_df=orderbook_df,
        binance_df=_load_reference_frame(),
    )
    feat_df       = compute_labels(feat_df, horizon=6)
    feat_df       = apply_regime_model(feat_df, stack.get("regime_bundle"))
    feat_df       = feat_df.dropna(subset=feature_cols + ["direction"]).reset_index(drop=True)

    n          = len(feat_df)
    test_start = int(n * 0.85)
    test_df    = feat_df.iloc[test_start:].copy().reset_index(drop=True)
    print(f"Test period: {len(test_df)} candles  ({test_df['date'].iloc[0]} -> {test_df['date'].iloc[-1]})")

    pred_test = _predict_with_stack(test_df, stack)
    dir_proba = pred_test["direction_probability"].to_numpy()
    # Apply the live MTF trend gate so sweep results match production.
    dir_proba, _ = _apply_backtest_mtf_gate(dir_proba, test_df)
    dir_proba, _ = _apply_backtest_conformal_gate(dir_proba, pred_test)
    vol_pred  = pred_test["predicted_volatility"].to_numpy()

    sweep_result = run_as_sweep(test_df, dir_proba, vol_pred)

    out_path = Path(__file__).resolve().parent / "as_tuning_results.json"
    _write_json_report(out_path, sweep_result)

    best = sweep_result["best_params"]
    print("\nSuggested updates to detect_adverse_selection() in signal_server.py:")
    print(f"  high_thresh         = {best['high_thresh']}")
    print(f"  spread_mult_strength = {best['spread_mult_strength']}  (replace 0.55)")
    print(f"  size_mult_strength   = {best['size_mult_strength']}  (replace 0.60)")

    return sweep_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML market-making backtests")
    parser.add_argument("--ab",       action="store_true", help="Run A/B experiment harness")
    parser.add_argument("--as-sweep", action="store_true", help="Run adverse-selection threshold sweep")
    args = parser.parse_args()

    if args.as_sweep:
        run_backtest_as_sweep()
    elif args.ab:
        run_backtest_ab()
    else:
        run_backtest()
