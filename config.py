"""
Shared configuration and utilities for the ML market-making system.
Centralises paths, parameters, and model-loading logic used across
train.py, signal_server.py, backtest.py, diagnostic.py, and experiments.py.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from shared.paths import (
    DATA_DIR as SHARED_DATA_DIR,
    EXECUTION_DATA_DIR as SHARED_EXECUTION_DATA_DIR,
    EXECUTION_FEATURES_PATH as SHARED_EXECUTION_FEATURES_PATH,
    EXECUTION_SUMMARY_PATH as SHARED_EXECUTION_SUMMARY_PATH,
    HB_EXECUTION_DB_FILE as SHARED_EXECUTION_DB_PATH,
    HB_SIGNAL_FILE as SHARED_SIGNAL_FILE,
    LATEST_MODEL_DIR as SHARED_LATEST_MODEL_DIR,
    LOG_DIR as SHARED_LOG_DIR,
    ML_DIR as SHARED_ML_DIR,
    MODEL_DIR as SHARED_MODEL_DIR,
    ORDERBOOK_DATA_DIR as SHARED_ORDERBOOK_DATA_DIR,
    ORDERBOOK_FEATURES_PATH as SHARED_ORDERBOOK_FEATURES_PATH,
    ORDERBOOK_SNAPSHOT_PATH as SHARED_ORDERBOOK_SNAPSHOT_PATH,
    ORDERBOOK_SUMMARY_PATH as SHARED_ORDERBOOK_SUMMARY_PATH,
    PRIMARY_EXCHANGE_ID as SHARED_PRIMARY_EXCHANGE_ID,
    PROJECT_ROOT as SHARED_PROJECT_ROOT,
    REFERENCE_DATA_DIR as SHARED_REFERENCE_DATA_DIR,
    REFERENCE_EXCHANGE_ID as SHARED_REFERENCE_EXCHANGE_ID,
    TRAINING_STATUS_FILE as SHARED_TRAINING_STATUS_FILE,
    detect_primary_exchange_from_hummingbot as _shared_detect_primary_exchange_from_hummingbot,
    resolve_primary_exchange_id as _shared_resolve_primary_exchange_id,
)

# ── Paths (all relative to this file) ──────────────────────────────




def _detect_primary_exchange_from_hummingbot() -> str:
    """Delegate exchange detection to the shared standalone path layer."""
    return _shared_detect_primary_exchange_from_hummingbot()


def _resolve_primary_exchange_id(default: str = "binanceus") -> str:
    """Delegate exchange resolution to the shared standalone path layer."""
    return _shared_resolve_primary_exchange_id(default)


PRIMARY_EXCHANGE_ID = SHARED_PRIMARY_EXCHANGE_ID
REFERENCE_EXCHANGE_ID = SHARED_REFERENCE_EXCHANGE_ID

ML_DIR = SHARED_ML_DIR
PROJECT_ROOT = SHARED_PROJECT_ROOT
DATA_DIR = SHARED_DATA_DIR
MODEL_DIR = SHARED_MODEL_DIR
LATEST_MODEL_DIR = SHARED_LATEST_MODEL_DIR
SIGNAL_FILE = SHARED_SIGNAL_FILE
LOG_DIR = SHARED_LOG_DIR
TRAINING_STATUS_FILE = SHARED_TRAINING_STATUS_FILE
EXECUTION_DB_PATH = SHARED_EXECUTION_DB_PATH
EXECUTION_DATA_DIR = SHARED_EXECUTION_DATA_DIR
EXECUTION_FEATURES_PATH = SHARED_EXECUTION_FEATURES_PATH
EXECUTION_SUMMARY_PATH = SHARED_EXECUTION_SUMMARY_PATH
ORDERBOOK_DATA_DIR = SHARED_ORDERBOOK_DATA_DIR
ORDERBOOK_SNAPSHOT_PATH = SHARED_ORDERBOOK_SNAPSHOT_PATH
ORDERBOOK_FEATURES_PATH = SHARED_ORDERBOOK_FEATURES_PATH
ORDERBOOK_SUMMARY_PATH = SHARED_ORDERBOOK_SUMMARY_PATH

# ── Trading parameters ─────────────────────────────────────────────
PAIR = "ETH_USDT"
SYMBOL = "ETH/USDT"
BTC_SYMBOL = "BTC/USDT"
SOL_SYMBOL = "SOL/USDT"
TIMEFRAME = "5m"
HORIZON = 6           # 6 candles × 5m = 30 min forward look (primary)
HORIZON_FAST = 1      # 1 candle × 5m = 5 min (adverse selection horizon)
HORIZON_MID = 3       # 3 candles × 5m = 15 min (medium context horizon)
CANDLES_NEEDED = 380  # EMA-200 warmup + 144 lookback + safety margin
CANDLE_SECONDS = 300

# Multi-horizon ensemble weights (must sum to 1.0).
# Fast model captures adverse selection; slow gives broader trend context.
# Shifted toward fast: 5-min adverse selection matters most for market making fills.
HORIZON_FAST_WEIGHT = 0.55
HORIZON_MID_WEIGHT  = 0.25
HORIZON_SLOW_WEIGHT = 0.20

# Training: exponential time-decay sample weights.
# Half-life of 60 days: 90-day-old candles keep ~35% weight (was 12% at 30d).
# Longer half-life gives more stable models across retrain cycles.
SAMPLE_DECAY_HALFLIFE_DAYS = 60

# ── Exchange alignment ─────────────────────────────────────────────
# Signal server, retrain refresh, and validation must use the same primary exchange.
EXCHANGE_ID = PRIMARY_EXCHANGE_ID

# ── Cross-exchange reference (Coinbase as leading-indicator data source) ────
# Coinbase remains the reference venue for cross-exchange premium features.
# Coinbase uses USD pairs (ETH/USD, BTC/USD); signal server handles the mapping.
BINANCE_EXCHANGE_ID = REFERENCE_EXCHANGE_ID    # legacy alias kept for compatibility
BINANCE_DATA_DIR = SHARED_REFERENCE_DATA_DIR

# ── Split ratios ───────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15      # remaining 15 % = test
DIR3_FLAT_BAND = 0.001  # ±0.1 % treated as "flat"

# ── Spread parameters ──────────────────────────────────────────────
# Spreads are now ANCHORED to the live order book spread, not arbitrary constants.
# The ML signal adjusts the spread around the book midpoint.
# Binance US maker fee: 0.10% (10 bps) at base tier; floor = fee + 2 bps adverse buffer.
# BASE is the default when book_spread_bps is unavailable; the book-anchoring path
# in compute_spreads() overrides this when live data is available.
BASE_SPREAD_PCT = 0.30       # 30 bps — optimal from A/B param search (Sharpe 14.81)
MIN_SPREAD_PCT = 0.05        # 5 bps floor — signal server minimum (Hummingbot enforces its own fee-aware floor)
MAX_SPREAD_PCT = 0.80        # 80 bps — safety cap for extreme volatility
DIRECTION_WEIGHT = 0.60      # stronger directional lean (A/B optimal)
VOLATILITY_WEIGHT = 0.45     # reduced from 0.6 — less vol amplification for stabler spreads
CONFIDENCE_THRESHOLD = 0.50  # lower bar = more signals activated (A/B optimal)
ORDER_AMOUNT_USD = 100.0     # smaller per-order size for tighter risk (A/B optimal)
MIN_ORDER_AMOUNT_USD = 20.0  # minimum total quote notional per side
MAX_ORDER_AMOUNT_USD = 70.0  # cap total quote notional per side before level split
ORDER_LEVELS = 2             # corrected-fill A/B winner
ORDER_LEVEL_SPREAD_STEP_PCT = 0.06  # 6 bps between quote levels
INVENTORY_SKEW_FACTOR = 0.4

# ── Circuit-breaker ────────────────────────────────────────────────
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_CYCLES = 10

# ── Neural LSTM ensemble parameters ───────────────────────────────
NN_SEQUENCE_LENGTH = 48    # 4-hour context window (4 × 12 × 5-min)
NN_HIDDEN_SIZE     = 256
NN_NUM_LAYERS      = 2
NN_DROPOUT         = 0.30
NN_ENSEMBLE_WEIGHT = 0.35  # real BiLSTM on GPU (PyTorch 2.7.1+cu118, Pascal sm_61) — captures 4h sequential context tree models miss

# ── XGBoost direction model hyperparameters ────────────────────────
XGB_DIR_PARAMS = {
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 1500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "colsample_bylevel": 0.8,
    "min_child_weight": 20,
    "gamma": 0.2,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": 1.0,
    "early_stopping_rounds": 80,
    "verbosity": 1,
}

# ── LightGBM volatility model hyperparameters ─────────────────────
LGB_VOL_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 1500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "min_child_samples": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

# ── XGBoost volatility fallback hyperparameters ────────────────────
XGB_VOL_FALLBACK_PARAMS = {
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "n_estimators": 1200,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "colsample_bylevel": 0.8,
    "min_child_weight": 25,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "verbosity": 1,
}


# ── Isotonic calibrator (defined here so pickle finds it regardless of ──────
# ── which script is __main__ at load time) ─────────────────────────────────
class _IsoCalibrator:
    """Thin wrapper: XGBoost model + fitted IsotonicRegression.

    Exposes predict_proba() so it's a drop-in replacement for the raw model
    wherever calibrated probabilities are needed.
    """
    def __init__(self, base_model, iso: IsotonicRegression):
        self._base = base_model
        self._iso  = iso

    def predict_proba(self, X):
        raw = self._base.predict_proba(X)[:, 1]
        cal = np.clip(self._iso.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


# ── LightGBM availability ─────────────────────────────────────────
try:
    import lightgbm as lgb  # noqa: F401
    LGB_AVAILABLE = True
    LGB_IMPORT_ERROR = ""
except Exception as e:
    lgb = None  # type: ignore[assignment]
    LGB_AVAILABLE = False
    LGB_IMPORT_ERROR = str(e)


# ═══════════════════════════════════════════════════════════════════
# Shared utility functions
# ═══════════════════════════════════════════════════════════════════

def load_models(
    model_dir: Optional[Path] = None,
) -> Tuple:
    """
    Load trained models+metadata from *model_dir* (default: ``LATEST_MODEL_DIR``).

    Returns
    -------
    dir_model, dir3_model, vol_model, calibrator, metadata
        *dir3_model* and *calibrator* may be ``None`` if not available.

    Raises
    ------
    FileNotFoundError  if a required artefact is missing.
    ValueError         if loaded model feature count != metadata feature count.
    """
    model_dir = model_dir or LATEST_MODEL_DIR

    # --- direction model (required) ---
    dir_path = model_dir / "direction_model.json"
    if not dir_path.exists():
        raise FileNotFoundError(f"Direction model not found: {dir_path}")
    dir_model = xgb.XGBClassifier()
    dir_model.load_model(str(dir_path))

    # --- 3-class direction model (optional) ---
    dir3_model = None
    dir3_path = model_dir / "direction3_model.json"
    if dir3_path.exists():
        dir3_model = xgb.XGBClassifier()
        dir3_model.load_model(str(dir3_path))

    # --- volatility model (pkl preferred, json fallback) ---
    vol_pkl = model_dir / "volatility_model.pkl"
    vol_json = model_dir / "volatility_model.json"
    if vol_pkl.exists():
        with open(vol_pkl, "rb") as f:
            vol_model = pickle.load(f)
    elif vol_json.exists():
        vol_model = xgb.XGBRegressor()
        vol_model.load_model(str(vol_json))
    else:
        raise FileNotFoundError(
            f"Volatility model not found (tried {vol_pkl} and {vol_json})"
        )

    # --- calibrator (optional) ---
    calibrator = None
    cal_path = model_dir / "direction_calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)

    # --- metadata ---
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path) as f:
        metadata = json.load(f)

    feature_cols = metadata.get("feature_columns", [])
    n_feat = len(feature_cols)

    # Validate feature count matches loaded models
    if hasattr(dir_model, "n_features_in_") and dir_model.n_features_in_ != n_feat:
        raise ValueError(
            f"Direction model expects {dir_model.n_features_in_} features "
            f"but metadata lists {n_feat}"
        )
    if hasattr(vol_model, "n_features_in_") and vol_model.n_features_in_ != n_feat:
        raise ValueError(
            f"Volatility model expects {vol_model.n_features_in_} features "
            f"but metadata lists {n_feat}"
        )

    return dir_model, dir3_model, vol_model, calibrator, metadata


def dynamic_confidence_threshold(base_threshold: float, vol_pred: float) -> float:
    """Adjust confidence threshold by volatility regime.

    Low vol = mean-reversion regime → raise threshold (fewer, higher-quality signals).
    High vol = trending regime → lower threshold (more signals to capture momentum).
    """
    vol_ratio = np.clip(vol_pred / 0.005, 0.5, 3.0)
    return float(np.clip(base_threshold - 0.04 * (vol_ratio - 1.0), 0.42, 0.58))


def compute_spreads(
    dir_prob: float,
    vol_pred: float,
    inventory_pct: float = 0.0,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    base_spread_pct: float = BASE_SPREAD_PCT,
    min_spread_pct: float = MIN_SPREAD_PCT,
    max_spread_pct: float = MAX_SPREAD_PCT,
    direction_weight: float = DIRECTION_WEIGHT,
    volatility_weight: float = VOLATILITY_WEIGHT,
    inventory_skew_factor: float = INVENTORY_SKEW_FACTOR,
    book_spread_bps: float = 0.0,
) -> Tuple[float, float]:
    """Compute asymmetric bid/ask spreads from ML predictions.

    When book_spread_bps > 0, the base spread is anchored to the live order book
    so the bot is always competitive with the market.  The ML signal then adjusts
    the spread asymmetrically around the book midpoint.
    """
    if book_spread_bps > 0:
        # Anchor to the live book: quote just outside half-spread to capture the spread.
        # On tight books (sub-fee half-spread) raw_base is very small — the min floor
        # below dominates and vol_multiplier amplification is neutralized, matching
        # the live signal server behavior where Hummingbot's floor dominates anyway.
        book_half_spread = (book_spread_bps / 10_000) / 2
        raw_base = book_half_spread * 1.2
    else:
        raw_base = base_spread_pct / 100

    vol_multiplier = 1.0 + volatility_weight * (vol_pred / 0.005 - 1.0)
    vol_multiplier = np.clip(vol_multiplier, 0.6, 2.5)
    # Apply vol amplification to the RAW base first, then enforce the min floor.
    # Previously the floor was applied before vol_mult so a 1-bps book-anchored
    # base was floored to 5 bps and then amplified 2.5× to 12 bps — amplifying
    # the arbitrary floor rather than the book signal. In backtest this broke
    # spectacularly: BASE_SPREAD_PCT=30 bps × 2.5 = 75 bps at high vol, 3× wider
    # than the 22-bps Hummingbot floor live actually quotes.
    adjusted_base = max(raw_base * vol_multiplier, min_spread_pct / 100)

    direction_shift = 0.0
    if dir_prob > conf_threshold or dir_prob < (1 - conf_threshold):
        direction_shift = (dir_prob - 0.5) * direction_weight * adjusted_base

    inventory_shift = inventory_pct * inventory_skew_factor * adjusted_base

    bid_spread = adjusted_base - direction_shift + inventory_shift
    ask_spread = adjusted_base + direction_shift - inventory_shift

    min_s = min_spread_pct / 100
    max_s = max_spread_pct / 100
    bid_spread = float(np.clip(bid_spread, min_s, max_s))
    ask_spread = float(np.clip(ask_spread, min_s, max_s))

    return bid_spread, ask_spread


def load_multi_horizon_models(model_dir: Optional[Path] = None) -> Tuple:
    """Load optional fast (5-min) and mid (15-min) direction models.

    Returns (fast_model, mid_model) — either may be None if not yet trained.
    Falls back gracefully so the signal server works before the first retrain
    that includes multi-horizon training.
    """
    model_dir = model_dir or LATEST_MODEL_DIR
    fast_model = None
    mid_model  = None

    fast_path = model_dir / "direction_model_fast.json"
    if fast_path.exists():
        try:
            fast_model = xgb.XGBClassifier()
            fast_model.load_model(str(fast_path))
        except Exception as e:
            print(f"[config] Fast direction model load failed: {e}")

    mid_path = model_dir / "direction_model_mid.json"
    if mid_path.exists():
        try:
            mid_model = xgb.XGBClassifier()
            mid_model.load_model(str(mid_path))
        except Exception as e:
            print(f"[config] Mid direction model load failed: {e}")

    loaded = sum(m is not None for m in (fast_model, mid_model))
    if loaded:
        print(f"[config] Multi-horizon models loaded: fast={'yes' if fast_model else 'no'}, "
              f"mid={'yes' if mid_model else 'no'}")
    return fast_model, mid_model


def load_neural_model(n_features: int, model_dir: Optional[Path] = None) -> Tuple:
    """Load the trained LSTM ensemble member.  Returns (model, meta) or (None, {})."""
    try:
        from neural_model import load_neural_model as _load
        return _load(model_dir or LATEST_MODEL_DIR, n_features)
    except Exception as e:
        print(f"Neural model load failed: {e}")
        return None, {}


def get_direction_probabilities(dir_model, dir3_model, X) -> np.ndarray:
    """Return binary up-probabilities from either a binary or 3-class direction model."""
    if dir3_model is not None:
        proba = dir3_model.predict_proba(X)
        p_down = proba[:, 0]
        p_up = proba[:, 2]
        return np.clip(0.5 + 0.5 * (p_up - p_down), 0.0, 1.0)
    return dir_model.predict_proba(X)[:, 1]
