from __future__ import annotations

from typing import Mapping

import numpy as np


def apply_orderbook_overlays(
    bid_spread: float,
    ask_spread: float,
    ob_state: Mapping[str, object] | None,
    *,
    min_spread_pct: float,
    max_spread_pct: float,
) -> tuple[float, float]:
    """Apply the live L1/L2 orderbook spread overlays.

    Spreads are fractional values, while min/max spread inputs are percentages
    to match the values exported by config.py.
    """
    if not ob_state or not bool(ob_state.get("ob_has_data", False)):
        return bid_spread, ask_spread

    def _float_value(key: str, default: float) -> float:
        try:
            value = ob_state.get(key, default)
            if value is None:
                return default
            value_f = float(value)
            if not np.isfinite(value_f):
                return default
            return value_f
        except (TypeError, ValueError):
            return default

    spread_bps = _float_value("ob_quoted_spread_bps", 0.0)
    depth_imb_5 = _float_value("ob_depth_imbalance_5", 0.0)
    book_pressure = _float_value("ob_book_pressure", 0.5)
    wmid_offset_bps = _float_value("ob_weighted_mid_offset_bps", 0.0)
    pressure_velocity = _float_value("ob_pressure_velocity", 0.0)

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


def adverse_size_cap(adverse_score: float) -> float | None:
    """Return the maximum size multiplier allowed for a toxicity score."""
    try:
        score = float(adverse_score)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score) or score < 0.60:
        return None
    return float(np.interp(score, [0.60, 0.85, 1.00], [0.85, 0.70, 0.55]))


def apply_adverse_size_cap(multiplier: float, adverse_score: float) -> float:
    cap = adverse_size_cap(adverse_score)
    return float(multiplier if cap is None else min(multiplier, cap))
