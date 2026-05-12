"""
L1 & L2 Order Book Feature Engineering
========================================
Captures live order book snapshots from the exchange via ccxt and
computes real microstructure features:

L1 (Top of Book):
  - Best bid/ask prices and sizes
  - Quoted spread (absolute & relative)
  - Mid-price
  - Bid-ask size imbalance

L2 (Depth):
  - Cumulative depth at N levels (5, 10, 20)
  - Depth imbalance at each level
  - Weighted mid-price
  - Book pressure (aggressive vs passive side weight)
  - Depth slope (rate of depth decay away from mid)

Rolling aggregations over 15m and 1h windows produce features that
capture short-horizon liquidity dynamics for the ML pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import pyarrow.feather as pf

from config import DATA_DIR

ORDERBOOK_DATA_DIR = DATA_DIR / "orderbook"
ORDERBOOK_SNAPSHOT_PATH = ORDERBOOK_DATA_DIR / "orderbook_snapshots.feather"
ORDERBOOK_FEATURES_PATH = ORDERBOOK_DATA_DIR / "orderbook_features.feather"
ORDERBOOK_SUMMARY_PATH = ORDERBOOK_DATA_DIR / "orderbook_summary.json"

# Depth levels to capture
_DEPTH_LEVELS = [5, 10, 20]
_MAX_DEPTH = max(_DEPTH_LEVELS)
_ROLL_15M = 3   # 3 × 5min candles
_ROLL_1H = 12   # 12 × 5min candles


def _atomic_write_feather(df: pd.DataFrame, path: Path, retries: int = 6) -> None:
    """Write Feather atomically with retry to tolerate transient file locks."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df.to_feather(tmp_path)
            tmp_path.replace(path)
            return
        except (OSError, PermissionError) as exc:
            last_err = exc
            if attempt >= retries:
                break
            time.sleep(0.15 * attempt)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    if last_err is not None:
        raise last_err


def _empty_snapshot_frame() -> pd.DataFrame:
    """Return empty DataFrame with the snapshot schema."""
    cols = ["date", "best_bid", "best_ask", "best_bid_size", "best_ask_size",
            "mid_price", "quoted_spread", "quoted_spread_bps",
            "bid_ask_size_imbalance"]
    for n in _DEPTH_LEVELS:
        cols += [
            f"bid_depth_{n}", f"ask_depth_{n}",
            f"depth_imbalance_{n}",
        ]
    cols += ["weighted_mid", "book_pressure", "bid_slope", "ask_slope"]
    return pd.DataFrame(columns=cols)


def _empty_feature_frame() -> pd.DataFrame:
    """Return empty DataFrame with the rolling-feature schema."""
    return pd.DataFrame(columns=["date"] + orderbook_feature_columns())


def orderbook_feature_columns() -> list[str]:
    """Return the list of L1/L2 feature column names fed to the ML model."""
    cols = []
    # L1 instant features
    cols += [
        "ob_quoted_spread_bps",
        "ob_bid_ask_size_imbalance",
    ]
    # L2 instant features
    for n in _DEPTH_LEVELS:
        cols += [
            f"ob_depth_imbalance_{n}",
        ]
    cols += [
        "ob_book_pressure",
        "ob_weighted_mid_offset_bps",
    ]
    # Rolling features (15m and 1h)
    for window_label in ["15m", "1h"]:
        cols += [
            f"ob_spread_mean_{window_label}",
            f"ob_spread_std_{window_label}",
            f"ob_bid_ask_imb_mean_{window_label}",
            f"ob_depth_imb_5_mean_{window_label}",
            f"ob_depth_imb_20_mean_{window_label}",
            f"ob_pressure_mean_{window_label}",
            f"ob_wmid_offset_mean_{window_label}",
        ]
    cols += ["ob_has_data"]
    return cols


def fetch_orderbook_snapshot(
    exchange: ccxt.Exchange,
    symbol: str = "ETH/USDT",
    depth: int = _MAX_DEPTH,
) -> dict:
    """Fetch a single order book snapshot and compute L1+L2 metrics.

    Returns a flat dict suitable for appending to the snapshot DataFrame.
    """
    book = exchange.fetch_order_book(symbol, limit=depth)
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    ts = book.get("timestamp") or int(time.time() * 1000)

    snap = {"date": pd.Timestamp(ts, unit="ms", tz="UTC")}

    if not bids or not asks:
        # Empty book – fill with NaN
        for col in _empty_snapshot_frame().columns:
            if col != "date":
                snap[col] = np.nan
        return snap

    # ── L1: top of book ────────────────────────────────────────────
    best_bid, best_bid_size = float(bids[0][0]), float(bids[0][1])
    best_ask, best_ask_size = float(asks[0][0]), float(asks[0][1])
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    spread_bps = (spread / mid) * 10_000 if mid > 0 else 0.0

    total_l1 = best_bid_size + best_ask_size
    imb_l1 = (best_bid_size - best_ask_size) / total_l1 if total_l1 > 0 else 0.0

    snap.update({
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_size": best_bid_size,
        "best_ask_size": best_ask_size,
        "mid_price": mid,
        "quoted_spread": spread,
        "quoted_spread_bps": spread_bps,
        "bid_ask_size_imbalance": imb_l1,
    })

    # ── L2: depth aggregations ─────────────────────────────────────
    bid_prices = np.array([float(b[0]) for b in bids[:_MAX_DEPTH]])
    bid_sizes = np.array([float(b[1]) for b in bids[:_MAX_DEPTH]])
    ask_prices = np.array([float(a[0]) for a in asks[:_MAX_DEPTH]])
    ask_sizes = np.array([float(a[1]) for a in asks[:_MAX_DEPTH]])

    for n in _DEPTH_LEVELS:
        bd = float(bid_sizes[:n].sum()) if len(bid_sizes) >= n else float(bid_sizes.sum())
        ad = float(ask_sizes[:n].sum()) if len(ask_sizes) >= n else float(ask_sizes.sum())
        total = bd + ad
        imb = (bd - ad) / total if total > 0 else 0.0
        snap[f"bid_depth_{n}"] = bd
        snap[f"ask_depth_{n}"] = ad
        snap[f"depth_imbalance_{n}"] = imb

    # Weighted mid-price (size-weighted)
    if best_bid_size + best_ask_size > 0:
        wmid = (best_bid * best_ask_size + best_ask * best_bid_size) / (best_bid_size + best_ask_size)
    else:
        wmid = mid
    snap["weighted_mid"] = wmid

    # Book pressure: ratio of total bid depth to total depth (top 20 levels)
    total_bid = float(bid_sizes.sum())
    total_ask = float(ask_sizes.sum())
    total_both = total_bid + total_ask
    snap["book_pressure"] = total_bid / total_both if total_both > 0 else 0.5

    # Depth slope: how fast depth decays away from mid
    # Simple linear regression of cumulative size vs price distance
    if len(bid_prices) >= 3:
        bid_dists = np.abs(bid_prices - mid)
        bid_cumsize = np.cumsum(bid_sizes)
        if bid_dists.std() > 0:
            snap["bid_slope"] = float(np.polyfit(bid_dists, bid_cumsize, 1)[0])
        else:
            snap["bid_slope"] = 0.0
    else:
        snap["bid_slope"] = 0.0

    if len(ask_prices) >= 3:
        ask_dists = np.abs(ask_prices - mid)
        ask_cumsize = np.cumsum(ask_sizes)
        if ask_dists.std() > 0:
            snap["ask_slope"] = float(np.polyfit(ask_dists, ask_cumsize, 1)[0])
        else:
            snap["ask_slope"] = 0.0
    else:
        snap["ask_slope"] = 0.0

    return snap


def collect_orderbook_snapshot(
    exchange: ccxt.Exchange,
    symbol: str = "ETH/USDT",
) -> pd.DataFrame:
    """Fetch one snapshot and return as a single-row DataFrame."""
    snap = fetch_orderbook_snapshot(exchange, symbol, depth=_MAX_DEPTH)
    return pd.DataFrame([snap])


def append_snapshot(
    new_snap: pd.DataFrame,
    path: Optional[Path] = None,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    """Append a snapshot row to the persistent feather file.

    Keeps at most ``max_rows`` most recent snapshots (≈ 173 days at 5min cadence).
    """
    path = Path(path or ORDERBOOK_SNAPSHOT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            existing = pf.read_feather(str(path), memory_map=False)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, new_snap], ignore_index=True)
    else:
        combined = new_snap.copy()

    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    if len(combined) > max_rows:
        combined = combined.iloc[-max_rows:]

    _atomic_write_feather(combined, path, retries=5)
    return combined


def build_orderbook_features(
    snapshots: pd.DataFrame,
    candle_dates: Optional[Iterable] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Compute rolling L1/L2 features from snapshot history.

    Aligns features to the ``candle_dates`` timeline (5min candle boundaries)
    via forward-fill so they join cleanly with OHLCV features.
    """
    if snapshots.empty:
        summary = _make_summary(0, None, None)
        if candle_dates is not None:
            dates = pd.Series(list(candle_dates), name="date")
            dates = pd.to_datetime(dates, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
            empty = _empty_feature_frame()
            empty = pd.DataFrame({"date": dates})
            for col in orderbook_feature_columns():
                empty[col] = np.nan
            return empty, summary
        return _empty_feature_frame(), summary

    df = snapshots.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Resample to 5min candle boundaries (use last snapshot per candle)
    df = df.set_index("date")
    resampled = df.resample("5min").last().dropna(how="all")

    if resampled.empty:
        summary = _make_summary(len(snapshots), df.index.min(), df.index.max())
        return _empty_feature_frame(), summary

    feat = pd.DataFrame(index=resampled.index)

    # ── Instant L1 features ─────────────────────────────────────────
    feat["ob_quoted_spread_bps"] = resampled["quoted_spread_bps"]
    feat["ob_bid_ask_size_imbalance"] = resampled["bid_ask_size_imbalance"]

    # ── Instant L2 features ─────────────────────────────────────────
    for n in _DEPTH_LEVELS:
        feat[f"ob_depth_imbalance_{n}"] = resampled[f"depth_imbalance_{n}"]

    feat["ob_book_pressure"] = resampled["book_pressure"]

    mid = resampled["mid_price"].replace(0, np.nan)
    wmid_offset = ((resampled["weighted_mid"] - resampled["mid_price"]) / mid * 10_000).fillna(0)
    feat["ob_weighted_mid_offset_bps"] = wmid_offset

    # ── Rolling features ────────────────────────────────────────────
    for window, label in [(_ROLL_15M, "15m"), (_ROLL_1H, "1h")]:
        feat[f"ob_spread_mean_{label}"] = feat["ob_quoted_spread_bps"].rolling(window, min_periods=1).mean()
        feat[f"ob_spread_std_{label}"] = feat["ob_quoted_spread_bps"].rolling(window, min_periods=1).std().fillna(0)
        feat[f"ob_bid_ask_imb_mean_{label}"] = feat["ob_bid_ask_size_imbalance"].rolling(window, min_periods=1).mean()
        feat[f"ob_depth_imb_5_mean_{label}"] = feat["ob_depth_imbalance_5"].rolling(window, min_periods=1).mean()
        feat[f"ob_depth_imb_20_mean_{label}"] = feat["ob_depth_imbalance_20"].rolling(window, min_periods=1).mean()
        feat[f"ob_pressure_mean_{label}"] = feat["ob_book_pressure"].rolling(window, min_periods=1).mean()
        feat[f"ob_wmid_offset_mean_{label}"] = feat["ob_weighted_mid_offset_bps"].rolling(window, min_periods=1).mean()

    feat["ob_has_data"] = 1.0

    feat = feat.reset_index(names="date")
    # Normalize both sides to identical UTC nanosecond dtype for merge_asof.
    feat["date"] = pd.to_datetime(feat["date"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")

    # ── Align to candle grid ────────────────────────────────────────
    if candle_dates is not None:
        candle_ts = pd.Series(list(candle_dates), name="date")
        candle_ts = (
            pd.to_datetime(candle_ts, utc=True, errors="coerce")
            .astype("datetime64[ns, UTC]")
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        grid = pd.DataFrame({"date": candle_ts})
        feat = pd.merge_asof(
            grid.sort_values("date"),
            feat.sort_values("date"),
            on="date",
            direction="backward",
        )

    n_snap = len(snapshots)
    cov_start = df.index.min() if not df.empty else None
    cov_end = df.index.max() if not df.empty else None
    summary = _make_summary(n_snap, cov_start, cov_end)

    return feat, summary


def _make_summary(n_snapshots: int, start, end) -> dict:
    return {
        "snapshots_total": n_snapshots,
        "feature_count": len(orderbook_feature_columns()),
        "coverage_start": start.isoformat() if start is not None else None,
        "coverage_end": end.isoformat() if end is not None else None,
    }


def export_orderbook_features(
    candle_dates: Optional[Iterable] = None,
    snapshot_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> dict:
    """Load persisted snapshots, build features, write feather + summary JSON."""
    snapshot_path = Path(snapshot_path or ORDERBOOK_SNAPSHOT_PATH)
    features_path = Path(features_path or ORDERBOOK_FEATURES_PATH)
    summary_path = Path(summary_path or ORDERBOOK_SUMMARY_PATH)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    snapshots = pd.DataFrame()
    if snapshot_path.exists():
        try:
            snapshots = pf.read_feather(str(snapshot_path), memory_map=False)
        except Exception:
            snapshots = pd.DataFrame()

    feat_df, summary = build_orderbook_features(snapshots, candle_dates=candle_dates)
    _atomic_write_feather(feat_df, features_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_or_build_orderbook_features(
    candle_dates: Optional[Iterable] = None,
    prefer_cached: bool = True,
) -> pd.DataFrame:
    """Load cached orderbook features, or build from snapshots if missing."""
    features_path = ORDERBOOK_FEATURES_PATH
    ob_df = pd.DataFrame()

    if prefer_cached and features_path.exists():
        try:
            ob_df = pf.read_feather(str(features_path), memory_map=False)
        except Exception:
            ob_df = pd.DataFrame()

    if ob_df.empty:
        snapshot_path = ORDERBOOK_SNAPSHOT_PATH
        snapshots = pd.DataFrame()
        if snapshot_path.exists():
            try:
                snapshots = pf.read_feather(str(snapshot_path), memory_map=False)
            except Exception:
                pass
        ob_df, _ = build_orderbook_features(snapshots, candle_dates=candle_dates)
    elif candle_dates is not None:
        ob_df = _align_to_candles(candle_dates, ob_df)

    if not ob_df.empty and "date" in ob_df.columns:
        ob_df["date"] = pd.to_datetime(ob_df["date"], utc=True, errors="coerce")
        ob_df = ob_df.sort_values("date").reset_index(drop=True)

    return ob_df


def _align_to_candles(candle_dates: Iterable, ob_df: pd.DataFrame) -> pd.DataFrame:
    """Align cached feature frame to a new candle grid via merge_asof."""
    candle_ts = pd.Series(list(candle_dates), name="date")
    candle_ts = (
        pd.to_datetime(candle_ts, utc=True, errors="coerce")
        .astype("datetime64[ns, UTC]")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    grid = pd.DataFrame({"date": candle_ts})
    ob_df["date"] = pd.to_datetime(ob_df["date"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    return pd.merge_asof(
        grid.sort_values("date"),
        ob_df.sort_values("date"),
        on="date",
        direction="backward",
    )


def get_latest_orderbook_state(ob_df: pd.DataFrame) -> dict:
    """Return the most recent orderbook metrics for real-time overlays."""
    defaults = {
        "ob_quoted_spread_bps": 0.0,
        "ob_bid_ask_size_imbalance": 0.0,
        "ob_depth_imbalance_5": 0.0,
        "ob_depth_imbalance_20": 0.0,
        "ob_book_pressure": 0.5,
        "ob_weighted_mid_offset_bps": 0.0,
        "ob_spread_mean_1h": 0.0,
        "ob_pressure_mean_1h": 0.5,
        "ob_has_data": False,
    }
    if ob_df is None or ob_df.empty:
        return defaults

    row = ob_df.iloc[-1]
    return {
        "ob_quoted_spread_bps": float(row.get("ob_quoted_spread_bps", 0) or 0),
        "ob_bid_ask_size_imbalance": float(row.get("ob_bid_ask_size_imbalance", 0) or 0),
        "ob_depth_imbalance_5": float(row.get("ob_depth_imbalance_5", 0) or 0),
        "ob_depth_imbalance_20": float(row.get("ob_depth_imbalance_20", 0) or 0),
        "ob_book_pressure": float(row.get("ob_book_pressure", 0.5) or 0.5),
        "ob_weighted_mid_offset_bps": float(row.get("ob_weighted_mid_offset_bps", 0) or 0),
        "ob_spread_mean_1h": float(row.get("ob_spread_mean_1h", 0) or 0),
        "ob_pressure_mean_1h": float(row.get("ob_pressure_mean_1h", 0.5) or 0.5),
        "ob_has_data": bool(row.get("ob_has_data", 0)),
    }
