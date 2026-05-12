"""
Execution-learning utilities.

Builds a time-aligned feature timeline from live Hummingbot paper-trading
order/fill history so the ML stack can incorporate recent execution quality.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from config import EXECUTION_DB_PATH, EXECUTION_FEATURES_PATH, EXECUTION_SUMMARY_PATH

_SCALE = 1_000_000.0
_ROLL_15M = 3
_ROLL_1H = 12


def _empty_execution_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date",
        "exec_orders_created_15m",
        "exec_orders_created_1h",
        "exec_fill_count_1h",
        "exec_fill_rate_1h",
        "exec_cancel_rate_1h",
        "exec_avg_order_lifetime_sec_1h",
        "exec_fill_notional_quote_1h",
        "exec_fee_quote_1h",
        "exec_quote_imbalance_1h",
        "exec_fill_imbalance_1h",
        "exec_buy_fill_share_1h",
        "exec_samples_1h",
        "exec_has_history",
    ])


def execution_feature_columns() -> list[str]:
    return [column for column in _empty_execution_frame().columns if column != "date"]


def _ensure_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _read_sql_table(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn)
    except Exception as exc:
        import traceback
        print(f"[execution_learning] SQL query failed: {exc}\n{traceback.format_exc()}")
        return pd.DataFrame()


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    empty = df.iloc[0:0].copy()
    empty.attrs.update(df.attrs)
    return empty


def _atomic_write_feather(df: pd.DataFrame, path: Path, retries: int = 6) -> None:
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


def _normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders

    raw_count = len(orders)
    orders = orders.copy()
    order_ids = orders["id"].fillna("").astype(str)
    creation_ts = _coerce_numeric(orders["creation_timestamp"])
    last_update_ts = _coerce_numeric(orders["last_update_timestamp"]).fillna(creation_ts)
    amount = _coerce_numeric(orders["amount"])
    price = _coerce_numeric(orders["price"])
    valid_side = order_ids.str.startswith(("buy://", "sell://"))
    valid_mask = valid_side & creation_ts.notna() & amount.notna() & price.notna()

    orders.attrs["rows_loaded"] = raw_count
    orders.attrs["invalid_rows_dropped"] = int((~valid_mask).sum())
    if not valid_mask.any():
        return _empty_like(orders)

    orders = orders.loc[valid_mask].copy()
    orders.attrs["rows_loaded"] = raw_count
    orders.attrs["invalid_rows_dropped"] = int((~valid_mask).sum())
    creation_ts = creation_ts.loc[valid_mask]
    last_update_ts = last_update_ts.loc[valid_mask]
    amount = amount.loc[valid_mask]
    price = price.loc[valid_mask]

    # Hummingbot stores timestamps as millisecond integers; must specify unit='ms'
    # pd.to_datetime without unit= treats integers as nanoseconds (puts dates in 1970).
    orders["date"] = pd.to_datetime(creation_ts, unit="ms", utc=True, errors="coerce")
    orders["last_update_date"] = pd.to_datetime(last_update_ts, unit="ms", utc=True, errors="coerce")
    orders["amount_base"] = amount / _SCALE
    orders["price_quote"] = price / _SCALE
    orders["notional_quote"] = orders["amount_base"] * orders["price_quote"]
    orders["side"] = np.where(order_ids.loc[valid_mask].str.startswith("buy://"), "BUY", "SELL")
    orders["last_status"] = orders["last_status"].fillna("").astype(str)
    orders["is_cancel"] = orders["last_status"].str.contains("Cancelled", case=False, na=False).astype(int)
    # Hummingbot uses "SellOrderCompleted" / "BuyOrderCompleted", not "Filled"
    orders["is_fill"] = orders["last_status"].str.contains("Completed", case=False, na=False).astype(int)
    orders["lifetime_sec"] = ((last_update_ts - creation_ts) / 1000.0).clip(lower=0.0)
    return orders.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _infer_fill_side(fills: pd.DataFrame) -> pd.Series:
    trade_type = fills["trade_type"].fillna("").astype(str).str.upper()
    order_id = fills["order_id"].fillna("").astype(str)
    config_file_path = fills["config_file_path"].fillna("").astype(str)
    side = pd.Series(pd.NA, index=fills.index, dtype="object")
    side.loc[trade_type.isin(["BUY", "SELL"])] = trade_type.loc[trade_type.isin(["BUY", "SELL"])]
    buy_mask = side.isna() & (order_id.str.startswith("buy://") | config_file_path.str.startswith("buy://"))
    sell_mask = side.isna() & (order_id.str.startswith("sell://") | config_file_path.str.startswith("sell://"))
    side.loc[buy_mask] = "BUY"
    side.loc[sell_mask] = "SELL"
    return side


def _normalize_fills(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return fills

    raw_count = len(fills)
    fills = fills.copy()
    timestamp = _coerce_numeric(fills["timestamp"])
    amount = _coerce_numeric(fills["amount"])
    price = _coerce_numeric(fills["price"])
    fee_quote = _coerce_numeric(fills["trade_fee_in_quote"]).fillna(0.0)
    side = _infer_fill_side(fills)
    valid_mask = timestamp.notna() & amount.notna() & price.notna() & side.isin(["BUY", "SELL"])

    fills.attrs["rows_loaded"] = raw_count
    fills.attrs["invalid_rows_dropped"] = int((~valid_mask).sum())
    if not valid_mask.any():
        return _empty_like(fills)

    fills = fills.loc[valid_mask].copy()
    fills.attrs["rows_loaded"] = raw_count
    fills.attrs["invalid_rows_dropped"] = int((~valid_mask).sum())
    timestamp = timestamp.loc[valid_mask]
    amount = amount.loc[valid_mask]
    price = price.loc[valid_mask]
    fee_quote = fee_quote.loc[valid_mask]
    side = side.loc[valid_mask]

    fills["date"] = pd.to_datetime(timestamp, unit="ms", utc=True, errors="coerce")
    fills["amount_base"] = amount / _SCALE
    fills["price_quote"] = price / _SCALE
    fills["notional_quote"] = fills["amount_base"] * fills["price_quote"]
    fills["fee_quote"] = fee_quote / _SCALE
    fills["side"] = side.astype(str)
    return fills.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_execution_tables(db_path: Optional[Path] = None, symbol: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_path = Path(db_path or EXECUTION_DB_PATH)
    if not db_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    try:
        # Order IDs encode the pair as e.g. "buy://ETH-USDT/..." so we can
        # filter per-pair when symbol is provided (e.g. "ETH-USDT").
        if symbol:
            order_where = f"WHERE id LIKE '%{symbol}%'"
            fill_where = f"WHERE symbol = '{symbol}'"
        else:
            order_where = ""
            fill_where = ""
        orders = _read_sql_table(
            conn,
            f"""
            SELECT
                id,
                creation_timestamp,
                last_update_timestamp,
                amount,
                price,
                last_status
            FROM [Order]
            {order_where}
            ORDER BY creation_timestamp
            """,
        )
        fills = _read_sql_table(
            conn,
            f"""
            SELECT
                config_file_path,
                symbol,
                timestamp,
                order_id,
                trade_type,
                amount,
                price,
                trade_fee_in_quote
            FROM TradeFill
            {fill_where}
            ORDER BY timestamp
            """,
        )
    finally:
        conn.close()

    return _normalize_orders(orders), _normalize_fills(fills)


def build_execution_feature_timeline(
    candle_dates: Optional[Iterable] = None,
    db_path: Optional[Path] = None,
    symbol: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    orders, fills = load_execution_tables(db_path=db_path, symbol=symbol)

    if candle_dates is not None:
        base_dates = pd.Series(list(candle_dates), name="date")
        base_dates = _ensure_utc(base_dates).dropna().drop_duplicates().sort_values()
    else:
        timestamps = []
        if not orders.empty:
            timestamps.append(orders["date"])
        if not fills.empty:
            timestamps.append(fills["date"])
        if not timestamps:
            return _empty_execution_frame(), {
                "orders_total": 0,
                "fills_total": 0,
                "coverage_start": None,
                "coverage_end": None,
            }
        combined = pd.concat(timestamps).sort_values()
        start = combined.iloc[0].floor("5min")
        end = combined.iloc[-1].ceil("5min")
        base_dates = pd.Series(pd.date_range(start=start, end=end, freq="5min", tz="UTC"), name="date")

    timeline = pd.DataFrame({"date": base_dates})
    if timeline.empty:
        return _empty_execution_frame(), {
            "orders_total": int(len(orders)),
            "fills_total": int(len(fills)),
            "coverage_start": None,
            "coverage_end": None,
        }

    index = pd.DatetimeIndex(timeline["date"])
    agg = pd.DataFrame(index=index)

    if not orders.empty:
        orders_idx = orders.set_index("date")
        order_roll = pd.DataFrame(index=index)
        order_roll["orders_created"] = orders_idx["id"].resample("5min").count().reindex(index, fill_value=0)
        order_roll["cancelled"] = orders_idx["is_cancel"].resample("5min").sum().reindex(index, fill_value=0)
        order_roll["filled_orders"] = orders_idx["is_fill"].resample("5min").sum().reindex(index, fill_value=0)
        order_roll["buy_orders"] = (orders_idx["side"] == "BUY").astype(int).resample("5min").sum().reindex(index, fill_value=0)
        order_roll["sell_orders"] = (orders_idx["side"] == "SELL").astype(int).resample("5min").sum().reindex(index, fill_value=0)
        order_roll["completed_orders"] = (
            (orders_idx["is_cancel"] + orders_idx["is_fill"]) > 0
        ).astype(int).resample("5min").sum().reindex(index, fill_value=0)
        order_roll["lifetime_sec_sum"] = (
            orders_idx["lifetime_sec"] * ((orders_idx["is_cancel"] + orders_idx["is_fill"]) > 0).astype(int)
        ).resample("5min").sum().reindex(index, fill_value=0.0)
    else:
        order_roll = pd.DataFrame(0.0, index=index, columns=[
            "orders_created", "cancelled", "filled_orders", "buy_orders", "sell_orders", "completed_orders", "lifetime_sec_sum"
        ])

    if not fills.empty:
        fills_idx = fills.set_index("date")
        fill_roll = pd.DataFrame(index=index)
        fill_roll["fills"] = fills_idx["side"].resample("5min").count().reindex(index, fill_value=0)
        fill_roll["buy_fills"] = (fills_idx["side"] == "BUY").astype(int).resample("5min").sum().reindex(index, fill_value=0)
        fill_roll["sell_fills"] = (fills_idx["side"] == "SELL").astype(int).resample("5min").sum().reindex(index, fill_value=0)
        fill_roll["fill_notional_quote"] = fills_idx["notional_quote"].resample("5min").sum().reindex(index, fill_value=0.0)
        fill_roll["fee_quote"] = fills_idx["fee_quote"].resample("5min").sum().reindex(index, fill_value=0.0)
    else:
        fill_roll = pd.DataFrame(0.0, index=index, columns=[
            "fills", "buy_fills", "sell_fills", "fill_notional_quote", "fee_quote"
        ])

    agg["exec_orders_created_15m"] = order_roll["orders_created"].rolling(_ROLL_15M, min_periods=1).sum()
    agg["exec_orders_created_1h"] = order_roll["orders_created"].rolling(_ROLL_1H, min_periods=1).sum()
    agg["exec_fill_count_1h"] = fill_roll["fills"].rolling(_ROLL_1H, min_periods=1).sum()
    agg["exec_cancel_rate_1h"] = _safe_ratio(
        order_roll["cancelled"].rolling(_ROLL_1H, min_periods=1).sum(),
        agg["exec_orders_created_1h"],
    )
    agg["exec_fill_rate_1h"] = _safe_ratio(agg["exec_fill_count_1h"], agg["exec_orders_created_1h"])
    agg["exec_avg_order_lifetime_sec_1h"] = _safe_ratio(
        order_roll["lifetime_sec_sum"].rolling(_ROLL_1H, min_periods=1).sum(),
        order_roll["completed_orders"].rolling(_ROLL_1H, min_periods=1).sum(),
    )
    agg["exec_fill_notional_quote_1h"] = fill_roll["fill_notional_quote"].rolling(_ROLL_1H, min_periods=1).sum()
    agg["exec_fee_quote_1h"] = fill_roll["fee_quote"].rolling(_ROLL_1H, min_periods=1).sum()
    agg["exec_quote_imbalance_1h"] = _safe_ratio(
        order_roll["buy_orders"].rolling(_ROLL_1H, min_periods=1).sum()
        - order_roll["sell_orders"].rolling(_ROLL_1H, min_periods=1).sum(),
        agg["exec_orders_created_1h"],
    )
    agg["exec_fill_imbalance_1h"] = _safe_ratio(
        fill_roll["buy_fills"].rolling(_ROLL_1H, min_periods=1).sum()
        - fill_roll["sell_fills"].rolling(_ROLL_1H, min_periods=1).sum(),
        agg["exec_fill_count_1h"],
    )
    agg["exec_buy_fill_share_1h"] = _safe_ratio(
        fill_roll["buy_fills"].rolling(_ROLL_1H, min_periods=1).sum(),
        agg["exec_fill_count_1h"],
    )
    agg["exec_samples_1h"] = agg["exec_orders_created_1h"] + agg["exec_fill_count_1h"]
    agg["exec_has_history"] = (agg["exec_orders_created_1h"] > 0).astype(float)

    agg = agg.reset_index(names="date")
    summary = {
        "orders_total": int(len(orders)),
        "fills_total": int(len(fills)),
        "orders_invalid_dropped": int(orders.attrs.get("invalid_rows_dropped", 0)),
        "fills_invalid_dropped": int(fills.attrs.get("invalid_rows_dropped", 0)),
        "feature_count": len(execution_feature_columns()),
        "coverage_start": agg["date"].min().isoformat() if not agg.empty else None,
        "coverage_end": agg["date"].max().isoformat() if not agg.empty else None,
        "latest_fill_rate_1h": float(agg["exec_fill_rate_1h"].iloc[-1]) if not agg.empty else 0.0,
        "latest_cancel_rate_1h": float(agg["exec_cancel_rate_1h"].iloc[-1]) if not agg.empty else 0.0,
    }
    return agg, summary


def export_execution_features(
    candle_dates: Optional[Iterable] = None,
    db_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> dict:
    features_path = Path(features_path or EXECUTION_FEATURES_PATH)
    summary_path = Path(summary_path or EXECUTION_SUMMARY_PATH)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    timeline, summary = build_execution_feature_timeline(candle_dates=candle_dates, db_path=db_path)
    _atomic_write_feather(timeline, features_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_or_build_execution_features(
    candle_dates: Optional[Iterable] = None,
    prefer_cached: bool = True,
    db_path: Optional[Path] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    # When filtering per-pair, skip the cache (it contains all-pair aggregates)
    features_path = EXECUTION_FEATURES_PATH
    execution_df = pd.DataFrame()
    if prefer_cached and not symbol and features_path.exists():
        try:
            execution_df = pf.read_feather(str(features_path), memory_map=False)
        except Exception:
            execution_df = pd.DataFrame()

    if execution_df.empty:
        execution_df, _ = build_execution_feature_timeline(
            candle_dates=candle_dates, db_path=db_path, symbol=symbol,
        )
    elif candle_dates is not None:
        execution_df = align_execution_features(candle_dates, execution_df)

    if not execution_df.empty and "date" in execution_df.columns:
        execution_df["date"] = _ensure_utc(execution_df["date"])
        execution_df = execution_df.sort_values("date").reset_index(drop=True)
    return execution_df


def align_execution_features(candle_dates: Iterable, execution_df: pd.DataFrame) -> pd.DataFrame:
    if execution_df.empty:
        return _empty_execution_frame()

    base = pd.DataFrame({"date": _ensure_utc(pd.Series(list(candle_dates), name="date"))})
    base = base.dropna().drop_duplicates().sort_values("date").reset_index(drop=True)
    exec_df = execution_df.copy()
    exec_df["date"] = _ensure_utc(exec_df["date"])
    exec_df = exec_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return pd.merge_asof(base, exec_df, on="date", direction="backward")


def get_latest_execution_state(execution_df: pd.DataFrame) -> dict:
    if execution_df.empty:
        return {
            "has_history": False,
            "orders_created_1h": 0.0,
            "fill_rate_1h": 0.0,
            "cancel_rate_1h": 0.0,
            "avg_order_lifetime_sec_1h": 0.0,
            "fill_imbalance_1h": 0.0,
        }

    latest = execution_df.sort_values("date").iloc[-1]
    return {
        "has_history": bool(latest.get("exec_has_history", 0) > 0),
        "orders_created_1h": float(latest.get("exec_orders_created_1h", 0.0)),
        "fill_rate_1h": float(latest.get("exec_fill_rate_1h", 0.0)),
        "cancel_rate_1h": float(latest.get("exec_cancel_rate_1h", 0.0)),
        "avg_order_lifetime_sec_1h": float(latest.get("exec_avg_order_lifetime_sec_1h", 0.0)),
        "fill_imbalance_1h": float(latest.get("exec_fill_imbalance_1h", 0.0)),
    }
