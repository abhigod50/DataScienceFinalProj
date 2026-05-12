"""
Standalone market data refresher.
Fetches OHLCV directly from exchange via ccxt and stores feather files
under data/<exchange>/PAIR-TIMEFRAME.feather inside this project.
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import ccxt
import pandas as pd
import pyarrow.feather as pf

from shared.paths import DATA_ROOT


def fetch_ohlcv_full(exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int):
    """Fetch paginated OHLCV data from since_ms until now."""
    all_rows = []
    cursor = since_ms

    while cursor < until_ms:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not rows:
            break

        all_rows.extend(rows)
        last_ts = rows[-1][0]

        # Advance cursor by one candle to avoid duplicates.
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

        # Respect exchange limits.
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Refresh OHLCV data for Hummingbot ML training")
    parser.add_argument("--exchange", default="binanceus")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--pairs", nargs="+", required=True)
    args = parser.parse_args()

    # Allow override via env var so retrain.py can redirect Binance output to
    # BINANCE_DATA_DIR without changing the command interface.
    data_root = os.environ.get("ML_DATA_DIR", "")
    if data_root:
        base_dir = Path(data_root)
    else:
        base_dir = DATA_ROOT / args.exchange
    base_dir.mkdir(parents=True, exist_ok=True)

    ex_class = getattr(ccxt, args.exchange)
    ex = ex_class({"enableRateLimit": True})

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - args.days * 24 * 60 * 60 * 1000

    print(f"Refreshing {len(args.pairs)} pairs from {args.exchange} ({args.timeframe}, last {args.days} days)")
    for pair in args.pairs:
        print(f"Fetching {pair} ...")
        df = fetch_ohlcv_full(ex, pair, args.timeframe, since_ms, now_ms)
        if df.empty:
            print(f"  WARNING: no candles received for {pair}")
            continue

        fname = f"{pair.replace('/', '_')}-{args.timeframe}.feather"
        out = base_dir / fname
        tmp = out.with_suffix(".feather.tmp")
        if out.exists():
            existing = pf.read_feather(str(out), memory_map=False)
            merged = pd.concat([existing, df], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], utc=True)
            merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            merged.to_feather(tmp)
            shutil.move(str(tmp), str(out))
            print(f"  Merged {len(df)} new rows -> {out} (total {len(merged)})")
        else:
            df.to_feather(tmp)
            shutil.move(str(tmp), str(out))
            print(f"  Saved {len(df)} rows -> {out}")

    if hasattr(ex, "close"):
        ex.close()


if __name__ == "__main__":
    main()
