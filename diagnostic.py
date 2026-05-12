"""
Deep diagnostic analysis of the current ML model.
Run as a standalone script to audit prediction quality, feature importance,
and potential issues before deploying updates.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from config import DATA_DIR, load_models
from execution_learning import load_or_build_execution_features
from orderbook_features import load_or_build_orderbook_features
from features import compute_features, compute_labels, get_feature_columns


def main():
    # ── Load data ───────────────────────────────────────────────────
    df = pf.read_feather(str(DATA_DIR / "ETH_USDT-5m.feather"), memory_map=False)
    btc_df = pf.read_feather(str(DATA_DIR / "BTC_USDT-5m.feather"), memory_map=False)

    execution_df = load_or_build_execution_features(candle_dates=df["date"], prefer_cached=True)
    orderbook_df = load_or_build_orderbook_features(candle_dates=df["date"], prefer_cached=True)
    feat_df = compute_features(df, btc_df, execution_df=execution_df, orderbook_df=orderbook_df)
    feat_df = compute_labels(feat_df, horizon=6)
    feature_cols = get_feature_columns(feat_df)
    feat_df = feat_df.dropna(
        subset=feature_cols + ["direction", "future_volatility"]
    ).reset_index(drop=True)

    # ── Load models via shared loader (validates feature counts) ───
    dir_model, _, vol_model, _, _ = load_models()

    # ── Slice test set ──────────────────────────────────────────────
    n = len(feat_df)
    test_start = int(n * 0.85)
    test_df = feat_df.iloc[test_start:].copy()

    X_test = test_df[feature_cols].values
    y_true = test_df["direction"].values
    dir_proba = dir_model.predict_proba(X_test)[:, 1]
    vol_pred = vol_model.predict(X_test)
    pred = (dir_proba > 0.5).astype(int)

    # 1. Prediction distribution
    print("=== PREDICTION DISTRIBUTION ===")
    print(f"Mean: {dir_proba.mean():.4f}, Std: {dir_proba.std():.4f}")
    for q in [5, 25, 50, 75, 95]:
        print(f"  P{q}: {np.percentile(dir_proba, q):.4f}")

    # 2. Accuracy by confidence bucket
    print("\n=== ACCURACY BY CONFIDENCE ===")
    for lo, hi in [(0.0, 0.45), (0.45, 0.48), (0.48, 0.52), (0.52, 0.55), (0.55, 1.0)]:
        mask = (dir_proba >= lo) & (dir_proba < hi)
        if mask.sum() > 0:
            acc = (y_true[mask] == (dir_proba[mask] > 0.5).astype(int)).mean()
            print(f"  [{lo:.2f}-{hi:.2f}): n={mask.sum():>5}, acc={acc:.4f}")

    # 3. Accuracy by volatility regime
    vol = test_df["realized_vol_12"].values
    vol_q = np.nanpercentile(vol, [25, 50, 75])
    for label, lo, hi in [
        ("Low", 0, vol_q[0]),
        ("Med-Low", vol_q[0], vol_q[1]),
        ("Med-High", vol_q[1], vol_q[2]),
        ("High", vol_q[2], 1),
    ]:
        mask = (vol >= lo) & (vol < hi) if hi < 1 else (vol >= lo)
        if mask.sum() > 0:
            acc = (y_true[mask] == pred[mask]).mean()
            print(f"  {label:>10}: acc={acc:.4f} (n={mask.sum()})")

    # 4. Accuracy by time-of-day
    hours = pd.to_datetime(test_df["date"]).dt.hour.values
    print("\n=== ACCURACY BY HOUR BLOCK ===")
    for h0, h1, name in [(0, 6, "Asia"), (6, 12, "EU"), (12, 18, "US Day"), (18, 24, "US Eve")]:
        mask = (hours >= h0) & (hours < h1)
        if mask.sum() > 0:
            acc = (y_true[mask] == pred[mask]).mean()
            print(f"  {name:>10} ({h0:02d}-{h1:02d}): acc={acc:.4f} (n={mask.sum()})")

    # 5. Feature importance
    imp = pd.Series(dir_model.feature_importances_, index=feature_cols)
    n_zero = (imp == 0).sum()
    print(f"\n=== FEATURE IMPORTANCE ===")
    print(f"Zero-importance features: {n_zero} / {len(feature_cols)}")
    print(f"Top 10: {imp.nlargest(10).index.tolist()}")

    # 6. Feature-target correlation
    print("\n=== FEATURE-TARGET CORRELATION (top20) ===")
    corr_with_dir = (
        feat_df[feature_cols + ["direction"]]
        .corr()["direction"]
        .drop("direction")
        .abs()
        .sort_values(ascending=False)
    )
    for name, val in corr_with_dir.head(20).items():
        print(f"  {name:>25}: {val:.4f}")

    # 7. Inter-feature correlation (redundancy)
    feat_corr = feat_df[feature_cols].corr().abs()
    np.fill_diagonal(feat_corr.values, 0)
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if feat_corr.iloc[i, j] > 0.95:
                high_corr_pairs.append(
                    (feature_cols[i], feature_cols[j], feat_corr.iloc[i, j])
                )
    print(f"\n=== HIGHLY CORRELATED PAIRS (>0.95) ===")
    for a, b, c in sorted(high_corr_pairs, key=lambda x: -x[2])[:15]:
        print(f"  {a:>25} <-> {b:<25} {c:.3f}")

    # 8. Autocorrelation analysis
    print(f"\n=== PREDICTION DYNAMICS ===")
    proba_series = pd.Series(dir_proba)
    print(f"Autocorrelation(1): {proba_series.autocorr(1):.4f}")
    print(f"Autocorrelation(5): {proba_series.autocorr(5):.4f}")
    print(f"Autocorrelation(12): {proba_series.autocorr(12):.4f}")

    # 9. Volatility model analysis
    y_vol_true = test_df["future_volatility"].values
    vol_corr = np.corrcoef(vol_pred, y_vol_true)[0, 1]
    print(f"\n=== VOLATILITY MODEL ===")
    print(f"Prediction-Actual correlation: {vol_corr:.4f}")
    print(f"Mean pred vol: {vol_pred.mean():.6f}, Mean actual: {np.nanmean(y_vol_true):.6f}")
    print(f"Std pred vol:  {vol_pred.std():.6f}, Std actual:  {np.nanstd(y_vol_true):.6f}")

    # 10. Horizon analysis
    print("\n=== HORIZON ANALYSIS ===")
    for h in [3, 6, 12, 24]:
        fret = df["close"].shift(-h) / df["close"] - 1
        fdir = (fret > 0).astype(int)
        print(f"  horizon={h:>2} ({h * 5:>3}min): up_pct={fdir.mean():.4f}, ret_std={fret.std():.6f}")

    # 11. Cross-asset check
    sol_path = DATA_DIR / "SOL_USDT-5m.feather"
    if sol_path.exists():
        sol_df = pf.read_feather(str(sol_path), memory_map=False)
        sol_ret = sol_df["close"].pct_change(1)
        eth_ret = df["close"].pct_change(1)
        sol_eth_corr = sol_ret.corr(eth_ret)
        print(f"\n=== CROSS-ASSET ===")
        print(f"ETH-SOL return correlation: {sol_eth_corr:.4f}")
        print(f'ETH-BTC return correlation: {eth_ret.corr(btc_df["close"].pct_change(1)):.4f}')


if __name__ == "__main__":
    main()
