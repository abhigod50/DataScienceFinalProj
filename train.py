"""
ML Model Training Pipeline
============================
Walk-forward training with XGBoost GPU for market-making signal generation.
Trains two models:
  1. Direction classifier (up/down probability)
  2. Volatility regressor (expected future price range)
"""

import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    mean_absolute_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit

from config import (
    DATA_DIR, LATEST_MODEL_DIR, MODEL_DIR, PAIR, TIMEFRAME, HORIZON, HORIZON_FAST, HORIZON_MID,
    TRAIN_RATIO, VAL_RATIO,
    DIR3_FLAT_BAND, XGB_DIR_PARAMS, LGB_VOL_PARAMS, XGB_VOL_FALLBACK_PARAMS,
    LGB_AVAILABLE, LGB_IMPORT_ERROR, _IsoCalibrator,
    SAMPLE_DECAY_HALFLIFE_DAYS, BINANCE_DATA_DIR,
)
from execution_learning import execution_feature_columns, export_execution_features, load_or_build_execution_features
from orderbook_features import orderbook_feature_columns, export_orderbook_features, load_or_build_orderbook_features
from features import compute_features, compute_labels, get_feature_columns
from regime_model import apply_regime_model, fit_regime_model, save_regime_model
from neural_model import train_neural_model, save_neural_model
from meta_ensemble import save_meta_ensemble, train_meta_ensemble
from runtime_backends import build_training_backend_params, summarize_selected_backends
from validation import (
    generate_purged_walk_forward_splits,
    summarize_walk_forward_results,
    write_walk_forward_results,
)

# Re-export for retrain.py subprocess compatibility
if LGB_AVAILABLE:
    import lightgbm as lgb
else:
    lgb = None

RESOLVED_XGB_DIR_PARAMS, RESOLVED_LGB_VOL_PARAMS, RESOLVED_XGB_VOL_FALLBACK_PARAMS, TRAINING_BACKEND_SELECTION = (
    build_training_backend_params(
        XGB_DIR_PARAMS,
        LGB_VOL_PARAMS,
        XGB_VOL_FALLBACK_PARAMS,
    )
)


def _load_reference_eth_frame() -> pd.DataFrame | None:
    """Load Coinbase ETH reference data, preferring the current USD symbol."""
    for path in (
        BINANCE_DATA_DIR / f"ETH_USD-{TIMEFRAME}.feather",
        BINANCE_DATA_DIR / f"ETH_USDT-{TIMEFRAME}.feather",
    ):
        if path.exists():
            return pf.read_feather(str(path), memory_map=False)
    return None


def load_data():
    """Load OHLCV data for primary pair, BTC, SOL, and Binance reference."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pair_file = DATA_DIR / f"{PAIR}-{TIMEFRAME}.feather"
    btc_file = DATA_DIR / f"BTC_USDT-{TIMEFRAME}.feather"
    sol_file = DATA_DIR / f"SOL_USDT-{TIMEFRAME}.feather"
    # Binance reference data (optional — provides cross-exchange premium feature)
    df = pf.read_feather(str(pair_file), memory_map=False)
    btc_df = pf.read_feather(str(btc_file), memory_map=False) if btc_file.exists() else None
    sol_df = pf.read_feather(str(sol_file), memory_map=False) if sol_file.exists() else None
    binance_df = _load_reference_eth_frame()

    print(f"Loaded {PAIR}: {len(df)} rows ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")
    if btc_df is not None:
        print(f"Loaded BTC: {len(btc_df)} rows")
    if sol_df is not None:
        print(f"Loaded SOL: {len(sol_df)} rows")
    if binance_df is not None:
        print(f"Loaded Binance ETH (reference): {len(binance_df)} rows")
    else:
        print("Binance reference data not found — cross-exchange premium feature disabled")

    return df, btc_df, sol_df, binance_df


def compute_sample_weights(feat_df: pd.DataFrame) -> np.ndarray:
    """Exponential time-decay sample weights so recent candles matter more.

    With SAMPLE_DECAY_HALFLIFE_DAYS=30, a candle 90 days ago has
    (0.5)^3 ≈ 12.5% the weight of the most recent candle.
    This makes models adapt faster to current market microstructure
    without discarding the stability that older data provides.
    """
    if "date" not in feat_df.columns:
        return np.ones(len(feat_df))

    dates = pd.to_datetime(feat_df["date"], utc=True, errors="coerce")
    latest = dates.max()
    days_ago = (latest - dates).dt.total_seconds() / 86400.0
    days_ago = days_ago.fillna(days_ago.median()).clip(lower=0)
    decay_rate = np.log(2) / SAMPLE_DECAY_HALFLIFE_DAYS
    weights = np.exp(-decay_rate * days_ago.values)
    # Normalise so mean weight = 1 (keeps effective sample size interpretation clean)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def prepare_dataset(df, btc_df, sol_df=None, binance_df=None):
    """Base feature engineering + labels, before regime enrichment."""
    print("Computing features...")
    execution_df = load_or_build_execution_features(candle_dates=df["date"], prefer_cached=True)
    orderbook_df = load_or_build_orderbook_features(candle_dates=df["date"], prefer_cached=True)
    feat_df = compute_features(
        df, btc_df, sol_df=sol_df,
        execution_df=execution_df, orderbook_df=orderbook_df,
        binance_df=binance_df,
    )

    print("Computing labels...")
    feat_df = compute_labels(feat_df, horizon=HORIZON, add_multi_horizon=True)

    base_feature_cols = get_feature_columns(feat_df)
    print(f"Base features before regime enrichment: {len(base_feature_cols)}")

    # Drop rows with NaN in features or primary labels
    required = base_feature_cols + ["direction", "future_volatility", "direction_1", "direction_3"]
    feat_df = feat_df.dropna(subset=required).reset_index(drop=True)
    print(f"Usable rows after NaN removal: {len(feat_df)}")

    return feat_df


def split_data(feat_df, feature_cols):
    """Time-based train/val/test split with multi-horizon labels."""
    n = len(feat_df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = feat_df.iloc[:train_end]
    val = feat_df.iloc[train_end:val_end]
    test = feat_df.iloc[val_end:]

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    X_train = train[feature_cols].values
    X_val = val[feature_cols].values
    X_test = test[feature_cols].values

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        # Primary 30-min direction
        "y_dir_train": train["direction"].values,
        "y_dir_val": val["direction"].values,
        "y_dir_test": test["direction"].values,
        # 3-class direction at 30-min
        "y_dir3_train": np.where(train["future_return"].values > DIR3_FLAT_BAND, 2,
                                    np.where(train["future_return"].values < -DIR3_FLAT_BAND, 0, 1)),
        "y_dir3_val": np.where(val["future_return"].values > DIR3_FLAT_BAND, 2,
                                  np.where(val["future_return"].values < -DIR3_FLAT_BAND, 0, 1)),
        "y_dir3_test": np.where(test["future_return"].values > DIR3_FLAT_BAND, 2,
                                   np.where(test["future_return"].values < -DIR3_FLAT_BAND, 0, 1)),
        # 5-min direction (adverse selection / fill-time horizon)
        "y_dir1_train": train["direction_1"].values,
        "y_dir1_val": val["direction_1"].values,
        "y_dir1_test": test["direction_1"].values,
        # 15-min direction
        "y_dir3m_train": train["direction_3"].values,
        "y_dir3m_val": val["direction_3"].values,
        "y_dir3m_test": test["direction_3"].values,
        # Volatility
        "y_vol_train": train["future_volatility"].values,
        "y_vol_val": val["future_volatility"].values,
        "y_vol_test": test["future_volatility"].values,
        "test_df": test,
    }


def train_direction_model(data, feature_cols, sample_weights=None):
    """Train XGBoost direction classifier with the selected runtime backend."""
    print("\n" + "=" * 60)
    print("TRAINING DIRECTION MODEL (up/down classifier, 30-min horizon)")
    print("=" * 60)

    params = RESOLVED_XGB_DIR_PARAMS.copy()
    n_est = params.pop("n_estimators")
    early = params.pop("early_stopping_rounds")

    model = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early, **params)
    model.fit(
        data["X_train"], data["y_dir_train"],
        eval_set=[(data["X_val"], data["y_dir_val"])],
        sample_weight=sample_weights,
        verbose=100,
    )

    # Evaluate on test set
    y_pred_proba = model.predict_proba(data["X_test"])[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(data["y_dir_test"], y_pred)
    auc = roc_auc_score(data["y_dir_test"], y_pred_proba)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test AUC-ROC:  {auc:.4f}")
    print(classification_report(data["y_dir_test"], y_pred, target_names=["Down", "Up"]))

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    print("Top 15 features (direction):")
    print(imp.nlargest(15).to_string())

    # ── Isotonic calibration on the val set ─────────────────────────
    # Fit isotonic regression on XGBoost val-set scores, then wrap it in a
    # thin class that exposes predict_proba() so the rest of the code is unchanged.
    # (sklearn >= 1.6 removed cv="prefit"; manual calibration is equivalent.)
    print("\nCalibrating probabilities (manual isotonic, val set)...")
    calibrator = None
    auc_cal = auc
    try:
        raw_val = model.predict_proba(data["X_val"])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_val, data["y_dir_val"])
        calibrator = _IsoCalibrator(model, iso)

        y_pred_cal = calibrator.predict_proba(data["X_test"])[:, 1]
        auc_cal = roc_auc_score(data["y_dir_test"], y_pred_cal)
        print(f"Calibrated AUC-ROC: {auc_cal:.4f} (was {auc:.4f})")
    except Exception as e:
        print(f"Calibration skipped: {e}")

    return model, calibrator, {"accuracy": acc, "auc": auc, "auc_calibrated": auc_cal}


def train_volatility_model(data, feature_cols, sample_weights=None):
    """Train volatility regressor with the selected runtime backend."""
    print("\n" + "=" * 60)
    print("TRAINING VOLATILITY MODEL (future price range)")
    print("=" * 60)

    if LGB_AVAILABLE:
        assert lgb is not None
        model = lgb.LGBMRegressor(**RESOLVED_LGB_VOL_PARAMS)
        model.fit(
            data["X_train"], data["y_vol_train"],
            eval_set=[(data["X_val"], data["y_vol_val"])],
            sample_weight=sample_weights,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        model_type = "lightgbm"
    else:
        print(f"LightGBM unavailable ({LGB_IMPORT_ERROR}). Falling back to XGBoost volatility model.")
        params = RESOLVED_XGB_VOL_FALLBACK_PARAMS.copy()
        n_est = params.pop("n_estimators")
        early = params.pop("early_stopping_rounds")
        model = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=early, **params)
        model.fit(
            data["X_train"], data["y_vol_train"],
            eval_set=[(data["X_val"], data["y_vol_val"])],
            verbose=100,
        )
        model_type = "xgboost"

    y_pred = np.asarray(model.predict(data["X_test"]), dtype=float)
    mae = mean_absolute_error(data["y_vol_test"], y_pred)
    r2 = r2_score(data["y_vol_test"], y_pred)

    print(f"\nTest MAE:  {mae:.6f}")
    print(f"Test R²:   {r2:.4f}")

    imp = pd.Series(model.feature_importances_, index=feature_cols)
    print("Top 15 features (volatility):")
    print(imp.nlargest(15).to_string())

    return model, {"mae": mae, "r2": r2, "model_type": model_type}


def train_direction3_model(data, feature_cols):
    """Train 3-class direction model: Down(0) / Flat(1) / Up(2)."""
    print("\n" + "=" * 60)
    print("TRAINING 3-CLASS DIRECTION MODEL (down/flat/up)")
    print("=" * 60)

    params = RESOLVED_XGB_DIR_PARAMS.copy()
    params.update({
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    })
    # scale_pos_weight is binary-only; remove to avoid XGBoost warning
    params.pop("scale_pos_weight", None)
    n_est = params.pop("n_estimators")
    early = params.pop("early_stopping_rounds")

    model = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early, **params)
    model.fit(
        data["X_train"], data["y_dir3_train"],
        eval_set=[(data["X_val"], data["y_dir3_val"])],
        verbose=100,
    )

    pred = model.predict(data["X_test"])
    proba = model.predict_proba(data["X_test"])
    acc3 = accuracy_score(data["y_dir3_test"], pred)
    up_auc = roc_auc_score((data["y_dir3_test"] == 2).astype(int), proba[:, 2])

    print(f"\n3-Class Accuracy: {acc3:.4f}")
    print(f"Up-vs-Rest AUC:    {up_auc:.4f}")
    print("Class distribution (test): "
          f"down={(data['y_dir3_test'] == 0).mean():.3f}, "
          f"flat={(data['y_dir3_test'] == 1).mean():.3f}, "
          f"up={(data['y_dir3_test'] == 2).mean():.3f}")

    return model, {"accuracy_3class": acc3, "up_auc": up_auc, "flat_band": DIR3_FLAT_BAND}


def train_direction_model_fast(data, feature_cols, sample_weights=None):
    """Train 5-min direction classifier (adverse selection / fill-time horizon).

    At 1-candle horizon the label is noisier, so we use a shallower tree
    (max_depth=4) and stronger regularisation to avoid overfitting noise.
    Even a 52-53% edge at 5-min is highly valuable for market-making spread skew.
    """
    print("\n" + "=" * 60)
    print("TRAINING FAST DIRECTION MODEL (up/down classifier, 5-min horizon)")
    print("=" * 60)

    params = RESOLVED_XGB_DIR_PARAMS.copy()
    params.update({
        "max_depth": 4,
        "min_child_weight": 30,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "n_estimators": 1200,
    })
    n_est = params.pop("n_estimators")
    early = params.pop("early_stopping_rounds")

    model = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early, **params)
    model.fit(
        data["X_train"], data["y_dir1_train"],
        eval_set=[(data["X_val"], data["y_dir1_val"])],
        sample_weight=sample_weights,
        verbose=200,
    )

    y_pred_proba = model.predict_proba(data["X_test"])[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(data["y_dir1_test"], y_pred)
    auc = roc_auc_score(data["y_dir1_test"], y_pred_proba)

    print(f"\nFast Model — Test Accuracy: {acc:.4f}  AUC: {auc:.4f}")
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    print("Top 10 features (fast direction):")
    print(imp.nlargest(10).to_string())

    return model, {"accuracy": acc, "auc": auc, "horizon_candles": 1}


def train_direction_model_mid(data, feature_cols, sample_weights=None):
    """Train 15-min direction classifier (medium-context horizon).

    Complements the 5-min (adverse selection) and 30-min (trend) models.
    Uses same params as 30-min model but trained on 3-candle labels.
    """
    print("\n" + "=" * 60)
    print("TRAINING MID DIRECTION MODEL (up/down classifier, 15-min horizon)")
    print("=" * 60)

    params = RESOLVED_XGB_DIR_PARAMS.copy()
    n_est = params.pop("n_estimators")
    early = params.pop("early_stopping_rounds")

    model = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early, **params)
    model.fit(
        data["X_train"], data["y_dir3m_train"],
        eval_set=[(data["X_val"], data["y_dir3m_val"])],
        sample_weight=sample_weights,
        verbose=200,
    )

    y_pred_proba = model.predict_proba(data["X_test"])[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(data["y_dir3m_test"], y_pred)
    auc = roc_auc_score(data["y_dir3m_test"], y_pred_proba)

    print(f"\nMid Model — Test Accuracy: {acc:.4f}  AUC: {auc:.4f}")
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    print("Top 10 features (mid direction):")
    print(imp.nlargest(10).to_string())

    return model, {"accuracy": acc, "auc": auc, "horizon_candles": 3}


def walk_forward_validation(feat_df, feature_cols, n_splits=5):
    """Walk-forward cross-validation to check for temporal stability."""
    print("\n" + "=" * 60)
    print(f"WALK-FORWARD VALIDATION ({n_splits} folds)")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = feat_df[feature_cols].values
    y_dir = feat_df["direction"].values
    y_vol = feat_df["future_volatility"].values

    dir_scores, vol_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_dir_tr, y_dir_te = y_dir[train_idx], y_dir[test_idx]
        y_vol_tr, y_vol_te = y_vol[train_idx], y_vol[test_idx]

        # Direction
        params = RESOLVED_XGB_DIR_PARAMS.copy()
        params["n_estimators"] = 300
        params["verbosity"] = 0
        params.pop("early_stopping_rounds", None)
        model_d = xgb.XGBClassifier(**params)
        model_d.fit(X_tr, y_dir_tr)
        proba = model_d.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_dir_te, proba)
        acc = accuracy_score(y_dir_te, (proba > 0.5).astype(int))
        dir_scores.append({"fold": fold, "auc": auc, "accuracy": acc})

        # Volatility
        if LGB_AVAILABLE:
            assert lgb is not None
            params_v = RESOLVED_LGB_VOL_PARAMS.copy()
            params_v["n_estimators"] = 300
            model_v = lgb.LGBMRegressor(**params_v)
            model_v.fit(X_tr, y_vol_tr)
        else:
            params_v = RESOLVED_XGB_VOL_FALLBACK_PARAMS.copy()
            params_v["n_estimators"] = 300
            params_v["verbosity"] = 0
            params_v.pop("early_stopping_rounds", None)
            model_v = xgb.XGBRegressor(**params_v)
            model_v.fit(X_tr, y_vol_tr)
        pred_v = np.asarray(model_v.predict(X_te), dtype=float)
        mae = mean_absolute_error(y_vol_te, pred_v)
        r2 = r2_score(y_vol_te, pred_v)
        vol_scores.append({"fold": fold, "mae": mae, "r2": r2})

        print(f"  Fold {fold}: dir_auc={auc:.4f} dir_acc={acc:.4f} vol_mae={mae:.6f} vol_r2={r2:.4f}")

    print(f"\n  Mean Direction AUC:  {np.mean([s['auc'] for s in dir_scores]):.4f} "
          f"± {np.std([s['auc'] for s in dir_scores]):.4f}")
    print(f"  Mean Direction Acc:  {np.mean([s['accuracy'] for s in dir_scores]):.4f}")
    print(f"  Mean Volatility MAE: {np.mean([s['mae'] for s in vol_scores]):.6f}")
    print(f"  Mean Volatility R²:  {np.mean([s['r2'] for s in vol_scores]):.4f}")

    return dir_scores, vol_scores


def run_purged_walk_forward_validation(base_feat_df: pd.DataFrame, n_splits: int = 6) -> dict:
    """Purged walk-forward validation with Sharpe-based fold diagnostics."""
    print("\n" + "=" * 60)
    print(f"PURGED WALK-FORWARD VALIDATION ({n_splits} folds)")
    print("=" * 60)

    from backtest import (
        MMConfig,
        _apply_backtest_mtf_gate,
        apply_regime_gate_env_override,
        simulate_ml_variant,
    )

    splits = generate_purged_walk_forward_splits(
        len(base_feat_df),
        n_splits=n_splits,
        purge_rows=HORIZON,
        embargo_rows=HORIZON,
    )
    if not splits:
        payload = {
            "status": "skipped",
            "reason": "insufficient_rows",
            "folds": [],
            "summary": summarize_walk_forward_results([]),
        }
        write_walk_forward_results(payload)
        return payload

    folds: list[dict] = []
    for split in splits:
        fold = int(split["fold"])
        train_end = int(split["train_end"])
        test_start = int(split["test_start"])
        test_end = int(split["test_end"])

        fold_source = base_feat_df.iloc[:test_end].copy().reset_index(drop=True)
        fold_regime_bundle, _ = fit_regime_model(fold_source, train_end=train_end)
        fold_df = apply_regime_model(fold_source, fold_regime_bundle)
        feature_cols = get_feature_columns(fold_df)

        train = fold_df.iloc[:train_end].copy()
        test = fold_df.iloc[test_start:test_end].copy().reset_index(drop=True)
        if train.empty or test.empty:
            continue

        X_tr = train[feature_cols].to_numpy(dtype=float)
        X_te = test[feature_cols].to_numpy(dtype=float)
        y_dir_tr = train["direction"].to_numpy(dtype=int)
        y_dir_te = test["direction"].to_numpy(dtype=int)
        y_vol_tr = train["future_volatility"].to_numpy(dtype=float)
        y_vol_te = test["future_volatility"].to_numpy(dtype=float)
        if len(np.unique(y_dir_tr)) < 2 or len(np.unique(y_dir_te)) < 2:
            continue

        train_weights = compute_sample_weights(train)
        dir_params = RESOLVED_XGB_DIR_PARAMS.copy()
        dir_params["n_estimators"] = 350
        dir_params["verbosity"] = 0
        dir_params.pop("early_stopping_rounds", None)
        dir_model = xgb.XGBClassifier(**dir_params)
        dir_model.fit(X_tr, y_dir_tr, sample_weight=train_weights)
        dir_proba = np.asarray(dir_model.predict_proba(X_te)[:, 1], dtype=float)
        dir_pred = (dir_proba > 0.5).astype(int)

        if LGB_AVAILABLE:
            assert lgb is not None
            vol_params = RESOLVED_LGB_VOL_PARAMS.copy()
            vol_params["n_estimators"] = 300
            vol_model = lgb.LGBMRegressor(**vol_params)
            vol_model.fit(X_tr, y_vol_tr, sample_weight=train_weights)
        else:
            vol_params = RESOLVED_XGB_VOL_FALLBACK_PARAMS.copy()
            vol_params["n_estimators"] = 300
            vol_params["verbosity"] = 0
            vol_params.pop("early_stopping_rounds", None)
            vol_model = xgb.XGBRegressor(**vol_params)
            vol_model.fit(X_tr, y_vol_tr)
        vol_pred = np.asarray(vol_model.predict(X_te), dtype=float)
        test["predicted_volatility"] = vol_pred

        gated_dir_proba, gate_stats = _apply_backtest_mtf_gate(dir_proba, test)
        wf_config = MMConfig()
        apply_regime_gate_env_override(wf_config, verbose=(fold == 0))
        sim_metrics, _, _, _ = simulate_ml_variant(
            test,
            gated_dir_proba,
            vol_pred,
            wf_config,
            label=f"WF Fold {fold}",
            use_inventory_target=False,
            use_multilevel=True,
            use_confidence_sizing=True,
            seed=42 + fold,
        )

        fold_result = {
            "fold": fold,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_start_ts": str(train["date"].iloc[0]),
            "train_end_ts": str(train["date"].iloc[-1]),
            "test_start_ts": str(test["date"].iloc[0]),
            "test_end_ts": str(test["date"].iloc[-1]),
            "auc": round(float(roc_auc_score(y_dir_te, dir_proba)), 4),
            "accuracy": round(float(accuracy_score(y_dir_te, dir_pred)), 4),
            "vol_mae": round(float(mean_absolute_error(y_vol_te, vol_pred)), 6),
            "vol_r2": round(float(r2_score(y_vol_te, vol_pred)), 4),
            "sharpe_ratio": float(sim_metrics["sharpe_ratio"]),
            "total_return_pct": float(sim_metrics["total_return_pct"]),
            "max_drawdown_pct": float(sim_metrics["max_drawdown_pct"]),
            "total_trades": int(sim_metrics["total_trades"]),
            "mtf_gate_fired": int(gate_stats.get("fired", 0)),
            "regime_breakdown": sim_metrics.get("regime_breakdown", {}),
        }
        folds.append(fold_result)
        print(
            f"  Fold {fold}: auc={fold_result['auc']:.4f} "
            f"acc={fold_result['accuracy']:.4f} sharpe={fold_result['sharpe_ratio']:.3f} "
            f"ret={fold_result['total_return_pct']:.3f}% trades={fold_result['total_trades']}"
        )

    summary = summarize_walk_forward_results(folds)
    payload = {
        "status": "ok" if folds else "skipped",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "purge_rows": int(HORIZON),
        "embargo_rows": int(HORIZON),
        "folds": folds,
        "summary": summary,
    }
    write_walk_forward_results(payload)

    if folds:
        print(f"\n  Mean Direction AUC:  {summary['mean_auc']:.4f}")
        print(f"  Mean Fold Sharpe:    {summary['mean_sharpe']:.4f}")
        print(f"  Positive last 6:     {summary['positive_sharpe_last_6']}/6")
    else:
        print("\n  Walk-forward skipped: no valid folds produced")

    return payload


def save_models(dir_model, dir3_model, calibrator, vol_model, feature_cols,
                dir_metrics, dir3_metrics, vol_metrics, execution_summary, orderbook_summary,
                nn_model=None, nn_metrics=None,
                dir_model_fast=None, dir_metrics_fast=None,
                dir_model_mid=None, dir_metrics_mid=None,
                meta_bundle=None, meta_metrics=None,
                regime_bundle=None, regime_metrics=None,
                walk_forward_payload=None):
    """Save trained models and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"ml_mm_{timestamp}"
    model_path.mkdir(parents=True, exist_ok=True)

    def _save_to(path):
        dir_model.save_model(str(path / "direction_model.json"))
        dir3_model.save_model(str(path / "direction3_model.json"))
        with open(path / "volatility_model.pkl", "wb") as f:
            pickle.dump(vol_model, f)
        if calibrator is not None:
            with open(path / "direction_calibrator.pkl", "wb") as f:
                pickle.dump(calibrator, f)
        # Multi-horizon models (optional — signal server falls back gracefully when absent)
        if dir_model_fast is not None:
            dir_model_fast.save_model(str(path / "direction_model_fast.json"))
        if dir_model_mid is not None:
            dir_model_mid.save_model(str(path / "direction_model_mid.json"))
        # Neural LSTM (optional)
        if nn_model is not None and nn_metrics:
            save_neural_model(nn_model, nn_metrics, path)
        if meta_bundle is not None and meta_metrics:
            save_meta_ensemble(meta_bundle, meta_metrics, path)
        if regime_bundle is not None:
            save_regime_model(regime_bundle, regime_metrics or {}, path)

    _save_to(model_path)

    latest_path = LATEST_MODEL_DIR
    # Remove stale files so XGBoost/signal-server don't hold locked handles on the old ones
    if latest_path.exists():
        shutil.rmtree(latest_path)
    latest_path.mkdir(parents=True, exist_ok=True)
    _save_to(latest_path)

    metadata = {
        "pair": PAIR,
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "direction_metrics": dir_metrics,
        "direction3_metrics": dir3_metrics,
        "direction_fast_metrics": dir_metrics_fast or {},
        "direction_mid_metrics": dir_metrics_mid or {},
        "volatility_metrics": vol_metrics,
        "trained_at": timestamp,
        "primary_exchange": str(DATA_DIR.name),
        "reference_exchange": str(BINANCE_DATA_DIR.name),
        "training_backends": summarize_selected_backends(TRAINING_BACKEND_SELECTION),
        "training_backend_selection": TRAINING_BACKEND_SELECTION,
        "xgb_dir_params": {k: str(v) for k, v in RESOLVED_XGB_DIR_PARAMS.items()},
        "vol_model_family": "lightgbm" if LGB_AVAILABLE else "xgboost",
        "lgb_vol_params": {k: str(v) for k, v in RESOLVED_LGB_VOL_PARAMS.items()} if LGB_AVAILABLE else {},
        "xgb_vol_fallback_params": {k: str(v) for k, v in RESOLVED_XGB_VOL_FALLBACK_PARAMS.items()} if not LGB_AVAILABLE else {},
        "execution_learning": execution_summary,
        "orderbook_learning": orderbook_summary,
        "neural_lstm": nn_metrics or {},
        "meta_ensemble": meta_metrics or {},
        "regime_model": regime_metrics or {},
        "walk_forward_validation": (walk_forward_payload or {}).get("summary", {}),
    }

    for p in [model_path, latest_path]:
        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nModels saved to: {model_path}")
    print(f"Latest symlink:  {latest_path}")
    return model_path


def main():
    print("=" * 60)
    print("ML Market-Making Model Training Pipeline")
    print("=" * 60)

    # 1. Load data (primary + cross-asset + binance reference)
    print(f"Primary exchange data dir: {DATA_DIR}")
    print(f"Reference exchange data dir: {BINANCE_DATA_DIR}")
    print(f"Training backend selection: {summarize_selected_backends(TRAINING_BACKEND_SELECTION)}")
    df, btc_df, sol_df, binance_df = load_data()
    execution_summary = export_execution_features(candle_dates=df["date"])
    execution_summary["feature_count"] = len(execution_feature_columns())
    print(f"Execution-learning snapshot: orders={execution_summary['orders_total']} fills={execution_summary['fills_total']}")
    orderbook_summary = export_orderbook_features(candle_dates=df["date"])
    print(f"Orderbook snapshot: snapshots={orderbook_summary['snapshots_total']} features={orderbook_summary['feature_count']}")

    # 2. Feature engineering (includes binance premium + multi-horizon labels)
    base_feat_df = prepare_dataset(df, btc_df, sol_df, binance_df=binance_df)

    # 3. Purged walk-forward validation on the base feature frame
    walk_forward_payload = run_purged_walk_forward_validation(base_feat_df)

    # 4. Fit the unsupervised regime model on the final-train slice, then enrich
    # the full frame so final model training and live inference share the schema.
    regime_train_end = int(len(base_feat_df) * TRAIN_RATIO)
    regime_bundle, regime_metrics = fit_regime_model(base_feat_df, train_end=regime_train_end)
    feat_df = apply_regime_model(base_feat_df, regime_bundle)
    feature_cols = get_feature_columns(feat_df)
    print(f"Total features after Tier 2 enrichment: {len(feature_cols)}")

    # 5. Compute exponential time-decay sample weights (recent candles weighted more)
    sample_weights = compute_sample_weights(feat_df)
    print(f"Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, "
          f"mean={sample_weights.mean():.3f} (effective N ~{int(len(feat_df) * sample_weights.mean())})")

    # 6. Split for final training
    data = split_data(feat_df, feature_cols)
    # Restrict weights to train split only (val/test untouched)
    sw_train = sample_weights[:len(data["X_train"])]

    # 7. Train models
    dir_model, calibrator, dir_metrics = train_direction_model(data, feature_cols, sample_weights=sw_train)
    dir3_model, dir3_metrics = train_direction3_model(data, feature_cols)
    vol_model, vol_metrics = train_volatility_model(data, feature_cols, sample_weights=sw_train)
    nn_model, nn_metrics = train_neural_model(feat_df, feature_cols)

    # 8. Multi-horizon models (5-min and 15-min direction)
    dir_model_fast, dir_metrics_fast = train_direction_model_fast(data, feature_cols, sample_weights=sw_train)
    dir_model_mid, dir_metrics_mid = train_direction_model_mid(data, feature_cols, sample_weights=sw_train)

    # 9. Regime-aware meta-ensemble stacker
    meta_bundle, meta_metrics = train_meta_ensemble(
        feat_df,
        data,
        feature_cols,
        dir_model,
        dir3_model,
        vol_model,
        calibrator=calibrator,
        nn_model=nn_model,
        nn_meta=nn_metrics,
        dir_model_fast=dir_model_fast,
        dir_model_mid=dir_model_mid,
    )
    if meta_metrics:
        state = "enabled" if meta_metrics.get("enabled") else "skipped"
        print(f"Meta-ensemble: {state} :: {meta_metrics}")

    # 10. Save all models
    save_models(
        dir_model, dir3_model, calibrator, vol_model, feature_cols,
        dir_metrics, dir3_metrics, vol_metrics, execution_summary, orderbook_summary,
        nn_model=nn_model, nn_metrics=nn_metrics,
        dir_model_fast=dir_model_fast, dir_metrics_fast=dir_metrics_fast,
        dir_model_mid=dir_model_mid, dir_metrics_mid=dir_metrics_mid,
        meta_bundle=meta_bundle, meta_metrics=meta_metrics,
        regime_bundle=regime_bundle, regime_metrics=regime_metrics,
        walk_forward_payload=walk_forward_payload,
    )

    print("\nTraining complete!")
    return dir_model, vol_model, data, feature_cols


if __name__ == "__main__":
    main()
