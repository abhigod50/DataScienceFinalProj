from __future__ import annotations

import json
import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from config import (
    HORIZON_FAST_WEIGHT,
    HORIZON_MID_WEIGHT,
    HORIZON_SLOW_WEIGHT,
    LATEST_MODEL_DIR,
    NN_ENSEMBLE_WEIGHT,
)
from neural_model import inference_neural_components_series


META_MODEL_FILE = "direction_meta_ensemble.pkl"
META_META_FILE = "direction_meta_meta.json"
POST_CALIBRATOR_FILE = "direction_post_calibrator.pkl"
LIVE_POST_CALIBRATOR_FILE = "direction_post_calibrator_live.pkl"
LIVE_POST_CALIBRATOR_META_FILE = "direction_post_calibrator_live_meta.json"
META_MODEL_VERSION = 1
META_MIN_TRAIN_ROWS = 200

META_BASE_FEATURE_COLUMNS = [
    "prob_binary",
    "prob_calibrated",
    "prob_slow",
    "slow_p_down",
    "slow_p_flat",
    "slow_p_up",
    "prob_fast",
    "prob_mid",
    "prob_tabular",
    "prob_neural",
    "prob_neural_fast",
    "prob_neural_mid",
    "prob_static_blend",
    "vol_pred_base",
    "vol_pred_neural",
    "vol_pred_blend",
    "leg_mean",
    "leg_std",
    "leg_range",
    "bullish_vote_share",
    "bearish_vote_share",
    "tabular_margin",
    "fast_slow_gap",
    "mid_slow_gap",
    "fast_mid_gap",
    "neural_tabular_gap",
    "neural_fast_mid_gap",
    "slow_calibrated_gap",
    "slow_binary_gap",
    "neural_available",
]

META_CONTEXT_CANDIDATES = [
    "realized_vol_24",
    "vol_ratio_12_72",
    "range_1h_pct",
    "volume_ratio",
    "close_vs_vwap",
    "return_zscore",
    "mtf_15m_ret_3",
    "mtf_1h_ret_3",
    "mtf_1h_realized_vol_12",
    "exec_fill_rate_1h",
    "exec_cancel_rate_1h",
    "exec_fill_imbalance_1h",
    "exec_buy_fill_share_1h",
    "exec_avg_order_lifetime_sec_1h",
    "exec_has_history",
    "ob_quoted_spread_bps",
    "ob_book_pressure",
    "ob_depth_imbalance_5",
    "ob_depth_imbalance_20",
    "ob_weighted_mid_offset_bps",
    "ob_spread_mean_15m",
    "ob_pressure_mean_15m",
    "ob_has_data",
    "coinbase_premium_pct",
    "coinbase_premium_zscore",
    "market_regime_id",
    "market_regime_stress",
    "market_regime_trend_score",
    "market_regime_transition",
]

META_MODEL_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 220,
    "max_depth": 3,
    "min_samples_leaf": 40,
    "l2_regularization": 0.15,
    "early_stopping": False,
    "random_state": 42,
}
CONFORMAL_ALPHA = 0.10


def compute_core_model_signature(model_dir: Path) -> str:
    """Return a compact signature for the deployed prediction stack.

    The live post-calibration override is only valid for the exact model stack
    that emitted the underlying raw probabilities. We key it off core artefact
    mtimes/sizes so a retrain promotion automatically invalidates an older live
    calibrator without needing destructive cleanup.
    """
    artefacts = [
        "metadata.json",
        "direction_model.json",
        "direction3_model.json",
        "direction_model_fast.json",
        "direction_model_mid.json",
        "direction_calibrator.pkl",
        "volatility_model.pkl",
        "volatility_model.json",
        META_MODEL_FILE,
        META_META_FILE,
        POST_CALIBRATOR_FILE,
        "neural_model.pt",
        "neural_model_sklearn.pkl",
        "neural_model_meta.json",
        "market_regime_model.pkl",
        "market_regime_meta.json",
    ]
    parts: list[str] = []
    for name in artefacts:
        path = model_dir / name
        if not path.exists():
            parts.append(f"{name}:missing")
            continue
        stat = path.stat()
        parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def prepare_live_feature_frame(feat_df: pd.DataFrame, feature_cols: list[str]) -> tuple[Optional[pd.DataFrame], int, list[str]]:
    """Forward-fill feature columns for latest-row inference without mutating callers."""
    working = feat_df.copy()
    latest = working[feature_cols].replace([np.inf, -np.inf], np.nan).iloc[[-1]]
    nan_cols = latest.columns[latest.isnull().iloc[0]].tolist()
    if nan_cols:
        cleaned = working[feature_cols].replace([np.inf, -np.inf], np.nan).ffill()
        working.loc[:, feature_cols] = cleaned
    latest_after = working[feature_cols].replace([np.inf, -np.inf], np.nan).iloc[[-1]]
    remaining = latest_after.columns[latest_after.isnull().iloc[0]].tolist()
    if remaining:
        return None, len(nan_cols), remaining
    return working, len(nan_cols), []


def _safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _build_leg_frame(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    dir_model,
    dir3_model,
    vol_model,
    calibrator=None,
    nn_model=None,
    nn_meta: Optional[dict] = None,
    dir_model_fast=None,
    dir_model_mid=None,
) -> tuple[pd.DataFrame, str]:
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    prob_binary = _safe_predict_proba(dir_model, X)
    prob_calibrated = _safe_predict_proba(calibrator, X) if calibrator is not None else prob_binary.copy()

    slow_source = "binary"
    slow_p_down = np.zeros(len(X), dtype=float)
    slow_p_flat = np.zeros(len(X), dtype=float)
    slow_p_up = np.zeros(len(X), dtype=float)
    if dir3_model is not None:
        slow_raw = np.asarray(dir3_model.predict_proba(X), dtype=float)
        slow_p_down = slow_raw[:, 0]
        slow_p_flat = slow_raw[:, 1]
        slow_p_up = slow_raw[:, 2]
        prob_slow = np.clip(0.5 + 0.5 * (slow_p_up - slow_p_down), 0.0, 1.0)
        slow_source = "dir3"
    elif calibrator is not None:
        prob_slow = prob_calibrated.copy()
        slow_source = "calibrator"
    else:
        prob_slow = prob_binary.copy()

    prob_fast = _safe_predict_proba(dir_model_fast, X) if dir_model_fast is not None else prob_slow.copy()
    prob_mid = _safe_predict_proba(dir_model_mid, X) if dir_model_mid is not None else prob_slow.copy()
    prob_tabular = (
        (HORIZON_FAST_WEIGHT * prob_fast)
        + (HORIZON_MID_WEIGHT * prob_mid)
        + (HORIZON_SLOW_WEIGHT * prob_slow)
    )

    vol_pred_base = np.asarray(vol_model.predict(X), dtype=float)

    neural_components = inference_neural_components_series(nn_model, nn_meta or {}, feat_df, feature_cols)
    neural_fast_raw = np.asarray(neural_components["fast"], dtype=float)
    neural_mid_raw = np.asarray(neural_components["mid"], dtype=float)
    neural_dir_raw = np.asarray(neural_components["blend"], dtype=float)
    neural_vol_raw = np.asarray(neural_components["vol"], dtype=float)
    neural_available = np.isfinite(neural_dir_raw)
    prob_neural_fast = np.where(np.isfinite(neural_fast_raw), neural_fast_raw, prob_fast)
    prob_neural_mid = np.where(np.isfinite(neural_mid_raw), neural_mid_raw, prob_mid)
    prob_neural = np.where(neural_available, neural_dir_raw, prob_tabular)
    vol_pred_neural = np.where(np.isfinite(neural_vol_raw), neural_vol_raw, vol_pred_base)
    prob_static_blend = np.where(
        neural_available,
        ((1.0 - NN_ENSEMBLE_WEIGHT) * prob_tabular) + (NN_ENSEMBLE_WEIGHT * prob_neural),
        prob_tabular,
    )
    vol_pred_blend = np.where(
        np.isfinite(neural_vol_raw),
        (0.50 * vol_pred_base) + (0.50 * vol_pred_neural),
        vol_pred_base,
    )

    leg_matrix = np.column_stack([
        prob_fast,
        prob_mid,
        prob_slow,
        np.where(neural_available, prob_neural, np.nan),
    ])
    leg_mean = np.nanmean(leg_matrix, axis=1)
    leg_std = np.nanstd(leg_matrix, axis=1)
    leg_range = np.nanmax(leg_matrix, axis=1) - np.nanmin(leg_matrix, axis=1)
    bullish_vote_share = np.nanmean(leg_matrix > 0.5, axis=1)
    bearish_vote_share = np.nanmean(leg_matrix < 0.5, axis=1)

    leg_frame = pd.DataFrame(
        {
            "prob_binary": prob_binary,
            "prob_calibrated": prob_calibrated,
            "prob_slow": prob_slow,
            "slow_p_down": slow_p_down,
            "slow_p_flat": slow_p_flat,
            "slow_p_up": slow_p_up,
            "prob_fast": prob_fast,
            "prob_mid": prob_mid,
            "prob_tabular": prob_tabular,
            "prob_neural": prob_neural,
            "prob_neural_fast": prob_neural_fast,
            "prob_neural_mid": prob_neural_mid,
            "prob_neural_raw": neural_dir_raw,
            "prob_static_blend": prob_static_blend,
            "vol_pred_base": vol_pred_base,
            "vol_pred_neural": vol_pred_neural,
            "vol_pred_blend": vol_pred_blend,
            "leg_mean": leg_mean,
            "leg_std": leg_std,
            "leg_range": leg_range,
            "bullish_vote_share": bullish_vote_share,
            "bearish_vote_share": bearish_vote_share,
            "tabular_margin": prob_tabular - 0.5,
            "fast_slow_gap": prob_fast - prob_slow,
            "mid_slow_gap": prob_mid - prob_slow,
            "fast_mid_gap": prob_fast - prob_mid,
            "neural_tabular_gap": prob_neural - prob_tabular,
            "neural_fast_mid_gap": prob_neural_fast - prob_neural_mid,
            "slow_calibrated_gap": prob_slow - prob_calibrated,
            "slow_binary_gap": prob_slow - prob_binary,
            "neural_available": neural_available.astype(float),
        },
        index=feat_df.index,
    )
    return leg_frame, slow_source


def build_meta_feature_frame(
    leg_frame: pd.DataFrame,
    feat_df: pd.DataFrame,
    context_columns: Optional[list[str]] = None,
    force_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    context_columns = context_columns or [c for c in META_CONTEXT_CANDIDATES if c in feat_df.columns]

    meta_df = pd.DataFrame(index=leg_frame.index)
    for col in META_BASE_FEATURE_COLUMNS:
        if col in leg_frame.columns:
            meta_df[col] = np.asarray(leg_frame[col], dtype=float)
        else:
            meta_df[col] = 0.0

    for col in context_columns:
        if col in feat_df.columns:
            values = pd.to_numeric(feat_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            meta_df[col] = np.asarray(values, dtype=float)
        else:
            meta_df[col] = 0.0

    meta_df = meta_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if force_columns is not None:
        for col in force_columns:
            if col not in meta_df.columns:
                meta_df[col] = 0.0
        meta_df = meta_df[force_columns]
    return meta_df


def _apply_meta_probability_guards(
    meta_prob: np.ndarray,
    static_blend: np.ndarray,
    *,
    max_divergence: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip meta probabilities back toward the static blend when they drift."""
    meta_prob = np.asarray(meta_prob, dtype=float)
    static_blend = np.asarray(static_blend, dtype=float)
    max_div = float(np.clip(max_divergence, 0.0, 0.5))
    upper = np.clip(static_blend + max_div, 0.0, 1.0)
    lower = np.clip(static_blend - max_div, 0.0, 1.0)
    clipped = np.clip(meta_prob, lower, upper)

    static_confidence = np.abs(static_blend - 0.5)
    meta_flips_direction = ((meta_prob - 0.5) * (static_blend - 0.5)) < 0
    uncertainty_gate = (static_confidence > 0.15) & meta_flips_direction
    guarded = np.where(uncertainty_gate, static_blend, clipped)
    return guarded, uncertainty_gate


def _fit_conformal_band(probabilities: np.ndarray, targets: np.ndarray, alpha: float = CONFORMAL_ALPHA) -> Optional[dict]:
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(probabilities) < 200 or len(probabilities) != len(targets):
        return None
    scores = np.abs(targets - probabilities)
    if not np.isfinite(scores).all():
        return None
    q_level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    try:
        q_hat = float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        q_hat = float(np.quantile(scores, q_level, interpolation="higher"))
    return {
        "alpha": float(alpha),
        "q_hat": float(np.clip(q_hat, 0.0, 0.5)),
        "calibration_rows": int(len(scores)),
    }


def _apply_conformal_band(probabilities: np.ndarray, conformal: Optional[dict]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    if not conformal:
        return probabilities.copy(), probabilities.copy()
    q_hat = float(np.clip(conformal.get("q_hat", 0.0), 0.0, 0.5))
    lower = np.clip(probabilities - q_hat, 0.0, 1.0)
    upper = np.clip(probabilities + q_hat, 0.0, 1.0)
    return lower, upper


def train_meta_ensemble(
    feat_df: pd.DataFrame,
    data: dict,
    feature_cols: list[str],
    dir_model,
    dir3_model,
    vol_model,
    calibrator=None,
    nn_model=None,
    nn_meta: Optional[dict] = None,
    dir_model_fast=None,
    dir_model_mid=None,
) -> tuple[Optional[dict], dict]:
    """Train a regime-aware stacker on validation-period base-model outputs."""
    train_rows = len(data["X_train"])
    val_rows = len(data["X_val"])
    test_rows = len(data["X_test"])
    if val_rows < META_MIN_TRAIN_ROWS or test_rows == 0:
        return None, {
            "enabled": False,
            "detail": f"insufficient_rows val={val_rows} test={test_rows}",
        }

    leg_frame, slow_source = _build_leg_frame(
        feat_df,
        feature_cols,
        dir_model,
        dir3_model,
        vol_model,
        calibrator=calibrator,
        nn_model=nn_model,
        nn_meta=nn_meta,
        dir_model_fast=dir_model_fast,
        dir_model_mid=dir_model_mid,
    )
    context_columns = [c for c in META_CONTEXT_CANDIDATES if c in feat_df.columns]
    meta_df = build_meta_feature_frame(leg_frame, feat_df, context_columns=context_columns)

    val_start = train_rows
    test_start = train_rows + val_rows
    X_meta_train_full = meta_df.iloc[val_start:test_start]
    y_meta_train_full = np.asarray(data["y_dir_val"], dtype=int)
    X_meta_test = meta_df.iloc[test_start:test_start + test_rows]
    y_meta_test = np.asarray(data["y_dir_test"], dtype=int)

    if len(np.unique(y_meta_train_full)) < 2 or len(np.unique(y_meta_test)) < 2:
        return None, {
            "enabled": False,
            "detail": "meta_train_or_test_has_single_class",
        }

    cal_split = int(len(X_meta_train_full) * 0.70)
    if cal_split >= META_MIN_TRAIN_ROWS and (len(X_meta_train_full) - cal_split) >= 200:
        X_meta_train = X_meta_train_full.iloc[:cal_split]
        y_meta_train = y_meta_train_full[:cal_split]
        X_post_cal = X_meta_train_full.iloc[cal_split:]
        y_post_cal = y_meta_train_full[cal_split:]
        post_static = np.asarray(
            leg_frame.iloc[val_start + cal_split:test_start]["prob_static_blend"],
            dtype=float,
        )
    else:
        X_meta_train = X_meta_train_full
        y_meta_train = y_meta_train_full
        X_post_cal = None
        y_post_cal = None
        post_static = np.array([], dtype=float)

    model = HistGradientBoostingClassifier(**META_MODEL_PARAMS)
    model.fit(X_meta_train, y_meta_train)

    meta_prob = np.asarray(model.predict_proba(X_meta_test)[:, 1], dtype=float)
    static_prob = np.asarray(leg_frame.iloc[test_start:test_start + test_rows]["prob_static_blend"], dtype=float)
    guarded_prob, _ = _apply_meta_probability_guards(meta_prob, static_prob)

    post_calibrator = None
    calibration_rows = 0
    conformal = None
    if X_post_cal is not None and y_post_cal is not None and len(np.unique(y_post_cal)) >= 2:
        cal_meta_prob = np.asarray(model.predict_proba(X_post_cal)[:, 1], dtype=float)
        cal_guarded_prob, _ = _apply_meta_probability_guards(cal_meta_prob, post_static)
        try:
            post_calibrator = IsotonicRegression(out_of_bounds="clip")
            post_calibrator.fit(cal_guarded_prob, y_post_cal)
            calibration_rows = int(len(y_post_cal))
            cal_for_conformal = np.clip(post_calibrator.predict(cal_guarded_prob), 0.0, 1.0)
        except Exception:
            post_calibrator = None
            calibration_rows = 0
            cal_for_conformal = cal_guarded_prob
        conformal = _fit_conformal_band(cal_for_conformal, y_post_cal, alpha=CONFORMAL_ALPHA)

    eval_prob = guarded_prob
    if post_calibrator is not None:
        eval_prob = np.clip(post_calibrator.predict(guarded_prob), 0.0, 1.0)
    lower_prob, upper_prob = _apply_conformal_band(eval_prob, conformal)

    meta_pred = (eval_prob >= 0.5).astype(int)
    static_pred = (static_prob >= 0.5).astype(int)

    metrics = {
        "enabled": True,
        "backend": "hist_gradient_boosting",
        "version": META_MODEL_VERSION,
        "train_rows": int(len(X_meta_train)),
        "post_calibration_rows": calibration_rows,
        "test_rows": int(len(X_meta_test)),
        "feature_count": int(X_meta_train.shape[1]),
        "context_columns": context_columns,
        "slow_source": slow_source,
        "accuracy": round(float(accuracy_score(y_meta_test, meta_pred)), 4),
        "auc": round(float(roc_auc_score(y_meta_test, eval_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_meta_test, eval_prob)), 6),
        "baseline_static_accuracy": round(float(accuracy_score(y_meta_test, static_pred)), 4),
        "baseline_static_auc": round(float(roc_auc_score(y_meta_test, static_prob)), 4),
        "baseline_static_brier": round(float(brier_score_loss(y_meta_test, static_prob)), 6),
        "conformal_enabled": bool(conformal is not None),
        "conformal_alpha": float(conformal.get("alpha", CONFORMAL_ALPHA)) if conformal else float(CONFORMAL_ALPHA),
        "conformal_q_hat": round(float(conformal.get("q_hat", 0.0)), 6) if conformal else 0.0,
        "conformal_rows": int(conformal.get("calibration_rows", 0)) if conformal else 0,
        "conformal_width_mean": round(float(np.mean(upper_prob - lower_prob)), 6),
        "conformal_mid_coverage": round(float(np.mean((lower_prob <= 0.5) & (upper_prob >= 0.5))), 4),
    }
    metrics["delta_accuracy"] = round(metrics["accuracy"] - metrics["baseline_static_accuracy"], 4)
    metrics["delta_auc"] = round(metrics["auc"] - metrics["baseline_static_auc"], 4)
    metrics["delta_brier"] = round(metrics["baseline_static_brier"] - metrics["brier_score"], 6)

    bundle = {
        "model": model,
        "feature_columns": list(X_meta_train.columns),
        "context_columns": context_columns,
        "backend": "hist_gradient_boosting",
        "version": META_MODEL_VERSION,
        "slow_source": slow_source,
        "post_calibrator": post_calibrator,
        "conformal": conformal,
    }
    return bundle, metrics


def save_meta_ensemble(bundle: Optional[dict], metrics: dict, model_dir: Path) -> None:
    if bundle is None:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / META_MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    if bundle.get("post_calibrator") is not None:
        with open(model_dir / POST_CALIBRATOR_FILE, "wb") as f:
            pickle.dump(bundle["post_calibrator"], f)
    meta_payload = {
        "backend": bundle.get("backend", "unknown"),
        "version": bundle.get("version", META_MODEL_VERSION),
        "feature_columns": bundle.get("feature_columns", []),
        "context_columns": bundle.get("context_columns", []),
        "slow_source": bundle.get("slow_source", "unknown"),
        "post_calibrator": bool(bundle.get("post_calibrator") is not None),
        "conformal": bundle.get("conformal", {}),
        "metrics": metrics,
    }
    with open(model_dir / META_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)


def load_meta_ensemble(model_dir: Optional[Path] = None) -> Optional[dict]:
    model_dir = model_dir or LATEST_MODEL_DIR
    model_path = model_dir / META_MODEL_FILE
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        meta_path = model_dir / META_META_FILE
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                bundle["metadata"] = json.load(f)
        post_cal_path = model_dir / POST_CALIBRATOR_FILE
        if post_cal_path.exists() and bundle.get("post_calibrator") is None:
            with open(post_cal_path, "rb") as f:
                bundle["post_calibrator"] = pickle.load(f)
        bundle["post_calibrator_source"] = "bundle" if bundle.get("post_calibrator") is not None else "none"
        live_meta_path = model_dir / LIVE_POST_CALIBRATOR_META_FILE
        live_post_cal_path = model_dir / LIVE_POST_CALIBRATOR_FILE
        if live_meta_path.exists() and live_post_cal_path.exists():
            try:
                with open(live_meta_path, encoding="utf-8") as f:
                    live_meta = json.load(f)
                bundle["live_post_calibration_metadata"] = live_meta
                if live_meta.get("base_model_signature") == compute_core_model_signature(model_dir):
                    with open(live_post_cal_path, "rb") as f:
                        bundle["post_calibrator"] = pickle.load(f)
                    bundle["post_calibrator_source"] = "live_incremental"
            except Exception as e:
                print(f"[meta_ensemble] live post-calibrator override skipped: {e}")
        return bundle
    except Exception as e:
        print(f"[meta_ensemble] load failed: {e}")
        return None


def predict_direction_series(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    dir_model,
    dir3_model,
    vol_model,
    calibrator=None,
    nn_model=None,
    nn_meta: Optional[dict] = None,
    dir_model_fast=None,
    dir_model_mid=None,
    meta_bundle: Optional[dict] = None,
) -> pd.DataFrame:
    leg_frame, slow_source = _build_leg_frame(
        feat_df,
        feature_cols,
        dir_model,
        dir3_model,
        vol_model,
        calibrator=calibrator,
        nn_model=nn_model,
        nn_meta=nn_meta,
        dir_model_fast=dir_model_fast,
        dir_model_mid=dir_model_mid,
    )

    static_blend = np.asarray(leg_frame["prob_static_blend"], dtype=float)
    direction_probability = static_blend.copy()
    meta_enabled = False
    # Meta-ensemble can be disabled via environment variable. When live prediction
    # quality drops far below the training claim (observed 2026-04-10: train 72%
    # accuracy → live 37-46%), it is usually the HistGradientBoosting meta model
    # that has drifted because its context features (exec_*, ob_*, coinbase_premium_*)
    # are distributed differently in live than they were at train time. Disabling
    # it falls back to the static weighted blend of fast/mid/slow/neural, which
    # is more robust to context-feature drift.
    meta_disabled_by_env = str(os.environ.get("ML_DISABLE_META_ENSEMBLE", "")).lower() in {"1", "true", "yes"}
    # Feature quality gate: the meta model was trained on execution and orderbook
    # context features (exec_has_history, ob_has_data, coinbase_premium_pct, …).
    # When these are all 0.0 — which happens while execution history is warming up
    # or orderbook collection hasn't run — the meta model receives OOD input and
    # reliably drifts bearish (~0.27) because zero-context rows in training
    # coincided with specific downside conditions. Skip meta entirely when the
    # key context dimensions are all-zero across the recent tail.
    _exec_ctx = [c for c in ("exec_has_history", "exec_fill_rate_1h") if c in feat_df.columns]
    _ob_ctx = [c for c in ("ob_has_data", "ob_book_pressure") if c in feat_df.columns]
    _tail_n = min(5, len(feat_df))
    _exec_all_zero = (
        all(float(feat_df[c].iloc[-_tail_n:].fillna(0.0).max()) == 0.0 for c in _exec_ctx)
        if _exec_ctx else True
    )
    _ob_all_zero = (
        all(float(feat_df[c].iloc[-_tail_n:].fillna(0.0).max()) == 0.0 for c in _ob_ctx)
        if _ob_ctx else True
    )
    _context_ood = _exec_all_zero and _ob_all_zero

    if meta_disabled_by_env:
        direction_probability = static_blend
    elif _context_ood:
        direction_probability = static_blend
        print(
            f"[meta_ensemble] context quality gate: exec_has_history and ob_has_data "
            f"are all-zero across the last {_tail_n} rows — falling back to static blend "
            "to prevent OOD-driven bearish drift"
        )
    elif meta_bundle is not None and meta_bundle.get("model") is not None:
        try:
            meta_feature_columns = meta_bundle.get("feature_columns") or []
            context_columns = meta_bundle.get("context_columns") or []
            meta_df = build_meta_feature_frame(
                leg_frame,
                feat_df,
                context_columns=context_columns,
                force_columns=meta_feature_columns,
            )
            meta_out = np.asarray(meta_bundle["model"].predict_proba(meta_df)[:, 1], dtype=float)

            # Divergence guard: the meta_ensemble was trained to improve on the
            # static blend by a few percentage points of accuracy (typical
            # `delta_accuracy` ~0.03). If the meta output pulls the direction
            # probability more than MAX_META_DIVERGENCE away from the static
            # blend, that's a strong signal the meta model is drifting on
            # out-of-distribution context features rather than producing a
            # legitimately informed regime call. We clip it toward the static
            # blend so the meta can still skew the prediction within a sane
            # envelope without inverting the base models' consensus.
            try:
                _default_cap = float(os.environ.get("ML_META_DIVERGENCE_CAP", "0.20"))
            except (TypeError, ValueError):
                _default_cap = 0.20
            max_div = float(np.clip(_default_cap, 0.0, 0.5))
            upper = np.clip(static_blend + max_div, 0.0, 1.0)
            lower = np.clip(static_blend - max_div, 0.0, 1.0)
            clipped = np.clip(meta_out, lower, upper)

            # ── Uncertainty gate: suppress meta when static blend is confident
            # but meta disagrees. The meta model was trained on context features
            # (exec_*, ob_*, coinbase_premium_*) that are often OOD in live.
            # When the static blend already has a clear view (|prob - 0.5| > 0.15)
            # and meta wants to flip the direction, that's almost always meta drift
            # rather than a legitimate regime signal. Fall back to static blend.
            static_confidence = np.abs(static_blend - 0.5)
            meta_flips_direction = (
                ((meta_out - 0.5) * (static_blend - 0.5)) < 0
            )
            uncertainty_gate = (static_confidence > 0.15) & meta_flips_direction
            clipped = np.where(uncertainty_gate, static_blend, clipped)
            n_gate = int(np.sum(uncertainty_gate[-5:])) if len(clipped) >= 5 else 0
            if n_gate > 0:
                print(f"[meta_ensemble] uncertainty gate suppressed {n_gate}/5 recent predictions (meta opposed confident static blend)")

            # Log when the clip is engaging on the tail (most recent candles)
            # so divergence is visible in signal_server output without spamming
            # every cycle for every historical row.
            tail_slice = slice(-min(5, len(meta_out)), None)
            if len(meta_out) > 0:
                pulled = np.abs(meta_out[tail_slice] - clipped[tail_slice])
                if np.any(pulled > 1e-6):
                    diffs = meta_out[tail_slice] - static_blend[tail_slice]
                    print(
                        f"[meta_ensemble] divergence clip engaged "
                        f"(cap={max_div:.2f}, recent meta-static deltas={np.round(diffs, 3).tolist()})"
                    )
            direction_probability = clipped
            meta_enabled = True
        except Exception as e:
            print(f"[meta_ensemble] inference fallback to static blend: {e}")

    pre_post_calibration_probability = direction_probability.copy()
    post_calibrated = False
    if meta_enabled and meta_bundle is not None and meta_bundle.get("post_calibrator") is not None:
        try:
            calibrated = meta_bundle["post_calibrator"].predict(direction_probability)
            direction_probability = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
            post_calibrated = True
        except Exception as e:
            print(f"[meta_ensemble] post-calibration skipped: {e}")
    conformal = meta_bundle.get("conformal") if meta_bundle is not None else None
    lower_bound, upper_bound = _apply_conformal_band(direction_probability, conformal)
    conformal_enabled = bool(conformal is not None)

    uncertainty_score = np.clip(
        (0.55 * np.asarray(leg_frame["slow_p_flat"], dtype=float))
        + (1.50 * np.asarray(leg_frame["leg_std"], dtype=float))
        + (0.35 * np.abs(np.asarray(leg_frame["neural_tabular_gap"], dtype=float))),
        0.0,
        1.0,
    )

    pred_frame = leg_frame.copy()
    pred_frame["direction_probability"] = direction_probability
    pred_frame["direction_probability_pre_calibration"] = np.asarray(
        pre_post_calibration_probability,
        dtype=float,
    )
    pred_frame["predicted_volatility"] = np.asarray(leg_frame["vol_pred_blend"], dtype=float)
    pred_frame["confidence_raw"] = np.clip(np.abs(direction_probability - 0.5) * 2.0, 0.0, 1.0)
    pred_frame["uncertainty_score"] = uncertainty_score
    pred_frame["meta_enabled"] = float(meta_enabled)
    pred_frame["post_calibrated"] = float(post_calibrated)
    pred_frame["direction_lower_bound"] = np.asarray(lower_bound, dtype=float)
    pred_frame["direction_upper_bound"] = np.asarray(upper_bound, dtype=float)
    pred_frame["conformal_enabled"] = float(conformal_enabled)
    pred_frame["post_calibrator_source"] = (
        meta_bundle.get("post_calibrator_source", "none") if meta_bundle is not None else "none"
    )
    if meta_disabled_by_env:
        pred_frame["ensemble_mode"] = "static_blend_forced"
    elif _context_ood:
        pred_frame["ensemble_mode"] = "static_blend_context_ood"
    elif meta_enabled:
        pred_frame["ensemble_mode"] = "regime_meta_clipped"
    else:
        pred_frame["ensemble_mode"] = "static_blend"
    pred_frame["slow_source"] = slow_source
    return pred_frame


def predict_latest_direction(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    dir_model,
    dir3_model,
    vol_model,
    calibrator=None,
    nn_model=None,
    nn_meta: Optional[dict] = None,
    dir_model_fast=None,
    dir_model_mid=None,
    meta_bundle: Optional[dict] = None,
) -> Optional[dict]:
    prepared, ffilled_count, remaining = prepare_live_feature_frame(feat_df, feature_cols)
    if prepared is None:
        return {
            "error": "remaining_nans_after_ffill",
            "remaining_nan_columns": remaining,
            "nan_forward_filled_count": ffilled_count,
        }

    pred_frame = predict_direction_series(
        prepared,
        feature_cols,
        dir_model,
        dir3_model,
        vol_model,
        calibrator=calibrator,
        nn_model=nn_model,
        nn_meta=nn_meta,
        dir_model_fast=dir_model_fast,
        dir_model_mid=dir_model_mid,
        meta_bundle=meta_bundle,
    )
    row = pred_frame.iloc[-1].to_dict()
    return {
        "prediction": row,
        "nan_forward_filled_count": ffilled_count,
        "remaining_nan_columns": remaining,
    }
