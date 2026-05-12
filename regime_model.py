from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

from shared.paths import LATEST_MODEL_DIR


REGIME_MODEL_FILE = "market_regime_model.pkl"
REGIME_META_FILE = "market_regime_meta.json"
REGIME_MODEL_VERSION = 1
REGIME_CLUSTER_COUNT = 4
REGIME_MIN_ROWS = 300

REGIME_INPUT_COLUMNS = [
    "realized_vol_24",
    "vol_ratio_12_72",
    "return_zscore",
    "rolling_sharpe_24",
    "variance_ratio_4",
]


def regime_feature_columns(n_clusters: int = REGIME_CLUSTER_COUNT) -> list[str]:
    cols = [
        "market_regime_id",
        "market_regime_distance",
        "market_regime_confidence",
        "market_regime_vol_rank",
        "market_regime_stress",
        "market_regime_trend_score",
        "market_regime_transition",
    ]
    cols.extend(f"market_regime_is_{i}" for i in range(max(1, int(n_clusters))))
    return cols


def _neutral_frame(index: pd.Index, n_clusters: int = REGIME_CLUSTER_COUNT) -> pd.DataFrame:
    cols = regime_feature_columns(n_clusters)
    frame = pd.DataFrame(0.0, index=index, columns=cols, dtype=float)
    frame["market_regime_confidence"] = 0.5
    if "market_regime_is_0" in frame.columns:
        frame["market_regime_is_0"] = 1.0
    return frame


def _sanitize_inputs(
    feat_df: pd.DataFrame,
    input_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    input_columns = input_columns or list(REGIME_INPUT_COLUMNS)
    clean = pd.DataFrame(index=feat_df.index)
    for col in input_columns:
        if col in feat_df.columns:
            values = pd.to_numeric(feat_df[col], errors="coerce")
        else:
            values = pd.Series(0.0, index=feat_df.index)
        clean[col] = values

    clean = clean.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    for col in clean.columns:
        series = clean[col]
        lo = float(series.quantile(0.005))
        hi = float(series.quantile(0.995))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            clean[col] = series.clip(lo, hi)
    return clean


def fit_regime_model(
    feat_df: pd.DataFrame,
    *,
    train_end: Optional[int] = None,
    n_clusters: int = REGIME_CLUSTER_COUNT,
) -> tuple[Optional[dict], dict]:
    train_end = len(feat_df) if train_end is None else int(np.clip(train_end, 0, len(feat_df)))
    if train_end < REGIME_MIN_ROWS:
        return None, {
            "enabled": False,
            "detail": f"insufficient_rows train_end={train_end}",
            "train_rows": train_end,
        }

    input_frame = _sanitize_inputs(feat_df, list(REGIME_INPUT_COLUMNS))
    fit_frame = input_frame.iloc[:train_end].copy()
    if fit_frame.empty:
        return None, {
            "enabled": False,
            "detail": "empty_fit_frame",
            "train_rows": 0,
        }

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    X_fit = scaler.fit_transform(fit_frame.to_numpy(dtype=float))
    unique_rows = int(np.unique(np.round(X_fit, 6), axis=0).shape[0])
    n_clusters = min(max(2, int(n_clusters)), unique_rows)
    if n_clusters < 2:
        return None, {
            "enabled": False,
            "detail": "not_enough_unique_rows",
            "train_rows": len(fit_frame),
        }

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(1024, max(128, len(fit_frame))),
        n_init=20,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(X_fit)
    fit_ref = feat_df.iloc[:train_end].copy()

    cluster_stats: dict[int, dict[str, float]] = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_slice = fit_ref.iloc[mask]
        if cluster_slice.empty:
            cluster_stats[cluster_id] = {
                "rows": 0,
                "mean_realized_vol_24": 0.0,
                "mean_return_zscore": 0.0,
                "mean_rolling_sharpe_24": 0.0,
                "mean_variance_ratio_4": 1.0,
            }
            continue
        cluster_stats[cluster_id] = {
            "rows": int(mask.sum()),
            "mean_realized_vol_24": float(pd.to_numeric(cluster_slice.get("realized_vol_24"), errors="coerce").fillna(0.0).mean()),
            "mean_return_zscore": float(pd.to_numeric(cluster_slice.get("return_zscore"), errors="coerce").fillna(0.0).mean()),
            "mean_rolling_sharpe_24": float(pd.to_numeric(cluster_slice.get("rolling_sharpe_24"), errors="coerce").fillna(0.0).mean()),
            "mean_variance_ratio_4": float(pd.to_numeric(cluster_slice.get("variance_ratio_4"), errors="coerce").fillna(1.0).mean()),
        }

    vol_order = sorted(
        range(n_clusters),
        key=lambda cluster_id: cluster_stats[cluster_id]["mean_realized_vol_24"],
    )
    vol_rank = {cluster_id: rank for rank, cluster_id in enumerate(vol_order)}
    trend_score = {
        cluster_id: float(np.clip(
            cluster_stats[cluster_id]["mean_rolling_sharpe_24"]
            + 0.50 * (cluster_stats[cluster_id]["mean_variance_ratio_4"] - 1.0)
            + 0.20 * cluster_stats[cluster_id]["mean_return_zscore"],
            -3.0,
            3.0,
        ))
        for cluster_id in range(n_clusters)
    }

    metrics = {
        "enabled": True,
        "version": REGIME_MODEL_VERSION,
        "train_rows": int(train_end),
        "n_clusters": int(n_clusters),
        "input_columns": list(REGIME_INPUT_COLUMNS),
        "cluster_stats": cluster_stats,
        "cluster_vol_rank": vol_rank,
        "cluster_trend_score": trend_score,
    }
    bundle = {
        "model": model,
        "scaler": scaler,
        "version": REGIME_MODEL_VERSION,
        "input_columns": list(REGIME_INPUT_COLUMNS),
        "n_clusters": int(n_clusters),
        "cluster_vol_rank": vol_rank,
        "cluster_trend_score": trend_score,
    }
    return bundle, metrics


def apply_regime_model(
    feat_df: pd.DataFrame,
    bundle: Optional[dict],
) -> pd.DataFrame:
    if feat_df is None or feat_df.empty:
        return feat_df

    n_clusters = int(bundle.get("n_clusters", REGIME_CLUSTER_COUNT)) if bundle else REGIME_CLUSTER_COUNT
    out = feat_df.copy()
    existing = [col for col in out.columns if str(col).startswith("market_regime_")]
    if existing:
        out = out.drop(columns=existing)
    neutral = _neutral_frame(out.index, n_clusters=n_clusters)

    if not bundle or bundle.get("model") is None or bundle.get("scaler") is None:
        return out.join(neutral)

    try:
        input_columns = list(bundle.get("input_columns") or REGIME_INPUT_COLUMNS)
        clean = _sanitize_inputs(out, input_columns=input_columns)
        X = bundle["scaler"].transform(clean.to_numpy(dtype=float))
        labels = np.asarray(bundle["model"].predict(X), dtype=int)
        distances = np.asarray(bundle["model"].transform(X), dtype=float).min(axis=1)
    except Exception:
        return out.join(neutral)

    regime_df = _neutral_frame(out.index, n_clusters=n_clusters)
    regime_df["market_regime_id"] = labels.astype(float)
    regime_df["market_regime_distance"] = distances
    regime_df["market_regime_confidence"] = 1.0 / (1.0 + np.maximum(distances, 0.0))
    vol_rank_map = bundle.get("cluster_vol_rank", {})
    trend_score_map = bundle.get("cluster_trend_score", {})
    regime_df["market_regime_vol_rank"] = [float(vol_rank_map.get(int(label), 0)) for label in labels]
    denom = max(1, n_clusters - 1)
    regime_df["market_regime_stress"] = regime_df["market_regime_vol_rank"] / float(denom)
    regime_df["market_regime_trend_score"] = [float(trend_score_map.get(int(label), 0.0)) for label in labels]
    transitions = np.zeros(len(labels), dtype=float)
    if len(labels) > 1:
        transitions[1:] = (labels[1:] != labels[:-1]).astype(float)
    regime_df["market_regime_transition"] = transitions

    for cluster_id in range(n_clusters):
        regime_df[f"market_regime_is_{cluster_id}"] = (labels == cluster_id).astype(float)

    return out.join(regime_df)


def save_regime_model(bundle: Optional[dict], metrics: dict, model_dir: Path) -> None:
    if bundle is None:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / REGIME_MODEL_FILE, "wb") as handle:
        pickle.dump(bundle, handle)
    with open(model_dir / REGIME_META_FILE, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_regime_model(model_dir: Optional[Path] = None) -> Optional[dict]:
    model_dir = model_dir or LATEST_MODEL_DIR
    path = model_dir / REGIME_MODEL_FILE
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            bundle = pickle.load(handle)
        meta_path = model_dir / REGIME_META_FILE
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as handle:
                bundle["metadata"] = json.load(handle)
        return bundle
    except Exception as exc:
        print(f"[regime_model] load failed: {exc}")
        return None
