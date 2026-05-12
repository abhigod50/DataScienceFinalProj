from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import LATEST_MODEL_DIR
from meta_ensemble import (
    LIVE_POST_CALIBRATOR_FILE,
    LIVE_POST_CALIBRATOR_META_FILE,
    compute_core_model_signature,
)
from shared.paths import PREDICTION_CALIBRATION_EVENTS_FILE


DEFAULT_LOOKBACK_HOURS = int(os.getenv("ML_LIVE_CALIBRATION_LOOKBACK_HOURS", "168"))
DEFAULT_MIN_ROWS = int(os.getenv("ML_LIVE_CALIBRATION_MIN_ROWS", "300"))
DEFAULT_MAX_ROWS = int(os.getenv("ML_LIVE_CALIBRATION_MAX_ROWS", "4000"))
DEFAULT_MIN_HOLDOUT_ROWS = int(os.getenv("ML_LIVE_CALIBRATION_MIN_HOLDOUT_ROWS", "80"))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _recent_calibration_events(
    *,
    events_path: Path,
    base_model_signature: str,
    lookback_hours: int,
    max_rows: int,
) -> pd.DataFrame:
    if not events_path.exists():
        return pd.DataFrame()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(int(lookback_hours), 1))
    rows: list[dict] = []
    with events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("model_signature", "")) != base_model_signature:
                continue
            if not bool(payload.get("meta_enabled", False)):
                continue

            resolved_ts = pd.to_datetime(payload.get("resolved_ts"), utc=True, errors="coerce")
            if pd.isna(resolved_ts):
                continue
            if resolved_ts.to_pydatetime() < cutoff:
                continue

            raw_prob = pd.to_numeric(payload.get("pre_calibration_probability"), errors="coerce")
            actual_up = pd.to_numeric(payload.get("actual_up"), errors="coerce")
            if pd.isna(raw_prob) or pd.isna(actual_up):
                continue
            post_cal_prob = pd.to_numeric(payload.get("post_calibration_probability"), errors="coerce")
            actual_return = pd.to_numeric(payload.get("actual_return"), errors="coerce")

            rows.append(
                {
                    "prediction_ts": str(payload.get("prediction_ts", "")),
                    "resolved_ts": resolved_ts.to_pydatetime(),
                    "pair": str(payload.get("pair", "")),
                    "pre_calibration_probability": float(np.clip(raw_prob, 0.0, 1.0)),
                    "post_calibration_probability": (
                        float(np.clip(post_cal_prob, 0.0, 1.0)) if pd.notna(post_cal_prob) else None
                    ),
                    "actual_up": int(float(actual_up) > 0.5),
                    "actual_return": 0.0 if pd.isna(actual_return) else float(actual_return),
                    "ensemble_mode": str(payload.get("ensemble_mode", "")),
                    "conformal_enabled": bool(payload.get("conformal_enabled", False)),
                    "mtf_gate_applied": bool(payload.get("mtf_gate_applied", False)),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("resolved_ts").reset_index(drop=True)
    if max_rows > 0 and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
    return df


def refresh_live_post_calibrator(
    model_dir: Optional[Path] = None,
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_holdout_rows: int = DEFAULT_MIN_HOLDOUT_ROWS,
) -> dict:
    model_dir = model_dir or LATEST_MODEL_DIR
    base_model_signature = compute_core_model_signature(model_dir)
    df = _recent_calibration_events(
        events_path=PREDICTION_CALIBRATION_EVENTS_FILE,
        base_model_signature=base_model_signature,
        lookback_hours=lookback_hours,
        max_rows=max_rows,
    )

    summary = {
        "status": "skipped",
        "reason": "",
        "base_model_signature": base_model_signature,
        "rows": int(len(df)),
        "lookback_hours": int(lookback_hours),
        "events_path": str(PREDICTION_CALIBRATION_EVENTS_FILE),
    }
    if df.empty:
        summary["reason"] = "no_recent_events_for_current_model"
        return summary
    if len(df) < max(int(min_rows), 50):
        summary["reason"] = "insufficient_rows"
        return summary

    y_full = df["actual_up"].to_numpy(dtype=int)
    if len(np.unique(y_full)) < 2:
        summary["reason"] = "single_class_window"
        return summary

    holdout_rows = max(int(min_holdout_rows), int(round(len(df) * 0.20)))
    holdout_rows = min(holdout_rows, max(1, len(df) // 2))
    train_end = len(df) - holdout_rows
    if train_end < max(120, int(min_rows * 0.6)):
        summary["reason"] = "insufficient_train_rows_after_holdout"
        return summary

    train_df = df.iloc[:train_end].copy()
    holdout_df = df.iloc[train_end:].copy()
    y_train = train_df["actual_up"].to_numpy(dtype=int)
    y_holdout = holdout_df["actual_up"].to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_holdout)) < 2:
        summary["reason"] = "train_or_holdout_single_class"
        return summary

    x_train = train_df["pre_calibration_probability"].to_numpy(dtype=float)
    x_holdout = holdout_df["pre_calibration_probability"].to_numpy(dtype=float)

    candidate = IsotonicRegression(out_of_bounds="clip")
    candidate.fit(x_train, y_train)
    holdout_calibrated = np.clip(candidate.predict(x_holdout), 0.0, 1.0)

    pre_brier = float(brier_score_loss(y_holdout, x_holdout))
    post_brier = float(brier_score_loss(y_holdout, holdout_calibrated))
    pre_auc = _safe_auc(y_holdout, x_holdout)
    post_auc = _safe_auc(y_holdout, holdout_calibrated)
    brier_improvement = pre_brier - post_brier

    summary.update(
        {
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "holdout_brier_pre": round(pre_brier, 6),
            "holdout_brier_post": round(post_brier, 6),
            "holdout_brier_delta": round(brier_improvement, 6),
            "holdout_auc_pre": None if pre_auc is None else round(pre_auc, 6),
            "holdout_auc_post": None if post_auc is None else round(post_auc, 6),
        }
    )
    if post_brier > pre_brier + 1e-6:
        summary["reason"] = "holdout_brier_regressed"
        return summary

    final_calibrator = IsotonicRegression(out_of_bounds="clip")
    x_full = df["pre_calibration_probability"].to_numpy(dtype=float)
    final_calibrator.fit(x_full, y_full)
    full_calibrated = np.clip(final_calibrator.predict(x_full), 0.0, 1.0)
    full_auc_pre = _safe_auc(y_full, x_full)
    full_auc_post = _safe_auc(y_full, full_calibrated)

    calibrator_path = model_dir / LIVE_POST_CALIBRATOR_FILE
    metadata_path = model_dir / LIVE_POST_CALIBRATOR_META_FILE
    model_dir.mkdir(parents=True, exist_ok=True)
    with calibrator_path.open("wb") as handle:
        pickle.dump(final_calibrator, handle)

    pair_counts = {
        str(pair): int(count)
        for pair, count in df["pair"].fillna("unknown").value_counts().sort_index().items()
    }
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "updated",
        "source": "live_incremental_isotonic",
        "base_model_signature": base_model_signature,
        "lookback_hours": int(lookback_hours),
        "events_path": str(PREDICTION_CALIBRATION_EVENTS_FILE),
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "pair_counts": pair_counts,
        "resolved_ts_start": train_df["resolved_ts"].iloc[0].isoformat() if not train_df.empty else None,
        "resolved_ts_end": df["resolved_ts"].iloc[-1].isoformat(),
        "holdout_brier_pre": round(pre_brier, 6),
        "holdout_brier_post": round(post_brier, 6),
        "holdout_brier_delta": round(brier_improvement, 6),
        "holdout_auc_pre": None if pre_auc is None else round(pre_auc, 6),
        "holdout_auc_post": None if post_auc is None else round(post_auc, 6),
        "full_brier_pre": round(float(brier_score_loss(y_full, x_full)), 6),
        "full_brier_post": round(float(brier_score_loss(y_full, full_calibrated)), 6),
        "full_auc_pre": None if full_auc_pre is None else round(full_auc_pre, 6),
        "full_auc_post": None if full_auc_post is None else round(full_auc_post, 6),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary.update(metadata)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the live post-ensemble isotonic calibrator.")
    parser.add_argument("--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--min-holdout-rows", type=int, default=DEFAULT_MIN_HOLDOUT_ROWS)
    args = parser.parse_args()

    result = refresh_live_post_calibrator(
        lookback_hours=args.lookback_hours,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
