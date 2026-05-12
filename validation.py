from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from shared.paths import WALK_FORWARD_RESULTS_FILE


def generate_purged_walk_forward_splits(
    n_rows: int,
    *,
    n_splits: int = 6,
    min_train_rows: int = 900,
    purge_rows: int = 6,
    embargo_rows: int = 6,
    min_test_rows: int = 120,
) -> list[dict]:
    if n_rows <= 0:
        return []

    gap_rows = max(0, int(purge_rows)) + max(0, int(embargo_rows))
    min_train_rows = min(max(300, int(min_train_rows)), max(1, n_rows - gap_rows - 1))
    available_rows = n_rows - min_train_rows - gap_rows
    if available_rows < max(1, min_test_rows):
        return []

    max_splits = max(1, available_rows // max(1, min_test_rows))
    n_splits = max(1, min(int(n_splits), max_splits))
    test_size = max(int(min_test_rows), available_rows // n_splits)

    splits: list[dict] = []
    for fold in range(n_splits):
        test_start = min_train_rows + gap_rows + fold * test_size
        test_end = n_rows if fold == (n_splits - 1) else min(n_rows, test_start + test_size)
        train_end = max(0, test_start - gap_rows)
        if train_end < 300 or test_end <= test_start:
            continue
        splits.append(
            {
                "fold": len(splits),
                "train_start": 0,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "purge_rows": int(purge_rows),
                "embargo_rows": int(embargo_rows),
                "gap_rows": int(gap_rows),
            }
        )
    return splits


def summarize_walk_forward_results(folds: Iterable[dict]) -> dict:
    folds = list(folds)
    if not folds:
        return {
            "enabled": False,
            "detail": "no_folds",
            "fold_count": 0,
            "positive_sharpe_folds": 0,
            "positive_sharpe_last_6": 0,
            "passes_promotion_gate": False,
        }

    sharpe_values = np.asarray([float(f.get("sharpe_ratio", 0.0) or 0.0) for f in folds], dtype=float)
    return_values = np.asarray([float(f.get("total_return_pct", 0.0) or 0.0) for f in folds], dtype=float)
    auc_values = np.asarray([float(f.get("auc", 0.0) or 0.0) for f in folds], dtype=float)
    last6 = sharpe_values[-6:]
    positive_last6 = int(np.sum(last6 > 0.0))

    return {
        "enabled": True,
        "fold_count": int(len(folds)),
        "positive_sharpe_folds": int(np.sum(sharpe_values > 0.0)),
        "positive_sharpe_last_6": positive_last6,
        "passes_promotion_gate": bool(positive_last6 >= 5),
        "mean_sharpe": round(float(np.mean(sharpe_values)), 4),
        "median_sharpe": round(float(np.median(sharpe_values)), 4),
        "mean_return_pct": round(float(np.mean(return_values)), 4),
        "mean_auc": round(float(np.mean(auc_values)), 4),
    }


def write_walk_forward_results(payload: dict, path: Path | None = None) -> None:
    target = path or WALK_FORWARD_RESULTS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
