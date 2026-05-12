"""Runtime backend probing for ML training and inference.

Selects the safest backend per library for the current machine instead of
assuming GPU is available and compatible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

_PROBE_CACHE: dict[bool, dict[str, Any]] = {}
_SELECTION_CACHE: dict[tuple[str, bool], dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_entry(name: str, requested: str) -> dict[str, Any]:
    return {
        "library": name,
        "requested": requested,
        "capability": "cpu",
        "selected": "cpu",
        "reason": "",
        "details": {},
    }


def _probe_torch(prefer_gpu: bool) -> dict[str, Any]:
    entry = _base_entry("torch", "gpu" if prefer_gpu else "cpu")
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        entry["reason"] = f"import_failed:{type(exc).__name__}"
        entry["details"] = {"error": str(exc)}
        return entry

    entry["details"] = {
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not prefer_gpu:
        entry["reason"] = "gpu_not_requested"
        return entry
    if not torch.cuda.is_available():
        entry["reason"] = "cuda_not_available"
        return entry

    try:
        capability = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
        arch_list = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
        sm_tag = f"sm_{capability[0]}{capability[1]}"
        entry["details"].update(
            {
                "device_name": device_name,
                "compute_capability": capability,
                "supported_arches": arch_list,
            }
        )
        if arch_list and sm_tag not in arch_list:
            entry["reason"] = f"unsupported_compute_capability_{sm_tag}"
            return entry

        dev = torch.device("cuda")
        probe = nn.LSTM(1, 1, batch_first=True).to(dev)
        x = torch.zeros((1, 1, 1), device=dev)
        with torch.no_grad():
            probe(x)
        entry["capability"] = "gpu"
        entry["reason"] = ""
        return entry
    except Exception as exc:
        entry["reason"] = f"gpu_probe_failed:{type(exc).__name__}"
        entry["details"]["error"] = str(exc)
        return entry


def _probe_xgboost(prefer_gpu: bool) -> dict[str, Any]:
    entry = _base_entry("xgboost", "gpu" if prefer_gpu else "cpu")
    try:
        import xgboost as xgb
    except Exception as exc:
        entry["reason"] = f"import_failed:{type(exc).__name__}"
        entry["details"] = {"error": str(exc)}
        return entry

    entry["details"] = {"version": getattr(xgb, "__version__", "unknown")}
    if not prefer_gpu:
        entry["reason"] = "gpu_not_requested"
        return entry

    try:
        X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.asarray([0, 0, 1, 1], dtype=float)
        model = xgb.XGBClassifier(
            device="cuda:0",
            tree_method="hist",
            n_estimators=2,
            max_depth=2,
            learning_rate=1.0,
            verbosity=0,
        )
        model.fit(X, y)
        model.predict_proba(X)
        entry["capability"] = "gpu"
        return entry
    except Exception as exc:
        entry["reason"] = f"gpu_probe_failed:{type(exc).__name__}"
        entry["details"]["error"] = str(exc)
        return entry


def _probe_lightgbm(prefer_gpu: bool) -> dict[str, Any]:
    entry = _base_entry("lightgbm", "gpu" if prefer_gpu else "cpu")
    try:
        import lightgbm as lgb
    except Exception as exc:
        entry["reason"] = f"import_failed:{type(exc).__name__}"
        entry["details"] = {"error": str(exc)}
        return entry

    entry["details"] = {"version": getattr(lgb, "__version__", "unknown")}
    if not prefer_gpu:
        entry["reason"] = "gpu_not_requested"
        return entry

    try:
        X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float)
        model = lgb.LGBMRegressor(
            device="gpu",
            n_estimators=3,
            max_depth=2,
            learning_rate=0.5,
            verbose=-1,
        )
        model.fit(X, y)
        model.predict(X)
        entry["capability"] = "gpu"
        return entry
    except Exception as exc:
        entry["reason"] = f"gpu_probe_failed:{type(exc).__name__}"
        entry["details"]["error"] = str(exc)
        return entry


def probe_ml_backends(prefer_gpu: bool = True, use_cache: bool = True) -> dict[str, Any]:
    if use_cache and prefer_gpu in _PROBE_CACHE:
        return _PROBE_CACHE[prefer_gpu]

    result = {
        "probed_at": _now_iso(),
        "prefer_gpu": bool(prefer_gpu),
        "libraries": {
            "torch": _probe_torch(prefer_gpu),
            "xgboost": _probe_xgboost(prefer_gpu),
            "lightgbm": _probe_lightgbm(prefer_gpu),
        },
    }
    _PROBE_CACHE[prefer_gpu] = result
    return result


def select_runtime_backends(
    phase: str,
    prefer_gpu: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    key = (phase, prefer_gpu)
    if use_cache and key in _SELECTION_CACHE:
        return _SELECTION_CACHE[key]

    probe = probe_ml_backends(prefer_gpu=prefer_gpu, use_cache=use_cache)
    selection = {
        "phase": phase,
        "selected_at": _now_iso(),
        "prefer_gpu": bool(prefer_gpu),
        "libraries": {},
    }

    for name, info in probe["libraries"].items():
        selected = "cpu"
        reason = str(info.get("reason", "") or "")
        if phase == "training" and info.get("capability") == "gpu":
            selected = "gpu"
        elif phase == "inference" and name == "torch" and info.get("capability") == "gpu":
            selected = "gpu"
        elif phase == "inference" and info.get("capability") == "gpu":
            selected = "cpu"
            reason = reason or "cpu_inference_preferred_for_single_row"
        elif not reason:
            reason = "cpu_selected"

        selection["libraries"][name] = {
            **info,
            "selected": selected,
            "reason": reason,
        }

    _SELECTION_CACHE[key] = selection
    return selection


def summarize_selected_backends(selection: dict[str, Any]) -> dict[str, str]:
    libraries = selection.get("libraries", {}) if isinstance(selection, dict) else {}
    return {
        name: str((info or {}).get("selected", "cpu"))
        for name, info in libraries.items()
    }


def build_training_backend_params(
    xgb_dir_params: dict[str, Any],
    lgb_vol_params: dict[str, Any],
    xgb_vol_fallback_params: dict[str, Any],
    prefer_gpu: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = select_runtime_backends("training", prefer_gpu=prefer_gpu)
    xgb_selected = selection["libraries"]["xgboost"]["selected"]
    lgb_selected = selection["libraries"]["lightgbm"]["selected"]

    resolved_xgb_dir = dict(xgb_dir_params)
    resolved_xgb_vol = dict(xgb_vol_fallback_params)
    resolved_lgb_vol = dict(lgb_vol_params)

    resolved_xgb_dir["device"] = "cuda:0" if xgb_selected == "gpu" else "cpu"
    resolved_xgb_vol["device"] = "cuda:0" if xgb_selected == "gpu" else "cpu"
    resolved_lgb_vol["device"] = "gpu" if lgb_selected == "gpu" else "cpu"

    return resolved_xgb_dir, resolved_lgb_vol, resolved_xgb_vol, selection


def apply_xgboost_runtime_backend(model: Any, selection: dict[str, Any]) -> None:
    if model is None:
        return
    libraries = selection.get("libraries", {}) if isinstance(selection, dict) else {}
    xgb_info = libraries.get("xgboost", {}) if isinstance(libraries, dict) else {}
    device = "cuda:0" if xgb_info.get("selected") == "gpu" else "cpu"
    try:
        if hasattr(model, "set_params"):
            model.set_params(device=device)
    except Exception:
        pass
    try:
        booster = model.get_booster()
        booster.set_param({"device": device})
    except Exception:
        pass
