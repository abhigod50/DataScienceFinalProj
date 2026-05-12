"""
Hummingbot ML setup doctor.
Runs fast checks for data/model/runtime readiness before launching live services.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from shared.paths import DATA_DIR, HB_SIGNAL_FILE, LATEST_MODEL_DIR, VENV_PYTHON


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Hummingbot ML environment")
    parser.add_argument("--quick", action="store_true", help="Skip full model inference smoke test")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ml_dir = root
    latest = LATEST_MODEL_DIR
    data_dir = DATA_DIR
    signal_path = HB_SIGNAL_FILE
    preferred_python = VENV_PYTHON

    failed = False

    # Python and package imports.
    pyv = sys.version_info
    if pyv < (3, 11):
        _fail(f"Python {pyv.major}.{pyv.minor} is too old. Need >= 3.11")
        failed = True
    else:
        _ok(f"Python {pyv.major}.{pyv.minor}.{pyv.micro}")

    if preferred_python.exists():
        if Path(sys.executable).resolve() == preferred_python.resolve():
            _ok(f"Running on managed venv interpreter: {preferred_python}")
        else:
            _warn(f"Running on {sys.executable} instead of managed venv {preferred_python}")

    try:
        import ccxt  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
        import xgboost  # noqa: F401
        _ok("Core packages import successfully")
    except Exception as exc:
        _fail(f"Core package import failed: {exc}")
        failed = True

    torch_available = False
    try:
        import torch  # noqa: F401
        torch_available = True
        _ok("PyTorch imports successfully")
    except Exception as exc:
        _warn(f"PyTorch unavailable: {exc}")

    try:
        import talib  # noqa: F401
        _ok("TA-Lib imports successfully")
    except Exception as exc:
        _warn(f"TA-Lib unavailable: {exc}")

    # Model artifacts.
    required_models = [
        latest / "direction_model.json",
        latest / "metadata.json",
    ]
    for p in required_models:
        if p.exists():
            _ok(f"Found model artifact: {p.name}")
        else:
            _fail(f"Missing model artifact: {p}")
            failed = True

    if (latest / "volatility_model.pkl").exists() or (latest / "volatility_model.json").exists():
        _ok("Found volatility model artifact")
    else:
        _fail("Missing volatility model (expected volatility_model.pkl or volatility_model.json)")
        failed = True

    # Data files.
    for pair in ["ETH_USDT-5m.feather", "BTC_USDT-5m.feather", "SOL_USDT-5m.feather"]:
        p = data_dir / pair
        if p.exists():
            _ok(f"Found data file: {pair}")
        else:
            _fail(f"Missing data file: {p}")
            failed = True

    # Metadata structure.
    metadata = {}
    meta_path = latest / "metadata.json"
    neural_meta_path = latest / "neural_model_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("feature_columns"):
                _ok(f"Metadata has {len(metadata['feature_columns'])} features")
            else:
                _fail("metadata.json missing feature_columns")
                failed = True
        except Exception as exc:
            _fail(f"Failed reading metadata.json: {exc}")
            failed = True

    if neural_meta_path.exists():
        try:
            with open(neural_meta_path, "r", encoding="utf-8") as f:
                neural_meta = json.load(f)
            backend = neural_meta.get("backend", "unknown")
            if backend == "pytorch":
                if torch_available:
                    _ok("Neural backend artifact is PyTorch LSTM")
                else:
                    _fail("Neural backend artifact is PyTorch, but PyTorch is not importable")
                    failed = True
            elif backend == "sklearn_mlp":
                _warn("Neural backend artifact is still sklearn_mlp fallback")
            else:
                _warn(f"Neural backend artifact is {backend}")
        except Exception as exc:
            _warn(f"Failed reading neural_model_meta.json: {exc}")

    # Signal directory writability.
    try:
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = signal_path.parent / ".doctor_write_test.tmp"
        tmp.write_text("ok", encoding="utf-8")
        tmp.unlink(missing_ok=True)
        _ok("Signal directory writable")
    except Exception as exc:
        _fail(f"Signal path not writable: {exc}")
        failed = True

    # Optional smoke inference.
    if not args.quick and not failed:
        try:
            import pickle
            import pyarrow.feather as pf
            import xgboost as xgb

            sys.path.insert(0, str(ml_dir))
            from execution_learning import load_or_build_execution_features  # noqa: E402
            from orderbook_features import load_or_build_orderbook_features  # noqa: E402
            from features import compute_features  # noqa: E402

            df = pf.read_feather(str(data_dir / "ETH_USDT-5m.feather"), memory_map=False).tail(400)
            btc = pf.read_feather(str(data_dir / "BTC_USDT-5m.feather"), memory_map=False).tail(400)
            execution_df = load_or_build_execution_features(candle_dates=df["date"], prefer_cached=True)
            orderbook_df = load_or_build_orderbook_features(candle_dates=df["date"], prefer_cached=True)
            feat = compute_features(df, btc, execution_df=execution_df, orderbook_df=orderbook_df)

            cols = metadata.get("feature_columns", [])
            latest_row = feat[cols].dropna().tail(1)
            if latest_row.empty:
                _fail("No non-NaN feature row available for inference smoke test")
                failed = True
            else:
                dir_model = xgb.XGBClassifier()
                dir_model.load_model(str(latest / "direction_model.json"))
                _ = dir_model.predict_proba(latest_row)

                if (latest / "volatility_model.pkl").exists():
                    with open(latest / "volatility_model.pkl", "rb") as f:
                        vol_model = pickle.load(f)
                    _ = vol_model.predict(latest_row)
                else:
                    vol_model = xgb.XGBRegressor()
                    vol_model.load_model(str(latest / "volatility_model.json"))
                    _ = vol_model.predict(latest_row)

                _ok("Inference smoke test passed")
        except Exception as exc:
            _fail(f"Inference smoke test failed: {exc}")
            failed = True
    elif args.quick:
        _warn("Quick mode: inference smoke test skipped")

    print("\nSetup doctor result: " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
