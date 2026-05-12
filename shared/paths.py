from __future__ import annotations

import os
from pathlib import Path

from .hb_config import normalize_exchange_id, read_hummingbot_exchange


# This standalone submission project is intentionally self-contained.
# All paths resolve inside this folder so the research pipeline can run without
# the original freqtrade workspace or any live Hummingbot services.
ML_DIR = Path(__file__).resolve().parent.parent
USER_DATA_DIR = ML_DIR
PROJECT_ROOT = ML_DIR
DEV_UI_DIR = ML_DIR / "dev_ui"
DEV_UI_STATIC_DIR = DEV_UI_DIR / "static"
CONFIG_DIR = ML_DIR / "config"
ARTIFACT_DIR = ML_DIR / "artifacts"

HB_WORKSPACE = ML_DIR
HB_FILES_DIR = ML_DIR
HB_CONF_DIR = CONFIG_DIR
HB_CONF_SCRIPTS_DIR = CONFIG_DIR
HB_SCRIPT_CONFIG_FILE = CONFIG_DIR / "ml_mm.yml"
HB_CONF_CLIENT_FILE = CONFIG_DIR / "conf_client.yml"
HB_FEE_OVERRIDES_FILE = CONFIG_DIR / "conf_fee_overrides.yml"
HB_SCRIPT_FILE = ARTIFACT_DIR / "ml_market_maker.py"
HB_SIGNAL_FILE = ARTIFACT_DIR / "ml_signal.json"
HB_EXECUTION_DB_FILE = ML_DIR / "data" / "execution" / "ml_mm.sqlite"
HB_DOCKER_COMPOSE_FILE = ML_DIR / "docker-compose.yml"
HB_ENV_FILE = ML_DIR / ".env"

LOG_DIR = ML_DIR / "logs"
DEV_UI_PROCESS_DIR = LOG_DIR / "dev_ui_processes"
RETRAIN_LOG_FILE = LOG_DIR / "retrain.log"
RETRAIN_DAEMON_LOG_FILE = LOG_DIR / "retrain_daemon.log"
RETRAIN_DAEMON_HISTORY_FILE = LOG_DIR / "retrain_daemon_history.jsonl"
TRAINING_STATUS_FILE = LOG_DIR / "ml_training_status.json"
RUNTIME_MONITOR_FILE = LOG_DIR / "runtime_monitor.jsonl"
PORTFOLIO_HISTORY_FILE = LOG_DIR / "portfolio_nav_history.jsonl"
NAV_HISTORY_1M_FILE = LOG_DIR / "portfolio_nav_history_1m.jsonl"
NAV_HISTORY_1H_FILE = LOG_DIR / "portfolio_nav_history_1h.jsonl"
SESSION_STATE_FILE = LOG_DIR / "session_state.json"
PREDICTION_QUALITY_HISTORY_FILE = LOG_DIR / "prediction_quality_history.jsonl"
PREDICTION_CALIBRATION_EVENTS_FILE = LOG_DIR / "prediction_calibration_events.jsonl"
RETRAIN_VALIDATION_HISTORY_FILE = LOG_DIR / "retrain_validation_history.jsonl"
COMPLIANCE_LOG_FILE = LOG_DIR / "compliance_events.jsonl"
SIGNAL_SUPERVISOR_STATE_FILE = LOG_DIR / "signal_supervisor_state.json"
SIGNAL_RUNTIME_EVENTS_FILE = LOG_DIR / "signal_runtime_events.jsonl"
SIGNAL_WATCHDOG_PID_FILE = LOG_DIR / "signal_watchdog.pid"
SIGNAL_WATCHDOG_LOG_FILE = LOG_DIR / "signal_watchdog.log"
SIGNAL_PID_FILE = LOG_DIR / "signal_server.pid"
SIGNAL_LOCK_FILE = LOG_DIR / "signal_server.lock"
SIGNAL_LOG_FILE = LOG_DIR / "signal_server.log"
DASHBOARD_PID_FILE = LOG_DIR / "dev_dashboard.pid"
DASHBOARD_LOCK_FILE = LOG_DIR / "dev_dashboard.lock"
RETRAIN_LOCK_FILE = LOG_DIR / "retrain.lock"
RETRAIN_DAEMON_PID_FILE = LOG_DIR / "retrain_daemon.pid"
RETRAIN_DAEMON_LOCK_FILE = LOG_DIR / "retrain_daemon.lock"
FORCE_PROMOTE_FLAG_FILE = LOG_DIR / "force_promote.flag"
PERFORMANCE_LOG_FILE = LOG_DIR / "performance_history.jsonl"
OPTIMIZER_STATE_FILE = LOG_DIR / "optimizer_state.json"
ORCHESTRATOR_LOG_FILE = LOG_DIR / "orchestrator.log"

MODEL_DIR = ML_DIR / "models"
LATEST_MODEL_DIR = MODEL_DIR / "latest"
BACKTEST_RESULTS_FILE = ML_DIR / "backtest_results.json"
WALK_FORWARD_RESULTS_FILE = ML_DIR / "walk_forward_results.json"
MODEL_METADATA_FILE = LATEST_MODEL_DIR / "metadata.json"
EXECUTION_DATA_DIR = ML_DIR / "data" / "execution"
EXECUTION_FEATURES_PATH = EXECUTION_DATA_DIR / "execution_features.feather"
EXECUTION_SUMMARY_PATH = EXECUTION_DATA_DIR / "execution_summary.json"
SIGNAL_WATCHDOG_SCRIPT = ML_DIR / "signal_runtime_watchdog.py"
PREFLIGHT_PAPER_TRADING_SCRIPT = ML_DIR / "scripts" / "preflight_paper_trading.ps1"

_LOCAL_VENV = ML_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
    "python.exe" if os.name == "nt" else "python"
)
_PARENT_VENV = ML_DIR.parent / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
    "python.exe" if os.name == "nt" else "python"
)
VENV_PYTHON = _LOCAL_VENV if _LOCAL_VENV.exists() else _PARENT_VENV


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip().lower()
    return value or default


def detect_primary_exchange_from_hummingbot() -> str:
    execution_only = {"kraken", "coinbaseadvanced", "coinbase", "kucoin", "gateio", "ftx"}
    exchange = read_hummingbot_exchange(HB_SCRIPT_CONFIG_FILE)
    if exchange in execution_only:
        return ""
    return exchange


def resolve_primary_exchange_id(default: str = "binanceus") -> str:
    explicit = normalize_exchange_id(os.getenv("ML_PRIMARY_EXCHANGE", ""))
    if explicit:
        return explicit

    signal_exchange = normalize_exchange_id(os.getenv("ML_SIGNAL_EXCHANGE", ""))
    if signal_exchange:
        return signal_exchange

    inferred = detect_primary_exchange_from_hummingbot()
    if inferred:
        return inferred

    return default


PRIMARY_EXCHANGE_ID = resolve_primary_exchange_id("binanceus")
REFERENCE_EXCHANGE_ID = normalize_exchange_id(_env_or_default("ML_REFERENCE_EXCHANGE", "coinbase")) or "coinbase"

DATA_ROOT = ML_DIR / "data"


def data_dir_for_exchange(exchange_id: str) -> Path:
    normalized = normalize_exchange_id(exchange_id)
    return DATA_ROOT / (normalized or PRIMARY_EXCHANGE_ID)


def dev_ui_process_pid_file(name: str) -> Path:
    return DEV_UI_PROCESS_DIR / f"{name}.pid"


def dev_ui_process_log_file(name: str) -> Path:
    return DEV_UI_PROCESS_DIR / f"{name}.log"


DATA_DIR = data_dir_for_exchange(PRIMARY_EXCHANGE_ID)
REFERENCE_DATA_DIR = data_dir_for_exchange(REFERENCE_EXCHANGE_ID)
ORDERBOOK_DATA_DIR = DATA_DIR / "orderbook"
ORDERBOOK_SNAPSHOT_PATH = ORDERBOOK_DATA_DIR / "orderbook_snapshots.feather"
ORDERBOOK_FEATURES_PATH = ORDERBOOK_DATA_DIR / "orderbook_features.feather"
ORDERBOOK_SUMMARY_PATH = ORDERBOOK_DATA_DIR / "orderbook_summary.json"
