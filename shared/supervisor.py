"""Single-authority supervision helpers for the signal-server lifecycle.

The signal runtime watchdog (`signal_runtime_watchdog.py`) is the only owner of
the signal-server process. Every other component (the dashboard, the PowerShell
launcher, ad-hoc operators) is a *client* of that owner:

- Clients express intent by writing `signal_supervisor_state.json`
  (see `shared.contracts.update_supervisor_state`).
- Clients may *ensure the watchdog is running* via `ensure_watchdog_running`
  below, which is idempotent and does not itself supervise the signal server.

The watchdog command itself is constructed in exactly one place
(`build_watchdog_command`), so dashboard Python and launcher PowerShell stay
in lockstep.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contracts import append_runtime_event
from .io import read_pid, write_pid
from .paths import (
    HB_SIGNAL_FILE,
    LOG_DIR,
    ML_DIR,
    PROJECT_ROOT,
    SIGNAL_LOG_FILE,
    SIGNAL_PID_FILE,
    SIGNAL_RUNTIME_EVENTS_FILE,
    SIGNAL_SUPERVISOR_STATE_FILE,
    SIGNAL_WATCHDOG_LOG_FILE,
    SIGNAL_WATCHDOG_PID_FILE,
    SIGNAL_WATCHDOG_SCRIPT,
    VENV_PYTHON,
)


SIGNAL_SERVER_SCRIPT = ML_DIR / "signal_server.py"

DEFAULT_CHECK_INTERVAL_SECONDS = 20
DEFAULT_STALE_SECONDS = 240
DEFAULT_BOOT_GRACE_SECONDS = 300


def managed_python_executable() -> str:
    """Return the preferred interpreter path: project venv if present, else current."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _pid_alive(pid: int) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _find_watchdog_pids() -> list[int]:
    """Return PIDs whose cmdline mentions the watchdog script. Empty if psutil is absent."""
    try:
        import psutil  # type: ignore
    except Exception:
        return []

    matches: list[int] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = str(proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "signal_runtime_watchdog.py" in cmdline:
                matches.append(int(proc.info["pid"]))
        except Exception:
            continue
    return sorted(set(matches))


def build_watchdog_command(
    *,
    python_exe: str | None = None,
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
    boot_grace_seconds: int = DEFAULT_BOOT_GRACE_SECONDS,
) -> list[str]:
    """Canonical argv for launching the watchdog. Used by every client."""
    interpreter = python_exe or managed_python_executable()
    return [
        interpreter,
        str(SIGNAL_WATCHDOG_SCRIPT),
        "--signal-file", str(HB_SIGNAL_FILE),
        "--signal-pid-file", str(SIGNAL_PID_FILE),
        "--watchdog-pid-file", str(SIGNAL_WATCHDOG_PID_FILE),
        "--desired-state-file", str(SIGNAL_SUPERVISOR_STATE_FILE),
        "--python-exe", interpreter,
        "--signal-script", str(SIGNAL_SERVER_SCRIPT),
        "--signal-log-file", str(SIGNAL_LOG_FILE),
        "--working-dir", str(PROJECT_ROOT),
        "--watchdog-log-file", str(SIGNAL_WATCHDOG_LOG_FILE),
        "--check-interval", str(int(check_interval_seconds)),
        "--stale-seconds", str(int(stale_seconds)),
        "--boot-grace-seconds", str(int(boot_grace_seconds)),
    ]


def _emit(event: str, message: str, *, source: str, **fields: Any) -> None:
    try:
        append_runtime_event(
            SIGNAL_RUNTIME_EVENTS_FILE,
            source=source,
            event=event,
            message=message,
            **fields,
        )
    except Exception:
        pass


def ensure_watchdog_running(
    *,
    source: str = "shared",
    reason: str = "ensure_watchdog",
    signal_exchange: str | None = None,
) -> dict[str, Any]:
    """Idempotently ensure exactly one watchdog is running and its PID file is current.

    Returns a JSON-friendly dict: {ok, pid, message, adopted?}.
    """
    # 1. Existing watchdog tracked via PID file.
    tracked_pid = read_pid(SIGNAL_WATCHDOG_PID_FILE)
    if tracked_pid and _pid_alive(tracked_pid):
        return {
            "ok": True,
            "pid": int(tracked_pid),
            "adopted": False,
            "message": f"Signal watchdog already running (pid={tracked_pid}).",
        }

    # 2. External watchdog (e.g. started by a different client) — adopt it.
    external = _find_watchdog_pids()
    if external:
        adopted = int(external[-1])
        write_pid(SIGNAL_WATCHDOG_PID_FILE, adopted)
        return {
            "ok": True,
            "pid": adopted,
            "adopted": True,
            "message": f"Signal watchdog already running (pid={adopted}).",
        }

    # 3. Launch a new watchdog.
    if not SIGNAL_WATCHDOG_SCRIPT.exists():
        return {
            "ok": False,
            "pid": None,
            "message": f"Signal watchdog script missing: {SIGNAL_WATCHDOG_SCRIPT}",
        }

    env = os.environ.copy()
    if signal_exchange:
        env["ML_SIGNAL_EXCHANGE"] = str(signal_exchange)
        env["ML_PRIMARY_EXCHANGE"] = str(signal_exchange)

    _emit(
        "signal_watchdog_launch_requested",
        "Client is launching the signal watchdog.",
        source=source,
        reason=reason,
        target_exchange=signal_exchange,
        log_path=str(SIGNAL_WATCHDOG_LOG_FILE),
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(  # noqa: S603
        build_watchdog_command(),
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    # Give the watchdog a moment to either stay alive or get replaced by the
    # singleton lock in `signal_runtime_watchdog.main`.
    time.sleep(1.5)

    if not _pid_alive(proc.pid):
        # Launch wrapper exited — check whether a live watchdog exists (the
        # singleton lock inside the watchdog may have caused this wrapper to
        # exit because another watchdog is already running).
        external = _find_watchdog_pids()
        if external:
            adopted = int(external[-1])
            write_pid(SIGNAL_WATCHDOG_PID_FILE, adopted)
            return {
                "ok": True,
                "pid": adopted,
                "adopted": True,
                "message": f"Signal watchdog already running (pid={adopted}).",
            }
        return {
            "ok": False,
            "pid": None,
            "message": "Signal watchdog failed to stay alive after startup.",
        }

    write_pid(SIGNAL_WATCHDOG_PID_FILE, proc.pid)
    _emit(
        "signal_watchdog_start_requested",
        "Client ensured the signal watchdog is running.",
        source=source,
        pid=proc.pid,
        reason=reason,
        target_exchange=signal_exchange,
    )
    return {
        "ok": True,
        "pid": int(proc.pid),
        "adopted": False,
        "message": f"Started signal watchdog (pid={proc.pid}).",
    }
