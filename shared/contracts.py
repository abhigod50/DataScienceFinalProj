from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .io import append_jsonl, read_json_object, write_json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_runtime_event(source: str, event: str, message: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "source": source,
        "event": event,
        "message": message,
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    return payload


def append_runtime_event(
    path: Path,
    *,
    source: str,
    event: str,
    message: str,
    **fields: Any,
) -> dict[str, Any]:
    payload = build_runtime_event(source=source, event=event, message=message, **fields)
    append_jsonl(path, payload)
    return payload


def build_training_status(state: str, stage: str, progress_pct: int, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": state,
        "stage": stage,
        "progress_pct": int(progress_pct),
        "updated_at": now_iso(),
    }
    payload.update(extra)
    return payload


def merge_training_status(
    path: Path,
    *,
    state: str,
    stage: str,
    progress_pct: int,
    **extra: Any,
) -> dict[str, Any]:
    status = read_json_object(path)
    status.update(build_training_status(state=state, stage=stage, progress_pct=progress_pct, **extra))
    write_json(path, status)
    return status


def update_supervisor_state(
    current_state: dict[str, Any],
    *,
    desired_running: bool,
    source: str,
    reason: str,
    signal_exchange: str | None = None,
    suppress_restart_seconds: int = 0,
) -> dict[str, Any]:
    effective_signal_exchange = signal_exchange
    if not effective_signal_exchange:
        if isinstance(current_state.get("target_exchange"), str):
            effective_signal_exchange = current_state.get("target_exchange")
        elif isinstance(current_state.get("signal_exchange"), str):
            effective_signal_exchange = current_state.get("signal_exchange")

    payload: dict[str, Any] = dict(current_state)
    payload.update(
        {
            "desired_running": bool(desired_running),
            "updated_at": now_iso(),
            "source": source,
            "reason": reason,
            "target_exchange": effective_signal_exchange,
            "signal_exchange": effective_signal_exchange,
            "suppress_restart_until": None,
        }
    )
    if suppress_restart_seconds > 0:
        payload["suppress_restart_until"] = (
            datetime.now(timezone.utc) + timedelta(seconds=suppress_restart_seconds)
        ).isoformat()
    return payload

