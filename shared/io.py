from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path, default={})
    return payload if isinstance(payload, dict) else {}


def write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, default=str),
        encoding="utf-8",
    )


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            safe_unlink(tmp_path)


def write_json_atomic_with_retry(
    path: Path,
    payload: Any,
    *,
    retries: int = 5,
    delay_s: float = 0.15,
    fallback_direct: bool = False,
    indent: int | None = 2,
    ensure_ascii: bool = True,
) -> None:
    last_error: PermissionError | None = None
    for attempt in range(1, retries + 1):
        try:
            write_json_atomic(path, payload, indent=indent, ensure_ascii=ensure_ascii)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == retries:
                if fallback_direct:
                    write_json(path, payload, indent=indent, ensure_ascii=ensure_ascii)
                    return
                raise
            time.sleep(delay_s * attempt)
    if last_error is not None:
        raise last_error


def append_jsonl(path: Path, payload: dict[str, Any], *, ensure_ascii: bool = True) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=ensure_ascii, default=str) + "\n")
    except OSError:
        pass


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def write_pid(path: Path, pid: int, *, retries: int = 1, delay_s: float = 0.0) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            path.write_text(str(pid), encoding="utf-8")
            return True
        except OSError:
            if attempt == retries:
                return False
            time.sleep(delay_s)
    return False


def acquire_singleton_lock(
    lock_path: Path,
    pid_path: Path,
    *,
    owner_pid: int | None = None,
    should_keep_existing: Callable[[int], bool] | None = None,
    retries: int = 2,
    pid_write_retries: int = 1,
    pid_write_delay_s: float = 0.0,
) -> tuple[bool, int | None]:
    owner_pid = int(owner_pid or os.getpid())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(retries):
        try:
            with lock_path.open("x", encoding="utf-8") as f:
                f.write(str(owner_pid))
            write_pid(pid_path, owner_pid, retries=pid_write_retries, delay_s=pid_write_delay_s)
            return True, None
        except FileExistsError:
            existing_pid = read_pid(lock_path)
            keep_existing = bool(existing_pid) and (
                should_keep_existing(existing_pid) if should_keep_existing else True
            )
            if keep_existing and existing_pid is not None:
                write_pid(
                    pid_path,
                    existing_pid,
                    retries=pid_write_retries,
                    delay_s=pid_write_delay_s,
                )
                return False, existing_pid
            safe_unlink(lock_path)

    return False, None


def release_singleton_lock(
    lock_path: Path,
    pid_path: Path,
    *,
    owner_pid: int | None = None,
) -> None:
    owner_pid = int(owner_pid or os.getpid())
    held_by = read_pid(lock_path)
    if held_by == owner_pid:
        safe_unlink(lock_path)

    current_pid = read_pid(pid_path)
    if current_pid == owner_pid:
        safe_unlink(pid_path)
