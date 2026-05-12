from __future__ import annotations

from pathlib import Path


def normalize_exchange_id(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return ""

    for suffix in ("_paper_trade", "_papertrade"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    aliases = {
        "coinbase_advanced_trade": "coinbaseadvanced",
        "gate_io": "gateio",
    }
    return aliases.get(normalized, normalized)


def read_simple_yaml_scalar(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    prefix = f"{key}:"
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or not line.startswith(prefix):
                continue
            value = line[len(prefix):].strip().strip('"').strip("'")
            return value or None
    except Exception:
        return None
    return None


def read_csv_setting(path: Path, key: str) -> list[str]:
    raw = read_simple_yaml_scalar(path, key)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def configured_hummingbot_pairs(
    script_config_path: Path,
    default: list[str] | None = None,
) -> list[str]:
    pairs = read_csv_setting(script_config_path, "trading_pairs_csv")
    if pairs:
        return pairs
    return list(default or [])


def read_hummingbot_exchange(script_config_path: Path) -> str:
    return normalize_exchange_id(read_simple_yaml_scalar(script_config_path, "exchange"))

