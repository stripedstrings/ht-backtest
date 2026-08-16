"""Discovery intake: validate human audit forms into intake dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED = ("entry", "stop", "timeframe", "instrument_type")
TIMEFRAMES = frozenset({"1m", "5m", "15m", "1h", "4h", "1d"})
INSTRUMENTS = frozenset({"crypto", "FX", "stocks"})


def load_intake(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("intake must be a YAML/JSON mapping")
    return validate_intake(raw)


def validate_intake(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"intake missing required fields: {missing}")

    entry = raw["entry"]
    if not isinstance(entry, dict) or not str(entry.get("plain_english", "")).strip():
        raise ValueError("entry.plain_english is required")
    if len(str(entry["plain_english"]).strip()) < 20:
        raise ValueError("entry.plain_english too short (min 20 chars)")

    stop = raw["stop"]
    if not isinstance(stop, dict) or not str(stop.get("plain_english", "")).strip():
        raise ValueError("stop.plain_english is required")

    tf = str(raw["timeframe"])
    if tf not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {sorted(TIMEFRAMES)}")

    inst = str(raw["instrument_type"])
    if inst not in INSTRUMENTS:
        raise ValueError(f"instrument_type must be one of {sorted(INSTRUMENTS)}")

    filters = raw.get("filters") or []
    if not isinstance(filters, list):
        raise ValueError("filters must be a list")

    out = {
        "title": str(raw.get("title") or "untitled").strip(),
        "entry": {
            "plain_english": str(entry["plain_english"]).strip(),
            "indicator_parameters": dict(entry.get("indicator_parameters") or {}),
        },
        "stop": {
            "plain_english": str(stop["plain_english"]).strip(),
            "parameters": dict(stop.get("parameters") or {}),
        },
        "filters": [
            {
                "plain_english": str(f.get("plain_english", "")).strip(),
                "parameters": dict(f.get("parameters") or {}),
            }
            for f in filters
            if isinstance(f, dict) and str(f.get("plain_english", "")).strip()
        ],
        "timeframe": tf,
        "instrument_type": inst,
        "theoretical_spine": str(raw.get("theoretical_spine") or "").strip(),
        "source": str(raw.get("source") or "").strip(),
    }
    return out
