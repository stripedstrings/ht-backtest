"""Claude translation: audit intake → candidate YAML dict."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import yaml

# Allowlisted dry-count methods the worker can execute without new Strategy code.
DRY_COUNT_METHODS = frozenset(
    {
        "kz_first_raid_reclaim",
        "kz_first_30m_raid_reclaim",
        "raid_reclaim_all",
        "raid_reclaim_vol_high",
        "raid_reclaim_vol_low",
        "london_ny_same_direction",
        "failed_raid_next_session_fade",
    }
)

CATEGORIES = frozenset({"timing", "volume", "range", "cross-asset", "pattern"})

SYSTEM_PROMPT = """You translate discretionary trading audit intakes into a strict YAML candidate
for a crypto futures research engine (15m bars, Binance USDM cache, train/holdout split).

Rules:
- Output ONLY valid YAML (no markdown fences).
- Prefer composing from known session-range raid/reclaim geometry; do not invent
  look-ahead indicators (no centered SMA, no future shift, no security() higher-TF
  without confirmed closed bars).
- dry_count.method MUST be one of:
  kz_first_raid_reclaim, kz_first_30m_raid_reclaim, raid_reclaim_all,
  raid_reclaim_vol_high, raid_reclaim_vol_low, london_ny_same_direction,
  failed_raid_next_session_fade
- If the idea cannot map to those methods, set dry_count.method to null and
  rejection_hint to "untranslatable_dry_count".
- theoretical_category must be one of: timing, volume, range, cross-asset, pattern
- confidence is your 0.0–1.0 belief that the YAML matches the intake without look-ahead.
- Flag repainting_risk / look_ahead_risk as true if the intake implies them.

Required YAML shape:
candidate_id: snake_case_id
title: string
theoretical_category: timing|volume|range|cross-asset|pattern
signal_type: snake_case
plain_english_description: string (why it might beat a coin + mechanics)
timeframe: string
instrument_type: crypto|FX|stocks
entry: {plain_english, parameters}
stop: {plain_english, parameters}
filters: []
dry_count:
  method: <allowlisted or null>
  params: {}
  min_n: 200
confidence: 0.0-1.0
repainting_risk: false
look_ahead_risk: false
rejection_hint: ""   # or untranslatable_dry_count / insufficient_data
key_parameters: {}
"""


def _extract_yaml(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:yaml|yml)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("translator did not return a YAML mapping")
    return data


def normalize_candidate(raw: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    method = raw.get("dry_count", {}) or {}
    if isinstance(method, dict):
        m = method.get("method")
        params = dict(method.get("params") or {})
        min_n = int(method.get("min_n") or 200)
    else:
        m, params, min_n = None, {}, 200
    if m is not None and m not in DRY_COUNT_METHODS:
        raw["rejection_hint"] = raw.get("rejection_hint") or "untranslatable_dry_count"
        m = None

    cat = str(raw.get("theoretical_category") or "pattern")
    if cat not in CATEGORIES:
        cat = "pattern"

    conf = float(raw.get("confidence") or 0.5)
    conf = max(0.0, min(1.0, conf))

    cid = str(raw.get("candidate_id") or "candidate").strip().lower()
    cid = re.sub(r"[^a-z0-9_]+", "_", cid).strip("_") or "candidate"

    return {
        "candidate_id": cid,
        "title": str(raw.get("title") or intake.get("title") or cid),
        "theoretical_category": cat,
        "signal_type": str(raw.get("signal_type") or cid),
        "plain_english_description": str(
            raw.get("plain_english_description")
            or intake.get("theoretical_spine")
            or intake["entry"]["plain_english"]
        ),
        "timeframe": str(raw.get("timeframe") or intake["timeframe"]),
        "instrument_type": str(raw.get("instrument_type") or intake["instrument_type"]),
        "entry": {
            "plain_english": str((raw.get("entry") or {}).get("plain_english") or intake["entry"]["plain_english"]),
            "parameters": dict((raw.get("entry") or {}).get("parameters") or intake["entry"]["indicator_parameters"]),
        },
        "stop": {
            "plain_english": str((raw.get("stop") or {}).get("plain_english") or intake["stop"]["plain_english"]),
            "parameters": dict((raw.get("stop") or {}).get("parameters") or intake["stop"]["parameters"]),
        },
        "filters": list(raw.get("filters") or intake.get("filters") or []),
        "dry_count": {"method": m, "params": params, "min_n": min_n},
        "confidence": conf,
        "repainting_risk": bool(raw.get("repainting_risk")),
        "look_ahead_risk": bool(raw.get("look_ahead_risk")),
        "rejection_hint": str(raw.get("rejection_hint") or ""),
        "key_parameters": dict(raw.get("key_parameters") or {}),
        "intake_title": intake.get("title"),
        "source": intake.get("source") or "",
    }


def translate_with_fixture(intake: dict[str, Any]) -> dict[str, Any]:
    """Offline deterministic translation for tests / no API key."""
    params = intake["entry"].get("indicator_parameters") or {}
    text = (intake["entry"]["plain_english"] + " " + intake.get("theoretical_spine", "")).lower()
    method = "raid_reclaim_all"
    cat = "pattern"
    if "volume" in text and ("high" in text or "absor" in text):
        method = "raid_reclaim_vol_high"
        cat = "volume"
    if "volume" in text and ("low" in text or "vacuum" in text):
        method = "raid_reclaim_vol_low"
        cat = "volume"
    if "fail" in text and "raid" in text:
        method = "failed_raid_next_session_fade"
        cat = "pattern"
    if "london" in text and "ny" in text and any(
        k in text for k in ("same direction", "same-dir", "then ny", "ny confirm", "both sessions")
    ):
        method = "london_ny_same_direction"
        cat = "pattern"
    if "first" in text and "raid" in text:
        method = "kz_first_raid_reclaim"
        cat = "timing"
    if ("30" in text or "opening half" in text or "first 30" in text) and "raid" in text:
        method = "kz_first_30m_raid_reclaim"
        cat = "timing"

    raw = {
        "candidate_id": re.sub(r"[^a-z0-9_]+", "_", intake["title"].lower()).strip("_") or "fixture_candidate",
        "title": intake["title"],
        "theoretical_category": cat,
        "signal_type": method,
        "plain_english_description": intake.get("theoretical_spine") or intake["entry"]["plain_english"],
        "timeframe": intake["timeframe"],
        "instrument_type": intake["instrument_type"],
        "entry": {"plain_english": intake["entry"]["plain_english"], "parameters": params},
        "stop": {"plain_english": intake["stop"]["plain_english"], "parameters": intake["stop"]["parameters"]},
        "filters": intake.get("filters") or [],
        "dry_count": {"method": method, "params": {"reclaim_bars": int(params.get("reclaim_bars", 3))}, "min_n": 200},
        "confidence": 0.75,
        "repainting_risk": False,
        "look_ahead_risk": False,
        "rejection_hint": "",
        "key_parameters": params,
    }
    return normalize_candidate(raw, intake)


def translate_with_claude(intake: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    """Call Anthropic Messages API. Requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set; use --fixture to translate offline")

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install anthropic: pip install anthropic") from e

    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic(api_key=api_key)
    user = "Audit intake JSON:\n" + json.dumps(intake, indent=2)
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    raw = _extract_yaml("\n".join(text_parts))
    return normalize_candidate(raw, intake)


def translate_intake(intake: dict[str, Any], *, fixture: bool = False, model: str | None = None) -> dict[str, Any]:
    if fixture or os.environ.get("HT_INTAKE_FIXTURE", "").lower() in ("1", "true", "yes"):
        return translate_with_fixture(intake)
    return translate_with_claude(intake, model=model)
