"""Static risk heuristics on intake text + candidate YAML."""

from __future__ import annotations

import re
from typing import Any

# Phrases that usually imply repaint / look-ahead when used as entry gates.
_REPAINT_PATTERNS = [
    r"\brepaint",
    r"\bnon[- ]repainting\b",  # often a lie — still flag for review via look_ahead if paired with pivots
    r"\bzigzag\b",
    r"\bfuture\s+bar",
    r"\bcenter(ed)?\s+(ma|sma|ema|average)",
    r"\bheiken\s*ashi\s+signal",  # often misused with open of current HA bar
    r"\brequest\.security",
    r"\bhigher[- ]timeframe\b.*\bcurrent\b",
    r"\bshift\s*\(\s*-",
    r"\b\[-\d+\]",  # pine future offset
]

_LOOKAHEAD_PATTERNS = [
    r"\bconfirm(ed)?\s+after\s+the\s+fact",
    r"\buse\s+tomorrow",
    r"\bnext\s+day'?s\s+open\s+known",
    r"\bforward[- ]fill\s+from\s+future",
    r"\bintrabar\s+wick\s+then\s+revise",
]


def _blob(intake: dict[str, Any], candidate: dict[str, Any] | None = None) -> str:
    parts = [
        intake.get("entry", {}).get("plain_english", ""),
        intake.get("stop", {}).get("plain_english", ""),
        intake.get("theoretical_spine", ""),
        " ".join(f.get("plain_english", "") for f in intake.get("filters") or []),
    ]
    if candidate:
        parts.append(candidate.get("plain_english_description", ""))
        parts.append(str(candidate.get("entry", {})))
    return " ".join(parts).lower()


def scan_risks(intake: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    text = _blob(intake, candidate)
    repaint_hits = [p for p in _REPAINT_PATTERNS if re.search(p, text, re.I)]
    look_hits = [p for p in _LOOKAHEAD_PATTERNS if re.search(p, text, re.I)]

    repainting = bool(repaint_hits) or bool(candidate and candidate.get("repainting_risk"))
    look_ahead = bool(look_hits) or bool(candidate and candidate.get("look_ahead_risk"))

    # Non-crypto data not in this engine's cache yet.
    insufficient_data = intake.get("instrument_type") != "crypto"
    if candidate and candidate.get("timeframe") not in (None, "15m"):
        # Engine dry-count / train path is 15m-locked for MVP.
        if str(candidate.get("timeframe")) != "15m":
            insufficient_data = True

    reasons: list[str] = []
    if repainting:
        reasons.append("repainting_indicator_detected")
    if look_ahead:
        reasons.append("look_ahead_risk")
    if insufficient_data:
        if intake.get("instrument_type") != "crypto":
            reasons.append("insufficient_data:engine_has_crypto_15m_cache_only")
        elif candidate and str(candidate.get("timeframe")) != "15m":
            reasons.append("insufficient_data:mvp_dry_count_is_15m_only")

    return {
        "repainting_risk": repainting,
        "look_ahead_risk": look_ahead,
        "insufficient_data": insufficient_data,
        "reasons": reasons,
        "repaint_pattern_hits": repaint_hits,
        "lookahead_pattern_hits": look_hits,
    }
