"""Intake → Claude (or fixture) → risk scan → dry-count → queue/reject."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ht_backtest.discovery.dry_count_candidate import dry_count_candidate
from ht_backtest.discovery.intake import load_intake, validate_intake
from ht_backtest.discovery.queue import ensure_dirs, queue_or_reject
from ht_backtest.discovery.risk_scan import scan_risks
from ht_backtest.discovery.translate import translate_intake
from ht_backtest.memory.hypothesis_log import prior_for_category


def run_intake_pipeline(
    intake: dict[str, Any] | str | Path,
    *,
    fixture: bool = False,
    skip_dry_count: bool = False,
    max_symbols: int | None = None,
    candidates_root: str | Path | None = None,
    split_path: str | Path = "specs/splits/v1.json",
    cache_dir: str = "data/raw",
    model: str | None = None,
    log_fn=print,
) -> dict[str, Any]:
    if not isinstance(intake, dict):
        intake = load_intake(intake)
    else:
        intake = validate_intake(intake)

    root = ensure_dirs(candidates_root)
    # Archive intake
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    intake_path = root / "intake" / f"intake_{stamp}.yaml"
    intake_path.write_text(yaml.safe_dump(intake, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log_fn(f"intake archived: {intake_path}")

    # Fast reject: non-crypto / wrong universe before spending Claude tokens
    early_risk = scan_risks(intake, None)
    if early_risk["insufficient_data"] and intake["instrument_type"] != "crypto":
        candidate = {
            "candidate_id": "rejected_insufficient_data",
            "title": intake["title"],
            "theoretical_category": "pattern",
            "signal_type": "none",
            "plain_english_description": intake["entry"]["plain_english"],
            "timeframe": intake["timeframe"],
            "instrument_type": intake["instrument_type"],
            "entry": intake["entry"],
            "stop": intake["stop"],
            "filters": intake["filters"],
            "dry_count": {"method": None, "params": {}, "min_n": 200},
            "confidence": 0.0,
            "repainting_risk": False,
            "look_ahead_risk": False,
            "rejection_hint": "insufficient_data",
            "key_parameters": {},
        }
        result = queue_or_reject(
            candidate,
            accepted=False,
            reject_reasons=early_risk["reasons"],
            dry_count={"n": 0, "ok": False, "reason": "insufficient_data"},
            risk=early_risk,
            root=root,
        )
        log_fn(f"REJECTED: {', '.join(early_risk['reasons'])}")
        return result

    log_fn("translating intake → candidate YAML...")
    candidate = translate_intake(intake, fixture=fixture, model=model)
    log_fn(
        f"  id={candidate['candidate_id']}  category={candidate['theoretical_category']}  "
        f"dry_count.method={candidate['dry_count'].get('method')}  confidence={candidate['confidence']:.2f}"
    )

    # Memory prior (informational; does not auto-reject MVP)
    try:
        prior = prior_for_category(candidate["theoretical_category"])
        log_fn("memory prior:\n" + prior)
        candidate["memory_prior"] = prior
    except Exception as exc:  # noqa: BLE001
        log_fn(f"memory prior skipped: {exc}")

    risk = scan_risks(intake, candidate)
    candidate["repainting_risk"] = risk["repainting_risk"]
    candidate["look_ahead_risk"] = risk["look_ahead_risk"]

    # Adjust confidence for risk flags
    conf = float(candidate["confidence"])
    if risk["repainting_risk"] or risk["look_ahead_risk"]:
        conf = min(conf, 0.35)
    if not candidate["dry_count"].get("method"):
        conf = min(conf, 0.40)
    candidate["confidence"] = conf

    reject_reasons: list[str] = []
    if risk["repainting_risk"]:
        reject_reasons.append("repainting_indicator_detected")
    if risk["look_ahead_risk"]:
        reject_reasons.append("look_ahead_risk")
    if risk["insufficient_data"]:
        reject_reasons.extend([r for r in risk["reasons"] if r.startswith("insufficient_data")])
    if candidate.get("rejection_hint") == "untranslatable_dry_count" or not candidate["dry_count"].get("method"):
        reject_reasons.append("untranslatable_dry_count")

    dry: dict[str, Any] = {"n": None, "ok": False, "reason": "skipped"}
    if not reject_reasons and not skip_dry_count:
        log_fn("running dry-count on train universe...")
        dry = dry_count_candidate(
            candidate,
            split_path=split_path,
            cache_dir=cache_dir,
            max_symbols=max_symbols,
        )
        log_fn(f"  dry-count n={dry.get('n')}  min_n={dry.get('min_n')}  reason={dry.get('reason') or 'ok'}")
        if not dry.get("ok"):
            reject_reasons.append(dry.get("reason") or "n_too_low")
    elif skip_dry_count and not reject_reasons:
        dry = {"n": None, "ok": True, "reason": "dry_count_skipped"}

    accepted = len(reject_reasons) == 0
    result = queue_or_reject(
        candidate,
        accepted=accepted,
        reject_reasons=reject_reasons,
        dry_count=dry,
        risk=risk,
        root=root,
    )
    result["confidence"] = candidate["confidence"]
    result["candidate"] = {
        "candidate_id": candidate["candidate_id"],
        "theoretical_category": candidate["theoretical_category"],
        "signal_type": candidate["signal_type"],
        "dry_count_method": candidate["dry_count"].get("method"),
    }
    if accepted:
        log_fn(f"QUEUED for training batch: {result['candidate_yaml']}")
    else:
        log_fn(f"REJECTED ({', '.join(reject_reasons)}): {result['candidate_yaml']}")
    return result
