"""Filesystem candidate queue: rejected / queued for training batch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path("data/candidates")


def ensure_dirs(root: str | Path | None = None) -> Path:
    root = Path(root) if root else DEFAULT_ROOT
    for sub in ("intake", "rejected", "queued", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_candidate_yaml(candidate: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def queue_or_reject(
    candidate: dict[str, Any],
    *,
    accepted: bool,
    reject_reasons: list[str],
    dry_count: dict[str, Any],
    risk: dict[str, Any],
    root: str | Path | None = None,
) -> dict[str, Any]:
    root = ensure_dirs(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cid = candidate.get("candidate_id", "candidate")
    base_name = f"{cid}_{stamp}"

    result = {
        "candidate_id": cid,
        "accepted": accepted,
        "reject_reasons": reject_reasons,
        "confidence": candidate.get("confidence"),
        "dry_count": dry_count,
        "risk": {k: risk[k] for k in ("repainting_risk", "look_ahead_risk", "insufficient_data", "reasons") if k in risk},
        "created_at_utc": stamp,
    }

    if accepted:
        yaml_path = root / "queued" / f"{base_name}.yaml"
        status_dir = root / "queued"
        result["status"] = "queued_for_training_batch"
    else:
        yaml_path = root / "rejected" / f"{base_name}.yaml"
        status_dir = root / "rejected"
        result["status"] = "rejected"

    # Attach audit trail onto candidate before write
    out_cand = dict(candidate)
    out_cand["intake_result"] = result
    write_candidate_yaml(out_cand, yaml_path)
    result_path = status_dir / f"{base_name}_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Also mirror under results/
    (root / "results" / f"{base_name}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["candidate_yaml"] = str(yaml_path)
    result["result_json"] = str(result_path)
    return result
