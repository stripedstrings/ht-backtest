"""Queue worker: compile queued YAML → Strategy → train → memory → completed/failed."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ht_backtest.data.split import SplitManifest
from ht_backtest.discovery.compile import CompileError, compile_candidate_file, register_compiler_targets
from ht_backtest.discovery.dry_count_candidate import dry_count_candidate
from ht_backtest.discovery.queue import DEFAULT_ROOT, ensure_dirs, write_candidate_yaml
from ht_backtest.memory.hypothesis_log import append_records, records_from_batch_comparison
from ht_backtest.reports.comparison import strategy_comparison_table
from ht_backtest.reports.reach import format_reach_table, reach_vs_random_walk
from ht_backtest.reports.universe_report import generate_pooled_trades
from ht_backtest.strategies.registry import get_strategy, list_strategies


def _ensure_worker_dirs(root: Path) -> Path:
    ensure_dirs(root)
    for sub in ("completed", "failed", "processing"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def list_queued_candidates(root: str | Path | None = None) -> list[Path]:
    root = Path(root) if root else DEFAULT_ROOT
    queued = root / "queued"
    if not queued.exists():
        return []
    return sorted(
        p for p in queued.glob("*.yaml") if not p.name.endswith("_result.yaml")
    )


def _fail(
    root: Path,
    candidate_path: Path,
    candidate: dict[str, Any] | None,
    reason: str,
    *,
    extra: dict[str, Any] | None = None,
    log_fn=print,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cid = (candidate or {}).get("candidate_id") or candidate_path.stem
    dest_dir = root / "failed" / f"{cid}_{stamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "reason": reason,
        "candidate_id": cid,
        "source_yaml": str(candidate_path),
        "created_at_utc": stamp,
        **(extra or {}),
    }
    (dest_dir / "failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if candidate is not None:
        write_candidate_yaml(candidate, dest_dir / "candidate.yaml")
    elif candidate_path.exists():
        shutil.copy2(candidate_path, dest_dir / "candidate.yaml")
    # Remove from queued / processing
    for p in (candidate_path,):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    sibling = candidate_path.with_name(candidate_path.stem + "_result.json")
    if sibling.exists():
        try:
            sibling.unlink()
        except OSError:
            pass
    log_fn(f"FAILED {cid}: {reason} → {dest_dir}")
    return payload


def process_one_candidate(
    candidate_path: Path,
    *,
    root: Path,
    split_path: str | Path = "specs/splits/v1.json",
    cache_dir: str = "data/raw",
    workers: int = 4,
    mfe_win: int = 100,
    min_edge_pp: float = 5.0,
    min_n: int = 200,
    exchange_id: str = "binanceusdm",
    skip_dry_count: bool = False,
    log_fn=print,
) -> dict[str, Any]:
    _ensure_worker_dirs(root)
    processing = root / "processing" / candidate_path.name
    processing.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        shutil.move(str(candidate_path), str(processing))
    candidate_path = processing

    try:
        candidate, strategy = compile_candidate_file(candidate_path)
    except CompileError as exc:
        return _fail(root, candidate_path, None, f"compilation_failure: {exc}", log_fn=log_fn)
    except Exception as exc:  # noqa: BLE001
        return _fail(root, candidate_path, None, f"compilation_failure: {exc}", log_fn=log_fn)

    meta = strategy.metadata()
    log_fn(f"=== worker compile ok: {meta.id} hash={meta.parameter_hash} method={candidate.get('dry_count', {}).get('method')} ===")

    if not skip_dry_count:
        log_fn("dry-count confirmation (full train universe)...")
        dry = dry_count_candidate(candidate, split_path=split_path, cache_dir=cache_dir)
        log_fn(f"  dry-count n={dry.get('n')} min_n={dry.get('min_n')} ok={dry.get('ok')}")
        if not dry.get("ok"):
            return _fail(
                root,
                candidate_path,
                candidate,
                dry.get("reason") or "n_too_low",
                extra={"dry_count": dry},
                log_fn=log_fn,
            )
    else:
        dry = {"skipped": True}

    register_compiler_targets()
    # Multiprocess pool re-imports registry in children — use canonical registry id.
    pool_workers = workers
    if strategy.registry_id not in list_strategies():
        log_fn(f"registry miss for {strategy.registry_id}; forcing workers=1")
        pool_workers = 1

    split = SplitManifest.load(split_path)
    t0 = time.time()
    try:
        # Prefer pooled name = registry_id so workers>1 resolve via get_strategy.
        run_strategy = strategy
        if pool_workers > 1:
            run_strategy = get_strategy(strategy.registry_id)
        trades, timings = generate_pooled_trades(
            strategy=run_strategy,
            split=split,
            timeframe=str(candidate.get("timeframe") or "15m"),
            exchange_id=exchange_id,
            cache_dir=cache_dir,
            mfe_win=mfe_win,
            workers=pool_workers,
            strategy_name=strategy.registry_id,
            split_path=split_path,
            use_primitive_cache=True,
            log_fn=log_fn,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            root,
            candidate_path,
            candidate,
            f"training_failure: {exc}",
            extra={"dry_count": dry},
            log_fn=log_fn,
        )
    wall = time.time() - t0

    if not trades.empty:
        trades = trades.copy()
        trades["strategy_description"] = meta.description
        trades["strategy_id"] = meta.id

    train = trades[trades["split"] == "train"] if not trades.empty else trades
    reach = reach_vs_random_walk(train)
    comparison = strategy_comparison_table(
        {meta.id: trades},
        split="train",
        min_edge_pp=min_edge_pp,
        min_n=min_n,
    )
    promoted = bool(comparison["promoted"].any()) if not comparison.empty else False

    # Memory append
    records = records_from_batch_comparison(
        comparison,
        strategies={meta.id: strategy},
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    mem_path = append_records(records)
    log_fn(f"hypothesis memory: appended {len(records)} row(s) → {mem_path}")

    # Completed artifacts
    out_dir = root / "completed" / meta.id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_yaml(candidate, out_dir / "candidate.yaml")
    reach.to_csv(out_dir / "training_reach_table.csv", index=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    if not trades.empty:
        trades.to_parquet(out_dir / "trades.parquet", index=False)

    summary = {
        "status": "completed",
        "strategy_id": meta.id,
        "parameter_hash": meta.parameter_hash,
        "promoted": promoted,
        "min_edge_pp": min_edge_pp,
        "min_n": min_n,
        "wall_seconds": wall,
        "dry_count": dry,
        "stage_timings": timings.to_dict() if hasattr(timings, "to_dict") else {},
        "training_result_summary": records[0]["training_result_summary"] if records else "",
        "note": "Holdout not scored. Human reviews completed/ before any holdout unlock.",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "promotion.json").write_text(
        json.dumps({"promoted": promoted, "strategy_id": meta.id, "rows": comparison.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )

    log_fn(format_reach_table(reach, f"TRAIN reach — {meta.id}"))
    log_fn(f"promoted={promoted}  → {out_dir}")

    if candidate_path.exists():
        candidate_path.unlink()
    sibling = candidate_path.with_name(candidate_path.stem + "_result.json")
    if sibling.exists():
        sibling.unlink()

    return summary


def run_queue(
    *,
    root: str | Path | None = None,
    limit: int | None = None,
    loop: bool = False,
    interval_s: float = 60.0,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Process queued candidates once, or loop on a schedule."""
    root_p = _ensure_worker_dirs(Path(root) if root else DEFAULT_ROOT)
    log_fn = kwargs.get("log_fn", print)
    results: list[dict[str, Any]] = []

    def _once() -> list[dict[str, Any]]:
        paths = list_queued_candidates(root_p)
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            log_fn("queue empty")
            return []
        out = []
        for path in paths:
            log_fn(f"\n>>> processing {path.name}")
            out.append(process_one_candidate(path, root=root_p, **{k: v for k, v in kwargs.items() if k != "log_fn"}, log_fn=log_fn))
        return out

    if not loop:
        return _once()

    log_fn(f"worker loop every {interval_s:.0f}s on {root_p / 'queued'}")
    while True:
        results.extend(_once())
        time.sleep(interval_s)
