"""Batch runner: queue strategies, run on one locked split, write comparison.

Non-negotiable: every strategy is scored with reach vs 1/(1+T) on train only
in the comparison table. Holdout trades are stored but not used for discovery.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ht_backtest.data.split import SplitManifest
from ht_backtest.memory.hypothesis_log import (
    append_records,
    format_prior_results_summary,
    records_from_batch_comparison,
)
from ht_backtest.reports.comparison import format_comparison_table, strategy_comparison_table
from ht_backtest.reports.reach import format_reach_table, reach_vs_random_walk
from ht_backtest.reports.universe_report import generate_pooled_trades
from ht_backtest.strategies.registry import get_strategy


@dataclass
class BatchConfig:
    split: str
    strategies: list[str]
    timeframe: str = "15m"
    exchange: str = "binanceusdm"
    cache_dir: str = "data/raw"
    splits_dir: str = "specs/splits"
    out_dir: str = "data/runs"
    mfe_win: int = 100
    workers: int = 1
    min_edge_pp: float = 5.0
    min_n: int = 200
    use_primitive_cache: bool = True
    memory_log_path: str | None = None


def load_batch_config(path: str | Path) -> BatchConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"batch config must be a mapping: {path}")
    strategies = raw.get("strategies")
    if not strategies or not isinstance(strategies, list):
        raise ValueError("batch config requires a non-empty 'strategies' list")
    split = raw.get("split")
    if not split:
        raise ValueError("batch config requires 'split'")
    return BatchConfig(
        split=str(split),
        strategies=[str(s) for s in strategies],
        timeframe=str(raw.get("timeframe", "15m")),
        exchange=str(raw.get("exchange", "binanceusdm")),
        cache_dir=str(raw.get("cache_dir", "data/raw")),
        splits_dir=str(raw.get("splits_dir", "specs/splits")),
        out_dir=str(raw.get("out_dir", "data/runs")),
        mfe_win=int(raw.get("mfe_win", 100)),
        workers=int(raw.get("workers", 1)),
        min_edge_pp=float(raw.get("min_edge_pp", 5.0)),
        min_n=int(raw.get("min_n", 200)),
        use_primitive_cache=bool(raw.get("use_primitive_cache", True)),
        memory_log_path=str(raw["memory_log_path"]) if raw.get("memory_log_path") else None,
    )


def run_batch(config: BatchConfig, log_fn=print) -> Path:
    split_path = Path(config.splits_dir) / f"{config.split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split manifest not found: {split_path}")
    split = SplitManifest.load(split_path)

    log_fn("")
    log_fn(format_prior_results_summary(config.memory_log_path))
    log_fn("")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = Path(config.out_dir) / f"batch_{config.split}_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    strategy_metas: list[dict[str, Any]] = []
    trades_by_strategy: dict[str, pd.DataFrame] = {}
    wall_times: dict[str, float] = {}
    strategy_objs: dict[str, Any] = {}

    for name in config.strategies:
        strategy = get_strategy(name)
        meta = strategy.metadata()
        strategy_objs[meta.id] = strategy
        strategy_metas.append(meta.to_dict())
        log_fn("")
        log_fn(f"=== {meta.id} v{meta.version} hash={meta.parameter_hash} ===")
        log_fn(meta.description)
        t0 = time.time()
        trades, timings = generate_pooled_trades(
            strategy=strategy,
            split=split,
            timeframe=config.timeframe,
            exchange_id=config.exchange,
            cache_dir=config.cache_dir,
            mfe_win=config.mfe_win,
            workers=config.workers,
            strategy_name=name,
            split_path=split_path,
            use_primitive_cache=getattr(config, "use_primitive_cache", True),
            log_fn=log_fn,
        )
        elapsed = time.time() - t0
        wall_times[meta.id] = elapsed
        if not trades.empty:
            trades = trades.copy()
            trades["strategy_description"] = meta.description
        strat_dir = batch_dir / meta.id
        strat_dir.mkdir(parents=True, exist_ok=True)
        with open(strat_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "strategy": meta.to_dict(),
                    "split": config.split,
                    "timeframe": config.timeframe,
                    "workers": config.workers,
                    "wall_seconds": elapsed,
                    "stage_timings": timings.to_dict(),
                },
                f,
                indent=2,
            )
        trades.to_parquet(strat_dir / "trades.parquet", index=False)
        train = trades[trades["split"] == "train"] if not trades.empty else trades
        reach = reach_vs_random_walk(train)
        reach.to_csv(strat_dir / "training_reach_table.csv", index=False)
        log_fn(format_reach_table(reach, f"TRAIN reach — {meta.id}"))
        log_fn(f"wall time: {elapsed:.1f}s")
        trades_by_strategy[meta.id] = trades

    comparison = strategy_comparison_table(
        trades_by_strategy,
        split="train",
        min_edge_pp=config.min_edge_pp,
        min_n=config.min_n,
    )
    comparison_path = batch_dir / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # Memory layer: one row per strategy from this batch (train results only here;
    # holdout_result is filled by a separate pre-registered holdout write).
    mem_records = records_from_batch_comparison(
        comparison,
        strategies=strategy_objs,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    mem_path = append_records(mem_records, config.memory_log_path)
    log_fn(f"hypothesis memory: appended {len(mem_records)} row(s) -> {mem_path}")

    manifest = {
        "batch_id": batch_dir.name,
        "created_at_utc": stamp,
        "split": config.split,
        "split_path": str(split_path),
        "timeframe": config.timeframe,
        "exchange": config.exchange,
        "mfe_win": config.mfe_win,
        "workers": config.workers,
        "min_edge_pp": config.min_edge_pp,
        "min_n": config.min_n,
        "strategies": strategy_metas,
        "wall_seconds_by_strategy": wall_times,
        "total_wall_seconds": float(sum(wall_times.values())),
        "hypothesis_log": str(mem_path),
        "note": (
            "Comparison table is TRAIN only. Holdout trades are stored per strategy "
            "but must not be used for discovery. Promotion requires >min_edge_pp at "
            "n>=min_n on train, then a separate holdout unlock. Batch rows are appended "
            "to data/memory/hypothesis_log.csv."
        ),
    }
    with open(batch_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log_fn("")
    log_fn(format_comparison_table(comparison))
    log_fn(f"\nWrote batch artifacts to {batch_dir}")
    log_fn(f"comparison: {comparison_path}")
    return batch_dir
