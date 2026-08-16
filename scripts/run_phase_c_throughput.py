"""Phase C measurement harness: profile -> cache -> vectorize, with golden checks.

Usage (repo root):
  .venv\\Scripts\\python.exe -u scripts/run_phase_c_throughput.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.data.split import SplitManifest
from ht_backtest.reports.reach import reach_vs_random_walk
from ht_backtest.reports.universe_report import generate_pooled_trades
from ht_backtest.strategies.registry import get_strategy

SPLIT = ROOT / "specs" / "splits" / "v1.json"
REF = ROOT / "data" / "reports" / "training_reach_table.csv"
PRIM_CACHE = ROOT / "data" / "cache" / "primitives"
OUT = ROOT / "data" / "reports" / "phase_c_throughput.json"
TOLERANCE_PP = 0.5


def golden_ok(trades) -> tuple[bool, str]:
    import pandas as pd

    ref = pd.read_csv(REF)
    train = trades[trades["split"] == "train"]
    live = reach_vs_random_walk(train)
    if int(live["n"].iloc[0]) != int(ref["n"].iloc[0]):
        return False, f"n drift live={live['n'].iloc[0]} ref={ref['n'].iloc[0]}"
    for _, row in ref.iterrows():
        t = float(row["target_R"])
        d = float(live[live["target_R"] == t].iloc[0]["reach_pct"]) - float(row["reach_pct"])
        if abs(d) > TOLERANCE_PP:
            return False, f"{t}R drift {d:+.4f}pp"
    return True, "ok"


def run_ht(*, use_cache: bool, workers: int, label: str) -> dict:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    if not use_cache and PRIM_CACHE.exists():
        shutil.rmtree(PRIM_CACHE, ignore_errors=True)
        print(f"cleared primitive cache at {PRIM_CACHE}")

    strategy = get_strategy("ht_v10")
    split = SplitManifest.load(SPLIT)
    t0 = time.perf_counter()
    trades, timings = generate_pooled_trades(
        strategy=strategy,
        split=split,
        timeframe="15m",
        exchange_id="binanceusdm",
        cache_dir=str(ROOT / "data" / "raw"),
        mfe_win=100,
        workers=workers,
        strategy_name="ht_v10",
        split_path=str(SPLIT),
        use_primitive_cache=use_cache,
        primitives_cache_dir=str(PRIM_CACHE),
    )
    wall = time.perf_counter() - t0
    ok, msg = golden_ok(trades)
    print(f"wall={wall:.1f}s  stages={timings.to_dict()}")
    print(f"golden: {'PASS' if ok else 'FAIL'} ({msg})")
    if not ok:
        raise SystemExit(f"GOLDEN FAILED after {label}: {msg}")
    return {
        "label": label,
        "wall_s": wall,
        "timings": timings.to_dict(),
        "golden": msg,
        "train_n": int((trades["split"] == "train").sum()),
    }


def main() -> int:
    results = []

    # Step 1: profile cold path (no cache) with workers=1 for clean stage sums
    results.append(
        run_ht(use_cache=False, workers=1, label="STEP1 profile cold (no prim cache, workers=1)")
    )

    # Step 2: warm cache then measure cache-hit path
    print("\nwarming primitive cache...")
    run_ht(use_cache=True, workers=1, label="STEP2a warm primitive cache")
    results.append(
        run_ht(use_cache=True, workers=1, label="STEP2b with primitive cache (workers=1)")
    )

    # Step 3: vectorized tracker already in place — measure with cache + workers=4
    results.append(
        run_ht(use_cache=True, workers=4, label="STEP3 cache + workers=4 (vectorized tracker)")
    )

    # Step 4: full 10-strategy batch
    from ht_backtest.reports.batch_runner import load_batch_config, run_batch

    print("\n" + "=" * 72)
    print("STEP4 full 10-strategy batch")
    print("=" * 72)
    cfg = load_batch_config(ROOT / "specs" / "batch" / "example_10.yaml")
    cfg.cache_dir = str(ROOT / "data" / "raw")
    cfg.splits_dir = str(ROOT / "specs" / "splits")
    cfg.out_dir = str(ROOT / "data" / "runs")
    cfg.use_primitive_cache = True
    t0 = time.perf_counter()
    batch_dir = run_batch(cfg)
    batch_wall = time.perf_counter() - t0
    print(f"batch wall={batch_wall:.1f}s  dir={batch_dir}")

    # single new-hypothesis style timing: one baseline on full universe
    t0 = time.perf_counter()
    strat = get_strategy("london_open_long_atr1")
    split = SplitManifest.load(SPLIT)
    trades, timings = generate_pooled_trades(
        strategy=strat,
        split=split,
        timeframe="15m",
        exchange_id="binanceusdm",
        cache_dir=str(ROOT / "data" / "raw"),
        workers=4,
        strategy_name="london_open_long_atr1",
        split_path=str(SPLIT),
        use_primitive_cache=True,
        primitives_cache_dir=str(PRIM_CACHE),
    )
    hypo_wall = time.perf_counter() - t0
    print(f"\nnew-hypothesis (london_open_long_atr1) wall={hypo_wall:.1f}s  trades={len(trades)}")

    payload = {
        "phase_a_baseline_s": 200.0,
        "phase_b_batch_baseline_s": 210.0,
        "steps": results,
        "batch_10_wall_s": batch_wall,
        "batch_dir": str(batch_dir),
        "new_hypothesis_wall_s": hypo_wall,
        "target_new_hypothesis_s": 60.0,
        "hit_60s_target": hypo_wall < 60.0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(
        f"\nSUMMARY: PhaseA baseline~200s -> HT now {results[-1]['wall_s']:.1f}s; "
        f"batch10 {batch_wall:.1f}s (was ~210s); new hypothesis {hypo_wall:.1f}s "
        f"(target <60: {'YES' if hypo_wall < 60 else 'NO'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
