"""Golden: kz_first_raid_reclaim through queue → completed; match hypothesis log ±0.5pp."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.discovery.worker import run_queue  # noqa: E402
from ht_backtest.memory.hypothesis_log import load_log  # noqa: E402

TOLERANCE_PP = 0.5
CANDIDATES = ROOT / "data" / "candidates"
MEMORY = ROOT / "data" / "memory" / "hypothesis_log.csv"


def _write_golden_queued() -> Path:
    queued = CANDIDATES / "queued"
    queued.mkdir(parents=True, exist_ok=True)
    # Clear other queued yaml so worker only runs this golden
    for p in queued.glob("*.yaml"):
        p.unlink()
    for p in queued.glob("*_result.json"):
        p.unlink()
    cand = {
        "candidate_id": "kz_first_raid_reclaim",
        "title": "First killzone raid reclaim (worker golden)",
        "theoretical_category": "timing",
        "signal_type": "first_kz_raid_reclaim",
        "plain_english_description": (
            "Why it might beat a coin: ICT’s judas-swing claim — the first raid of a "
            "killzone is the real stop-run that clears resting liquidity; later raids "
            "are often continuation or noise. A reclaim after that first raid traps the "
            "traders who chased the sweep, so the move back through the edge has a "
            "defined pool of losers funding the reversal. Mechanics: in London or NY, "
            "take only the first session-range-edge raid of that killzone instance; "
            "enter on close back inside within 3 bars; stop beyond the sweep extreme; "
            "target the far session edge (or 2R)."
        ),
        "timeframe": "15m",
        "instrument_type": "crypto",
        "entry": {
            "plain_english": "First raid of killzone then reclaim.",
            "parameters": {"reclaim_bars": 3},
        },
        "stop": {"plain_english": "Beyond sweep extreme", "parameters": {}},
        "filters": [],
        "dry_count": {"method": "kz_first_raid_reclaim", "params": {"reclaim_bars": 3}, "min_n": 200},
        "confidence": 1.0,
        "repainting_risk": False,
        "look_ahead_risk": False,
        "rejection_hint": "",
        "key_parameters": {"reclaim_bars": 3},
    }
    path = queued / "kz_first_raid_reclaim_golden.yaml"
    path.write_text(yaml.safe_dump(cand, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _parse_log_best(summary: str) -> tuple[float, float, int]:
    m = re.search(
        r"best_diff_pp=([+-]?\d+(?:\.\d+)?)\s+at\s+(\d+(?:\.\d+)?)R\s+n=(\d+)",
        str(summary),
    )
    if not m:
        raise ValueError(f"cannot parse training_result_summary: {summary!r}")
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def main() -> int:
    print("=" * 72)
    print("WORKER GOLDEN — kz_first_raid_reclaim queue → completed vs hypothesis log")
    print("=" * 72)
    if not MEMORY.exists():
        print(f"MISSING memory log: {MEMORY}", file=sys.stderr)
        return 2
    log = load_log(MEMORY)
    prior = log[log["strategy_id"].astype(str) == "kz_first_raid_reclaim"]
    if prior.empty:
        print("MISSING kz_first_raid_reclaim in hypothesis_log.csv", file=sys.stderr)
        return 2
    # Use the earliest backfill row (known anchor), not a later worker append
    anchor = prior.iloc[0]
    exp_diff, exp_t, exp_n = _parse_log_best(anchor["training_result_summary"])
    print(f"anchor log: {anchor['training_result_summary']}  hash={anchor['parameter_hash']}")

    _write_golden_queued()
    results = run_queue(
        root=CANDIDATES,
        limit=1,
        split_path=str(ROOT / "specs" / "splits" / "v1.json"),
        cache_dir=str(ROOT / "data" / "raw"),
        workers=4,
    )
    if not results or results[0].get("status") != "completed":
        print(f"GOLDEN FAIL: worker result={results}", file=sys.stderr)
        return 1

    reach_path = CANDIDATES / "completed" / "kz_first_raid_reclaim" / "training_reach_table.csv"
    if not reach_path.exists():
        print(f"MISSING {reach_path}", file=sys.stderr)
        return 1
    reach = pd.read_csv(reach_path)
    row = reach[reach["target_R"] == exp_t]
    if row.empty:
        # fall back to best live row
        best_i = reach["diff_pp"].idxmax()
        live = reach.loc[best_i]
        print(f"WARN: expected horizon {exp_t}R missing; comparing live best {live['target_R']}R")
    else:
        live = row.iloc[0]

    live_diff = float(live["diff_pp"])
    live_n = int(live["n"])
    delta = live_diff - exp_diff
    print(f"live: diff_pp={live_diff:+.4f} at {float(live['target_R']):.1f}R n={live_n}")
    print(f"delta vs log best: {delta:+.4f}pp (tolerance ±{TOLERANCE_PP}pp)")
    print(f"n: live={live_n} log={exp_n} delta={live_n - exp_n}")

    failures = []
    if abs(delta) > TOLERANCE_PP:
        failures.append(f"diff_pp drifted {delta:+.4f}pp")
    if live_n != exp_n:
        failures.append(f"n drifted live={live_n} log={exp_n}")

    # Also check all horizons vs prior completed batch table if present
    ref_batch = (
        ROOT
        / "data"
        / "runs"
        / "batch_v1_20260816T130731Z"
        / "kz_first_raid_reclaim"
        / "training_reach_table.csv"
    )
    if ref_batch.exists():
        ref = pd.read_csv(ref_batch)
        print()
        print("per-horizon vs prior batch table:")
        for _, rrow in ref.iterrows():
            t = float(rrow["target_R"])
            lrow = reach[reach["target_R"] == t].iloc[0]
            d = float(lrow["diff_pp"]) - float(rrow["diff_pp"])
            print(f"  {t:.1f}R  live={lrow['diff_pp']:+.4f}  ref={rrow['diff_pp']:+.4f}  d={d:+.4f}pp")
            if abs(d) > TOLERANCE_PP:
                failures.append(f"{t}R drifted {d:+.4f}pp vs batch ref")

    print("=" * 72)
    if failures:
        print("GOLDEN TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GOLDEN TEST PASSED — queue→train matches hypothesis log within tolerance.")
    print("Holdout remains sealed; human reviews data/candidates/completed/.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
