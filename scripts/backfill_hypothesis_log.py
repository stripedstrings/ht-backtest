"""Backfill hypothesis_log.csv from the Step-5 survivors batch + SMT holdout note.

Idempotent: skips strategy_id+parameter_hash pairs already present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.memory.hypothesis_log import (  # noqa: E402
    append_records,
    default_log_path,
    format_prior_results_summary,
    load_log,
    records_from_batch_comparison,
    update_holdout_result,
)
from ht_backtest.strategies.registry import get_strategy  # noqa: E402

BATCH = ROOT / "data" / "runs" / "batch_v1_20260816T130731Z"
HOLDOUT_SMT = (
    "pre_registered_6h_time_exit mean_fwd_R=+0.087 median=+0.059 "
    "n=1147 pct_pos=50.9% bar=+0.10 FAIL (specs/pre_registered/smt_6h_exit.md)"
)


def main() -> int:
    comp_path = BATCH / "comparison.csv"
    if not comp_path.exists():
        print(f"missing {comp_path}", file=sys.stderr)
        return 2
    comparison = pd.read_csv(comp_path)
    strategies = {}
    for sid in comparison["strategy_id"].unique():
        try:
            strategies[str(sid)] = get_strategy(str(sid))
        except KeyError:
            pass
    records = records_from_batch_comparison(
        comparison,
        strategies=strategies,
        date="2026-08-16",
    )
    log_path = default_log_path(ROOT)
    existing = load_log(log_path)
    if not existing.empty:
        seen = set(zip(existing["strategy_id"].astype(str), existing["parameter_hash"].astype(str)))
        records = [r for r in records if (r["strategy_id"], r["parameter_hash"]) not in seen]
    if records:
        append_records(records, log_path)
        print(f"appended {len(records)} row(s) -> {log_path}")
    else:
        print(f"no new rows (already present) -> {log_path}")

    # Attach the only pre-registered holdout result we have.
    update_holdout_result(
        "smt_trade_holder_btc_sol",
        HOLDOUT_SMT,
        path=log_path,
    )
    print(format_prior_results_summary(log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
