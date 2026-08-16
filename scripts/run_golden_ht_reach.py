"""Standalone Phase A golden check (avoids pytest pandas.testing DLL issues on some Windows policies).

Usage (repo root):
    .venv\\Scripts\\python.exe scripts/run_golden_ht_reach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.data.split import SplitManifest  # noqa: E402
from ht_backtest.reports.reach import format_reach_table, reach_vs_random_walk  # noqa: E402
from ht_backtest.reports.universe_report import generate_pooled_trades  # noqa: E402
from ht_backtest.strategies.registry import get_strategy  # noqa: E402

REFERENCE_CSV = ROOT / "data" / "reports" / "training_reach_table.csv"
SPLIT_PATH = ROOT / "specs" / "splits" / "v1.json"
TOLERANCE_PP = 0.5

FINDINGS_REACH_PCT = {
    0.5: 65.1,
    1.0: 50.2,
    1.5: 39.9,
    2.0: 32.8,
    2.5: 27.9,
    3.0: 23.6,
}


def main() -> int:
    if not REFERENCE_CSV.exists():
        print(f"MISSING reference table: {REFERENCE_CSV}", file=sys.stderr)
        return 2
    if not SPLIT_PATH.exists():
        print(f"MISSING split: {SPLIT_PATH}", file=sys.stderr)
        return 2

    reference = pd.read_csv(REFERENCE_CSV)
    strategy = get_strategy("ht_v10")
    meta = strategy.metadata()
    split = SplitManifest.load(SPLIT_PATH)

    print("=" * 72)
    print("GOLDEN TEST - Holy Trinity v10 via Strategy protocol")
    print(f"strategy metadata: {meta.to_dict()}")
    print("=" * 72)

    trades = generate_pooled_trades(
        strategy=strategy,
        split=split,
        timeframe="15m",
        exchange_id="binanceusdm",
        cache_dir=str(ROOT / "data" / "raw"),
        mfe_win=100,
    )
    if isinstance(trades, tuple):
        trades, _timings = trades
    if trades.empty:
        print("FAIL: pooled trades empty — OHLCV cache missing?", file=sys.stderr)
        return 2

    train = trades[trades["split"] == "train"]
    live = reach_vs_random_walk(train)

    print()
    print(format_reach_table(live, "LIVE train reach (protocol path)"))
    print()
    print(format_reach_table(reference, "REFERENCE train reach (FINDINGS / training_reach_table.csv)"))
    print()

    ref_n = int(reference["n"].iloc[0])
    live_n = int(live["n"].iloc[0])
    print(f"train n: live={live_n}  reference={ref_n}  delta={live_n - ref_n}")

    failures: list[str] = []
    if live_n != ref_n:
        failures.append(f"train n drifted: live={live_n} reference={ref_n}")

    for _, ref_row in reference.iterrows():
        t = float(ref_row["target_R"])
        live_row = live[live["target_R"] == t].iloc[0]
        delta = float(live_row["reach_pct"]) - float(ref_row["reach_pct"])
        findings_delta = float(live_row["reach_pct"]) - FINDINGS_REACH_PCT[t]
        line = (
            f"  {t:.1f}R  live={live_row['reach_pct']:.4f}%  "
            f"ref={ref_row['reach_pct']:.4f}%  dRef={delta:+.4f}pp  "
            f"dFINDINGS={findings_delta:+.4f}pp"
        )
        print(line)
        if abs(delta) > TOLERANCE_PP:
            failures.append(f"{t}R reach% drifted by {delta:+.4f}pp (tolerance +/-{TOLERANCE_PP}pp)")

    print("=" * 72)
    if failures:
        print("GOLDEN TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 72)
        return 1

    print("GOLDEN TEST PASSED - within tolerance. Phase A clean; do not start Phase B yet.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
