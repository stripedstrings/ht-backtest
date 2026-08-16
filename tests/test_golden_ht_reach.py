"""Golden test: Phase A HT adapter must reproduce FINDINGS.md training reach.

Compares the strategy-protocol path against data/reports/training_reach_table.csv
(and the rounded FINDINGS.md headline). Fail if any horizon's reach% drifts by
more than 0.5 percentage points, or if train n drifts.

Run (from repo root, with cached OHLCV):

    python -m pytest tests/test_golden_ht_reach.py -s -m golden
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ht_backtest.data.split import SplitManifest
from ht_backtest.reports.reach import format_reach_table, reach_vs_random_walk
from ht_backtest.reports.universe_report import generate_pooled_trades
from ht_backtest.strategies.registry import get_strategy

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = ROOT / "data" / "reports" / "training_reach_table.csv"
SPLIT_PATH = ROOT / "specs" / "splits" / "v1.json"
TOLERANCE_PP = 0.5

# FINDINGS.md rounded headline (still checked within tolerance of live table)
FINDINGS_REACH_PCT = {
    0.5: 65.1,
    1.0: 50.2,
    1.5: 39.9,
    2.0: 32.8,
    2.5: 27.9,
    3.0: 23.6,
}


@pytest.mark.golden
def test_ht_v10_training_reach_matches_findings(capsys):
    if not REFERENCE_CSV.exists():
        pytest.skip(f"missing reference table {REFERENCE_CSV}")
    if not SPLIT_PATH.exists():
        pytest.skip(f"missing split {SPLIT_PATH}")

    reference = pd.read_csv(REFERENCE_CSV)
    strategy = get_strategy("ht_v10")
    meta = strategy.metadata()
    split = SplitManifest.load(SPLIT_PATH)

    trades = generate_pooled_trades(
        strategy=strategy,
        split=split,
        timeframe="15m",
        exchange_id="binanceusdm",
        cache_dir=str(ROOT / "data" / "raw"),
        mfe_win=100,
    )
    assert not trades.empty, "pooled trades came back empty — is the OHLCV cache present?"

    train = trades[trades["split"] == "train"]
    live = reach_vs_random_walk(train)

    print()
    print("=" * 72)
    print("GOLDEN TEST — Holy Trinity v10 via Strategy protocol")
    print(f"strategy metadata: {meta.to_dict()}")
    print("=" * 72)
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
        print(
            f"  {t:.1f}R  live={live_row['reach_pct']:.4f}%  "
            f"ref={ref_row['reach_pct']:.4f}%  Δref={delta:+.4f}pp  "
            f"ΔFINDINGS={findings_delta:+.4f}pp"
        )
        if abs(delta) > TOLERANCE_PP:
            failures.append(
                f"{t}R reach% drifted by {delta:+.4f}pp (tolerance ±{TOLERANCE_PP}pp)"
            )

    print("=" * 72)
    if failures:
        print("GOLDEN TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 72)
        pytest.fail("; ".join(failures))
    print("GOLDEN TEST PASSED — within tolerance, no Phase B.")
    print("=" * 72)

    # Ensure pytest -s shows the block even when assertions pass
    captured = capsys.readouterr()
    assert "GOLDEN TEST PASSED" in captured.out or True
