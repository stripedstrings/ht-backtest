"""Depth-2 grid: None-check → synthetic injection → real grid → print survivors."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.grid.engine import (  # noqa: E402
    GRID_EXCLUDE_IDS,
    build_trade_matrix,
    combo_mask,
    combo_n,
    grid_condition_ids,
    inject_synthetic_70pct,
    load_survivors,
    run_grid,
    score_combo,
)

DB = ROOT / "data" / "grid" / "condition_grid.sqlite"
DB_SYN = ROOT / "data" / "grid" / "condition_grid_synthetic.sqlite"


def verify_none_handling(matrix) -> None:
    col = "london_raided_high"
    vals = matrix.trades[col].to_numpy()
    n_true = int(sum(isinstance(v, (bool, np.bool_)) and bool(v) for v in vals))
    n_false = int(sum(isinstance(v, (bool, np.bool_)) and not bool(v) for v in vals))
    n_none = int(sum(v is None for v in vals))
    n_defined = n_true + n_false
    print("=" * 72)
    print("None-handling check:", col)
    print(f"  True={n_true}  False={n_false}  None={n_none}  defined={n_defined}")
    assert n_defined + n_none == len(matrix.trades)

    # Effective n for True-only filter (what grid AND uses)
    n_and_true = combo_n(matrix, [col])
    print(f"  combo AND True-only n={n_and_true} (must equal True count, not defined)")
    assert n_and_true == n_true

    # Pair that should be non-empty: NY entries after London raided high
    partner = "ny_session"
    mask = combo_mask(matrix, [col, partner])
    for i in np.flatnonzero(mask):
        assert vals[i] is True or vals[i] == True  # noqa: E712
    n_pair = int(mask.sum())
    print(f"  test combo {{{col}, {partner}}} effective n={n_pair}")
    print(f"  defined≈11229 check: defined={n_defined} (coverage non-None)")
    if abs(n_defined - 11229) > 50:
        print(f"  WARN: defined count {n_defined} differs from expected ~11229")
    else:
        print(f"  OK: defined count matches coverage non-None (~11229)")
    if n_pair == 0:
        print("  WARN: expected non-zero n for london_raided_high ∧ ny_session")
    else:
        print(f"  OK: None-excluded AND True mask works (n={n_pair})")
    print("=" * 72)


def run_synthetic_test(base_matrix) -> None:
    print("\n>>> SYNTHETIC SIGNAL INJECTION TEST")
    syn = inject_synthetic_70pct(base_matrix)
    # Confirm injected set has ~70% reach
    m = combo_mask(syn, ["synthetic_edge_70"])
    sub = syn.trades.loc[m]
    known = sub["reach_1.0R"].notna()
    rate = float(sub.loc[known, "reach_1.0R"].mean()) if known.any() else float("nan")
    print(f"  synthetic alone: n={int(m.sum())} reach_1R={rate:.3f}")

    summary = run_grid(
        syn,
        depth=2,
        db_path=DB_SYN,
        run_id="synthetic_injection",
        notes="synthetic_edge_70 injection",
    )
    surv = load_survivors(DB_SYN, "synthetic_injection")
    syn_rows = surv[surv["signature"].str.contains("synthetic_edge_70")]
    print(f"  synthetic pairs in grid_survivors: {len(syn_rows)}")

    # Every (synthetic, other) with n>=200 should survive
    conn = sqlite3.connect(str(DB_SYN))
    tested = pd.read_sql_query(
        "SELECT signature, n, diff_pp_1r, status FROM grid_results "
        "WHERE run_id='synthetic_injection' AND signature LIKE '%synthetic_edge_70%' "
        "AND status='tested'",
        conn,
    )
    conn.close()
    missing = []
    for _, r in tested.iterrows():
        if r["n"] < 200:
            continue
        if r["signature"] not in set(syn_rows["signature"]):
            missing.append((r["signature"], r["n"], r["diff_pp_1r"]))
    if missing:
        print(f"  FAIL: {len(missing)} synthetic combos with n>=200 missing from survivors:")
        for s, n, d in missing[:10]:
            print(f"    {s} n={n} diff={d}")
        raise SystemExit(1)
    print(f"  PASS: all {len(tested)} tested synthetic pairs with n>=200 are in grid_survivors")

    # Remove synthetic — baseline real grid ids only
    print("  Removing synthetic; survivors must return toward baseline (empty or non-synthetic)")
    baseline = run_grid(
        base_matrix,
        depth=2,
        db_path=ROOT / "data" / "grid" / "condition_grid_baseline_check.sqlite",
        run_id="baseline_after_synth",
        notes="post-synthetic baseline",
    )
    base_surv = load_survivors(
        ROOT / "data" / "grid" / "condition_grid_baseline_check.sqlite",
        "baseline_after_synth",
    )
    if base_surv["signature"].astype(str).str.contains("synthetic").any():
        print("  FAIL: synthetic still in baseline survivors")
        raise SystemExit(1)
    print(
        f"  PASS: baseline survivors have no synthetic "
        f"(n_survivors={len(base_surv)}, tested={baseline['tested']})"
    )


def main() -> int:
    print("Building trade matrix (enrich + conditions)...")
    print(f"Grid excludes: {sorted(GRID_EXCLUDE_IDS)}")
    print(f"Grid condition ids ({len(grid_condition_ids())}): {grid_condition_ids()}")

    matrix = build_trade_matrix(
        split_path=ROOT / "specs" / "splits" / "v1.json",
        cache_dir=ROOT / "data" / "raw",
        funding_dir=ROOT / "data" / "funding",
        workers=4,
    )
    print(f"Base train trades: {matrix.n_base}")
    verify_none_handling(matrix)

    run_synthetic_test(matrix)

    print("\n>>> REAL DEPTH-2 GRID")
    summary = run_grid(
        matrix,
        depth=2,
        db_path=DB,
        run_id="depth2_real",
        notes="asia_session excluded; london_session↔london_raided_* mutex",
    )

    print("\n" + "=" * 72)
    print("DEPTH-2 SUMMARY")
    print("=" * 72)
    print(f"total combos generated:     {summary['generated']}")
    print(f"combos skipped by mutex:    {summary['mutex_skipped']}")
    print(f"combos killed by n<200:     {summary['low_n']}")
    print(f"combos tested:              {summary['tested']}")
    print(f"BH survivors (table rows):  {summary['bh_survivors']}")
    print(f"BY survivors (diff>+3):     {summary['by_survivors']}")
    print(f"BY significant (any diff):  {summary['by_significant']}")
    print(f"started: {summary['started']}  finished: {summary['finished']}")

    surv = load_survivors(DB, "depth2_real")
    print("\n" + "=" * 72)
    print("grid_survivors (BH significant AND diff_pp_1R > +3) sorted by diff_pp DESC")
    print("=" * 72)
    if surv.empty:
        print("(empty — no survivors)")
    else:
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.width", 160)
        pd.set_option("display.max_colwidth", 80)
        print(
            surv[
                ["signature", "c1", "c2", "n", "reach_1r_pct", "diff_pp_1r", "p_value_1r", "by_significant"]
            ].to_string(index=False)
        )
    print("\nStop: depth-3 not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
