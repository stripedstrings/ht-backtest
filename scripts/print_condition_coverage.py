"""Print condition True-rate coverage on kz_first_raid_reclaim train trades (n=20382)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.conditions.features import enrich_condition_features  # noqa: E402
from ht_backtest.conditions.registry import ALL_CONDITIONS  # noqa: E402
from ht_backtest.data.downloader import OHLCVDownloader  # noqa: E402
from ht_backtest.data.split import SplitManifest  # noqa: E402
from ht_backtest.reports.universe_report import generate_pooled_trades  # noqa: E402
from ht_backtest.strategies.registry import get_strategy  # noqa: E402

EXPECTED_N = 20_382


def main() -> int:
    split = SplitManifest.load(ROOT / "specs" / "splits" / "v1.json")
    trades, _ = generate_pooled_trades(
        get_strategy("kz_first_raid_reclaim"),
        split,
        timeframe="15m",
        workers=4,
        strategy_name="kz_first_raid_reclaim",
        split_path=str(ROOT / "specs" / "splits" / "v1.json"),
        attach_funding=True,
        log_fn=lambda *a, **k: None,
    )
    train = trades[trades["split"] == "train"][["symbol", "entry_bar", "entry_time"]].copy()
    if len(train) != EXPECTED_N:
        print(f"WARN: train n={len(train)} expected {EXPECTED_N}")

    dl = OHLCVDownloader(cache_dir=str(ROOT / "data" / "raw"))
    rows = []
    for sym, g in train.groupby("symbol"):
        bars = dl.cached_range(sym, "15m", 0, 4_102_444_800_000)
        if bars.empty:
            continue
        enriched = enrich_condition_features(
            bars,
            sym,
            cache_dir=str(ROOT / "data" / "raw"),
            funding_dir=str(ROOT / "data" / "funding"),
        )
        # entry_bar is positional index into symbol bars
        idx = g["entry_bar"].astype(int).to_numpy()
        # guard
        idx = idx[(idx >= 0) & (idx < len(enriched))]
        slice_df = enriched.iloc[idx]
        for cond in ALL_CONDITIONS:
            s = cond.eval(slice_df)
            n_true = int(sum(v is True for v in s))
            n_false = int(sum(v is False for v in s))
            n_none = int(sum(v is None for v in s))
            rows.append(
                {
                    "symbol": sym,
                    "condition": cond.id,
                    "category": cond.category,
                    "n_true": n_true,
                    "n_false": n_false,
                    "n_none": n_none,
                    "n": len(s),
                }
            )

    detail = pd.DataFrame(rows)
    agg = (
        detail.groupby(["condition", "category"], as_index=False)[["n_true", "n_false", "n_none", "n"]]
        .sum()
    )
    agg["pct_true"] = 100.0 * agg["n_true"] / agg["n"]
    agg["pct_defined_true"] = np.where(
        (agg["n_true"] + agg["n_false"]) > 0,
        100.0 * agg["n_true"] / (agg["n_true"] + agg["n_false"]),
        np.nan,
    )
    agg = agg.sort_values(["category", "condition"])

    print("=" * 88)
    print(f"Condition coverage on base trades n={int(agg['n'].iloc[0]) if len(agg) else 0}")
    print("=" * 88)
    print(f"{'condition':<36} {'cat':<14} {'%True':>8} {'n_true':>8} {'n_none':>8}  flag")
    print("-" * 88)
    flags = []
    for _, r in agg.iterrows():
        flag = ""
        if r["pct_true"] < 5:
            flag = "LOW (<5%)"
            flags.append(r["condition"])
        elif r["pct_true"] > 95:
            flag = "HIGH (>95%)"
            flags.append(r["condition"])
        print(
            f"{r['condition']:<36} {r['category']:<14} {r['pct_true']:7.2f}% "
            f"{int(r['n_true']):8d} {int(r['n_none']):8d}  {flag}"
        )
    print("-" * 88)
    if flags:
        print("Near-constant / low-utility:", ", ".join(flags))
    else:
        print("No condition outside 5–95% True band.")
    out = ROOT / "data" / "reports" / "condition_coverage_base.csv"
    # data/reports may be gitignored partially — still write
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
