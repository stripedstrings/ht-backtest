"""Multi-strategy comparison tables (train reach vs random walk).

Holdout is never scored here unless the caller explicitly passes a holdout
frame after a promotion artifact exists.
"""

from __future__ import annotations

import pandas as pd

from ht_backtest.reports.reach import reach_vs_random_walk
from ht_backtest.trades.forward_tracker import DEFAULT_TARGETS


def strategy_comparison_table(
    trades_by_strategy: dict[str, pd.DataFrame],
    *,
    split: str = "train",
    targets: tuple[float, ...] = DEFAULT_TARGETS,
    min_edge_pp: float = 5.0,
    min_n: int = 200,
) -> pd.DataFrame:
    """One row per (strategy_id, target_R) with reach% vs 1/(1+T)."""
    rows: list[dict] = []
    for strategy_id, trades in trades_by_strategy.items():
        if trades.empty:
            continue
        subset = trades[trades["split"] == split] if "split" in trades.columns else trades
        meta_desc = ""
        version = ""
        param_hash = ""
        if len(subset):
            if "strategy_description" in subset.columns:
                meta_desc = str(subset["strategy_description"].iloc[0])
            if "strategy_version" in subset.columns:
                version = str(subset["strategy_version"].iloc[0])
            if "strategy_parameter_hash" in subset.columns:
                param_hash = str(subset["strategy_parameter_hash"].iloc[0])
        table = reach_vs_random_walk(subset, targets=targets)
        promo = (table["diff_pp"] > min_edge_pp) & (table["n"] >= min_n)
        for i, row in table.iterrows():
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_version": version,
                    "strategy_parameter_hash": param_hash,
                    "description": meta_desc,
                    "split": split,
                    "target_R": float(row["target_R"]),
                    "n": int(row["n"]),
                    "reach_pct": float(row["reach_pct"]),
                    "random_walk_pct": float(row["random_walk_pct"]),
                    "diff_pp": float(row["diff_pp"]),
                    "promoted": bool(promo.iloc[i]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["strategy_id", "target_R"]).reset_index(drop=True)


def format_comparison_table(table: pd.DataFrame, title: str = "STRATEGY COMPARISON vs RW (train)") -> str:
    if table.empty:
        return title + "\n(no rows)"
    lines = [title]
    header = (
        f"{'strategy':<28} {'tgt':>5} {'n':>6} {'reach%':>7} {'RW%':>6} {'diff':>7} {'promo':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in table.iterrows():
        lines.append(
            f"{str(row['strategy_id'])[:28]:<28} {row['target_R']:>4.1f}R {int(row['n']):>6d} "
            f"{row['reach_pct']:>6.1f}% {row['random_walk_pct']:>5.1f}% "
            f"{row['diff_pp']:>+6.1f} {'YES' if row['promoted'] else 'no':>5}"
        )
    return "\n".join(lines)
