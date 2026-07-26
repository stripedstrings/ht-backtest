"""Reach-vs-random-walk evidence tables. The headline number at every target
horizon T is the random-walk baseline 1/(1+T) -- printed as a literal column,
never left for the reader to eyeball -- against the actual reach rate. A
setup or tag only has evidence of an edge if it beats this baseline by the
predeclared margin, with enough trades, on data the rule wasn't chosen from.
"""

from __future__ import annotations

import pandas as pd

from ht_backtest.trades.forward_tracker import DEFAULT_TARGETS


def reach_vs_random_walk(
    trades_df: pd.DataFrame,
    targets: tuple[float, ...] = DEFAULT_TARGETS,
    date_span_days: float | None = None,
) -> pd.DataFrame:
    """One row per target horizon: n (trades with a known outcome at that
    horizon), reach rate, the random-walk baseline 1/(1+T), and the
    difference in percentage points. Trades with an unknown outcome at a
    given horizon (insufficient forward data) are excluded from that
    horizon's n, not counted as misses."""
    rows = []
    for T in targets:
        col = f"reach_{T}R"
        known = trades_df[col].notna()
        n = int(known.sum())
        hits = int(trades_df.loc[known, col].sum()) if n else 0
        reach_pct = 100.0 * hits / n if n else float("nan")
        rw_pct = 100.0 / (1.0 + T)
        rows.append(
            {
                "target_R": T,
                "n": n,
                "reach_pct": reach_pct,
                "random_walk_pct": rw_pct,
                "diff_pp": reach_pct - rw_pct if n else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if date_span_days and date_span_days > 0:
        out.attrs["trades_per_month"] = len(trades_df) * 30.44 / date_span_days
    out.attrs["n_trades"] = len(trades_df)
    return out


def tag_on_off_tables(
    trades_df: pd.DataFrame,
    tag_col: str,
    on_value,
    off_value,
    exclude_values: tuple = (),
    targets: tuple[float, ...] = DEFAULT_TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits trades_df into ON/OFF by tag_col and returns (on_table,
    off_table, comparison) where comparison adds delta-reach and a
    promotion flag per horizon. Trades in exclude_values (e.g. -1 =
    undefined median tag) are dropped from BOTH sides, matching Pine's
    f_filt/f_replay: `tagOn>=0` gates whether a trade counts at all."""
    on_df = trades_df[trades_df[tag_col] == on_value]
    off_df = trades_df[trades_df[tag_col] == off_value]
    on_table = reach_vs_random_walk(on_df, targets=targets)
    off_table = reach_vs_random_walk(off_df, targets=targets)
    comparison = on_table[["target_R"]].copy()
    comparison["n_on"] = on_table["n"]
    comparison["reach_on_pct"] = on_table["reach_pct"]
    comparison["n_off"] = off_table["n"]
    comparison["reach_off_pct"] = off_table["reach_pct"]
    comparison["random_walk_pct"] = on_table["random_walk_pct"]
    comparison["edge_on_pp"] = on_table["diff_pp"]
    comparison["delta_reach_pp"] = on_table["reach_pct"] - off_table["reach_pct"]
    return on_table, off_table, comparison


def promotion_flags(comparison: pd.DataFrame, min_edge_pp: float, min_n: int) -> pd.Series:
    """Pine's promotion rule: ON side must beat random-walk by min_edge_pp,
    with at least min_n trades on the ON side."""
    return (comparison["edge_on_pp"] > min_edge_pp) & (comparison["n_on"] >= min_n)


def format_tag_comparison(comparison: pd.DataFrame, title: str, min_edge_pp: float, min_n: int) -> str:
    flags = promotion_flags(comparison, min_edge_pp, min_n)
    lines = [title, f"(promotion bar: ON edge > {min_edge_pp}pp at n_on >= {min_n})"]
    header = f"{'target':>8} {'n_on':>6} {'ON%':>7} {'n_off':>6} {'OFF%':>7} {'RW%':>7} {'ON edge':>9} {'ΔReach':>8}  promoted"
    lines.append(header)
    lines.append("-" * len(header))
    for i, row in comparison.iterrows():
        lines.append(
            f"{row['target_R']:>6.1f}R {int(row['n_on']):>6d} {row['reach_on_pct']:>6.1f}% "
            f"{int(row['n_off']):>6d} {row['reach_off_pct']:>6.1f}% {row['random_walk_pct']:>6.1f}% "
            f"{row['edge_on_pp']:>+8.1f} {row['delta_reach_pp']:>+7.1f}  {'YES' if flags.iloc[i] else 'no'}"
        )
    return "\n".join(lines)


def format_reach_table(table: pd.DataFrame, title: str) -> str:
    lines = [title]
    header = f"{'target':>8} {'n':>6} {'reach%':>8} {'1/(1+T)%':>10} {'diff(pp)':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in table.iterrows():
        lines.append(
            f"{row['target_R']:>6.1f}R {int(row['n']):>6d} {row['reach_pct']:>7.1f}% "
            f"{row['random_walk_pct']:>9.1f}% {row['diff_pp']:>+8.1f}"
        )
    if "trades_per_month" in table.attrs:
        lines.append(f"\ntrades/month: {table.attrs['trades_per_month']:.1f}  (n={table.attrs['n_trades']})")
    else:
        lines.append(f"\nn={table.attrs.get('n_trades', len(table))}")
    return "\n".join(lines)
