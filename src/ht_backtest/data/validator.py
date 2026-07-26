"""OHLCV validation. Flags problems, never silently drops or fixes rows —
per the project's data cleaning policy, exclusion/winsorizing is a separate,
explicit downstream step so a training run can be repeated with and without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ht_backtest.data.downloader import TIMEFRAME_MS


@dataclass
class ValidationReport:
    symbol: str
    timeframe: str
    total_bars: int
    duplicates: pd.DataFrame
    gaps: pd.DataFrame
    impossible_ohlc: pd.DataFrame
    non_positive_price: pd.DataFrame
    extreme_jumps: pd.DataFrame
    outlier_wicks: pd.DataFrame
    dead_runs: pd.DataFrame
    flagged_bar_count: int = field(init=False)

    def __post_init__(self) -> None:
        flagged = set()
        for df, col in [
            (self.impossible_ohlc, "timestamp"),
            (self.non_positive_price, "timestamp"),
            (self.extreme_jumps, "timestamp"),
            (self.outlier_wicks, "timestamp"),
        ]:
            flagged.update(df[col].tolist())
        self.flagged_bar_count = len(flagged)

    def missing_bar_count(self) -> int:
        return int(self.gaps["missing_bars"].sum()) if not self.gaps.empty else 0

    def dead_bar_count(self) -> int:
        return int(self.dead_runs["bars"].sum()) if not self.dead_runs.empty else 0

    def summary(self) -> str:
        lines = [
            f"Validation report: {self.symbol} {self.timeframe}",
            f"  total bars cached          : {self.total_bars}",
            f"  duplicate timestamps       : {len(self.duplicates)}",
            f"  gaps (runs of missing bars): {len(self.gaps)}  ({self.missing_bar_count()} bars missing)",
            f"  impossible OHLC rows       : {len(self.impossible_ohlc)}",
            f"  zero/negative price rows   : {len(self.non_positive_price)}",
            f"  dead/zero-volume runs      : {len(self.dead_runs)}  ({self.dead_bar_count()} bars)",
            f"  extreme jump bars          : {len(self.extreme_jumps)}",
            f"  isolated outlier wick bars : {len(self.outlier_wicks)}",
            f"  distinct bars flagged      : {self.flagged_bar_count} / {self.total_bars} "
            f"({100 * self.flagged_bar_count / self.total_bars:.3f}%)" if self.total_bars else "  distinct bars flagged      : 0",
        ]
        return "\n".join(lines)


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    jump_zscore: float = 8.0,
    jump_min_abs_move: float = 0.03,
    jump_window: int = 100,
    wick_atr_mult: float = 6.0,
    wick_atr_window: int = 14,
    dead_run_min_bars: int = 4,
) -> ValidationReport:
    """df must have columns timestamp(ms int64 UTC), open, high, low, close, volume,
    sorted ascending by timestamp. Returns a report; does not mutate df."""
    tf_ms = TIMEFRAME_MS[timeframe]
    d = df.sort_values("timestamp").reset_index(drop=True).copy()

    # 1. duplicates
    dup_mask = d.duplicated(subset="timestamp", keep=False)
    duplicates = d.loc[dup_mask]

    d = d.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    # 2. gaps: runs of missing expected bars between consecutive cached timestamps
    ts = d["timestamp"].to_numpy()
    gaps_rows = []
    if len(ts) > 1:
        deltas = np.diff(ts)
        gap_idx = np.where(deltas > tf_ms)[0]
        for i in gap_idx:
            missing = int(deltas[i] / tf_ms) - 1
            gaps_rows.append(
                {
                    "gap_start": int(ts[i]),
                    "gap_end": int(ts[i + 1]),
                    "missing_bars": missing,
                }
            )
    gaps = pd.DataFrame(gaps_rows, columns=["gap_start", "gap_end", "missing_bars"])

    # 3. impossible OHLC relationships
    hi, lo, op, cl = d["high"], d["low"], d["open"], d["close"]
    impossible_mask = (
        (hi < lo)
        | (hi < op)
        | (hi < cl)
        | (lo > op)
        | (lo > cl)
    )
    impossible_ohlc = d.loc[impossible_mask]

    # 4. zero/negative prices or negative volume
    nonpos_mask = (op <= 0) | (hi <= 0) | (lo <= 0) | (cl <= 0) | (d["volume"] < 0)
    non_positive_price = d.loc[nonpos_mask]

    # 5. extreme jumps: robust z-score of log return over a trailing window,
    # plus an absolute-move floor so a flat/low-vol window doesn't manufacture
    # spurious flags from a tiny, meaningless move.
    log_ret = np.log(cl / cl.shift(1))
    roll_std = log_ret.rolling(jump_window, min_periods=20).std()
    z = log_ret / roll_std.replace(0, np.nan)
    jump_mask = (z.abs() > jump_zscore) & (log_ret.abs() > jump_min_abs_move)
    jump_mask = jump_mask.fillna(False)
    extreme_jumps = d.loc[jump_mask].assign(log_return=log_ret[jump_mask], zscore=z[jump_mask])

    # 6. isolated outlier wicks: a wick many multiples of local ATR, on an
    # extreme that neither the previous nor next bar comes close to touching
    # (i.e. one print, not corroborated -- a classic bad-tick signature).
    tr = pd.concat(
        [hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(wick_atr_window, min_periods=wick_atr_window).mean()
    wick_up = hi - pd.concat([op, cl], axis=1).max(axis=1)
    wick_dn = pd.concat([op, cl], axis=1).min(axis=1) - lo
    prev_hi, next_hi = hi.shift(1), hi.shift(-1)
    prev_lo, next_lo = lo.shift(1), lo.shift(-1)
    atr_safe = atr.replace(0, np.nan)

    up_spike = (
        (wick_up > wick_atr_mult * atr_safe)
        & (hi - prev_hi.combine(next_hi, max) > 0.5 * wick_up)
    )
    dn_spike = (
        (wick_dn > wick_atr_mult * atr_safe)
        & (prev_lo.combine(next_lo, min) - lo > 0.5 * wick_dn)
    )
    wick_mask = (up_spike | dn_spike).fillna(False)
    outlier_wicks = d.loc[wick_mask].assign(
        wick_up=wick_up[wick_mask], wick_dn=wick_dn[wick_mask], atr=atr[wick_mask]
    )

    # 7. dead/zero-volume runs: a market that has stopped trading prints a flat
    # close with zero volume forever (e.g. a delisted/paused perpetual) rather
    # than gapping. Flag runs of consecutive zero-volume bars, not single ones,
    # since a single quiet 15m bar on a real market is unremarkable.
    is_dead = d["volume"] <= 0
    dead_run_rows = []
    if is_dead.any():
        run_id = (is_dead != is_dead.shift(fill_value=False)).cumsum()
        for _, grp in d.assign(_dead=is_dead, _run=run_id)[is_dead].groupby("_run"):
            if len(grp) >= dead_run_min_bars:
                dead_run_rows.append(
                    {
                        "run_start": int(grp["timestamp"].iloc[0]),
                        "run_end": int(grp["timestamp"].iloc[-1]),
                        "bars": len(grp),
                    }
                )
    dead_runs = pd.DataFrame(dead_run_rows, columns=["run_start", "run_end", "bars"])

    return ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        total_bars=len(d),
        duplicates=duplicates,
        gaps=gaps,
        impossible_ohlc=impossible_ohlc,
        non_positive_price=non_positive_price,
        extreme_jumps=extreme_jumps,
        outlier_wicks=outlier_wicks,
        dead_runs=dead_runs,
    )
