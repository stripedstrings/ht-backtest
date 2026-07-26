"""Median-based tags and final trade-row assembly.

Pine holds SIX running arrays (wick, body, coil, age, efficiency-ratio,
range-width), shared between long and short trades on one symbol, capped at
the most recent 300 trades, with an 8-trade warm-up before a tag can fire
(`f_medTag`): -1 undefined, 1 = at-or-above the median of everything seen so
far, 0 = below it. A trade is compared against the median BEFORE its own
value is pushed in, so it never taints its own tag.

Per the user's confirmed judgment call, the arrays are pushed ONLY from
trades classified "train" by the split manifest (per-symbol, chronological,
no leakage from holdout dates) -- so a train symbol's holdout-dated trades
still get a tag (compared against the frozen train-derived median), while a
symbol that is ENTIRELY held out never accumulates any history of its own
and all six of its median-based tags come out -1 (undefined) throughout.
That is a real consequence of "per-symbol", not a bug -- surfaced in the
tag-coverage report rather than hidden.

Two of the six are used INVERTED from the raw median comparison: tight coil
and range-before-grab both mean "BELOW the median", not above it.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from ht_backtest.data.split import SplitManifest

_RAW_COLUMNS = {
    "wick": "wick_atr",
    "body": "body_atr",
    "coil": "coil_atr",
    "age": "grab_age_bars",
    "er": "efficiency_ratio",
    "rngwidth": "range_width_atr",
}


def _med_tag(arr: deque, v: float, warm_n: int) -> int:
    if v is None or (isinstance(v, float) and np.isnan(v)) or len(arr) < warm_n:
        return -1
    return 1 if v >= np.median(arr) else 0


def apply_median_tags(
    trades_df: pd.DataFrame, split: SplitManifest, symbol: str, warm_n: int = 8, window: int = 300
) -> pd.DataFrame:
    if trades_df.empty:
        for col in ["big_wick", "big_displacement", "tight_coil", "old_liquidity", "range_before_grab", "wide_range"]:
            trades_df[col] = pd.Series(dtype=int)
        return trades_df

    trades_df = trades_df.sort_values("entry_bar").reset_index(drop=True)
    arrays = {k: deque(maxlen=window) for k in _RAW_COLUMNS}

    big_wick, big_disp, tight_coil, old_liq, range_before_grab, wide_range = [], [], [], [], [], []

    for _, row in trades_df.iterrows():
        wk = _med_tag(arrays["wick"], row["wick_atr"], warm_n)
        bd = _med_tag(arrays["body"], row["body_atr"], warm_n)
        cl_raw = _med_tag(arrays["coil"], row["coil_atr"], warm_n)
        ag = _med_tag(arrays["age"], row["grab_age_bars"], warm_n)
        er_raw = _med_tag(arrays["er"], row["efficiency_ratio"], warm_n)
        rw = _med_tag(arrays["rngwidth"], row["range_width_atr"], warm_n)

        big_wick.append(wk)
        big_disp.append(bd)
        old_liq.append(ag)
        wide_range.append(rw)
        # inverted: tight coil / ranging-before-grab both mean BELOW median
        tight_coil.append(0 if cl_raw == 1 else (1 if cl_raw == 0 else -1))
        range_before_grab.append(0 if er_raw == 1 else (1 if er_raw == 0 else -1))

        if split.classify(symbol, int(row["entry_time"])) == "train":
            for key, col in _RAW_COLUMNS.items():
                v = row[col]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    arrays[key].append(v)

    trades_df["big_wick"] = big_wick
    trades_df["big_displacement"] = big_disp
    trades_df["tight_coil"] = tight_coil
    trades_df["old_liquidity"] = old_liq
    trades_df["range_before_grab"] = range_before_grab
    trades_df["wide_range"] = wide_range
    return trades_df


def assemble_trade_frame(
    trades: list[dict], symbol: str, timeframe: str, split: SplitManifest, warm_n: int = 8, window: int = 300
) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df.insert(0, "symbol", symbol)
    df.insert(1, "timeframe", timeframe)
    is_short = df["direction"] == "short"
    df["session_range_high"] = np.where(is_short, df["grab_price"], df["session_edge_target"])
    df["session_range_low"] = np.where(is_short, df["session_edge_target"], df["grab_price"])
    df["split"] = [split.classify(symbol, int(t)) for t in df["entry_time"]]
    df = apply_median_tags(df, split, symbol, warm_n=warm_n, window=window)
    return df
