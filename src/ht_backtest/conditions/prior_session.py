"""Prior session outcome conditions."""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool


class LondonRaidedHigh:
    id = "london_raided_high"
    category = "prior_session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        col = "feat_london_raided_high"
        if col not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        s = bars[col]
        return as_object_bool(s == 1.0, s.notna())


class LondonRaidedLow:
    id = "london_raided_low"
    category = "prior_session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        col = "feat_london_raided_low"
        if col not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        s = bars[col]
        return as_object_bool(s == 1.0, s.notna())


class PriorSessionSameDirection:
    id = "prior_session_same_direction"
    category = "prior_session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        col = "feat_prior_session_same"
        if col not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        s = bars[col]
        return as_object_bool(s == 1.0, s.notna())


class PriorSessionOppositeDirection:
    id = "prior_session_opposite_direction"
    category = "prior_session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        col = "feat_prior_session_opp"
        if col not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        s = bars[col]
        return as_object_bool(s == 1.0, s.notna())


PRIOR_SESSION_CONDITIONS = (
    LondonRaidedHigh(),
    LondonRaidedLow(),
    PriorSessionSameDirection(),
    PriorSessionOppositeDirection(),
)
