"""Open-interest conditions (orthogonal to OHLCV-only features).

Requires ``bars['open_interest']`` from ``attach_open_interest``. Undefined
(NaN) → None. Rising/falling compare vs the previous bar's as-of OI.
``oi_extreme`` uses trailing p10/p90 of **prior** bars only — the current
bar's OI is the test value, never a member of the percentile window.
"""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool
from ht_backtest.data.oi import oi_extreme_mask


class OiRising:
    id = "oi_rising"
    category = "oi"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "open_interest" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        oi = bars["open_interest"].astype(float)
        prior = oi.shift(1)
        return as_object_bool(oi > prior, oi.notna() & prior.notna())


class OiFalling:
    id = "oi_falling"
    category = "oi"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "open_interest" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        oi = bars["open_interest"].astype(float)
        prior = oi.shift(1)
        return as_object_bool(oi < prior, oi.notna() & prior.notna())


class OiExtreme:
    id = "oi_extreme"
    category = "oi"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "open_interest" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        oi = bars["open_interest"].astype(float)
        extreme, defined = oi_extreme_mask(oi)
        return as_object_bool(extreme, defined)


OI_CONDITIONS = (
    OiRising(),
    OiFalling(),
    OiExtreme(),
)
