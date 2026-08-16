"""Funding-rate conditions (orthogonal to OHLCV-only features).

Requires ``bars['funding_rate']`` from ``attach_funding_rate`` (Binance decimal:
0.0001 == 0.01%). Undefined (NaN) → None.
"""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool
from ht_backtest.data.funding import EXTREME_ABS


class FundingPositive:
    id = "funding_positive"
    category = "funding"
    # longs paying shorts — market overcrowded long
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        rate = bars["funding_rate"]
        return as_object_bool(rate > 0, rate.notna())


class FundingNegative:
    id = "funding_negative"
    category = "funding"
    # shorts paying longs — market overcrowded short
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        rate = bars["funding_rate"]
        return as_object_bool(rate < 0, rate.notna())


class FundingExtreme:
    id = "funding_extreme"
    category = "funding"
    # |rate| > 0.01% — unusually high either direction (squeeze risk)
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "funding_rate" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        rate = bars["funding_rate"]
        return as_object_bool(rate.abs() > EXTREME_ABS, rate.notna())


FUNDING_CONDITIONS = (
    FundingPositive(),
    FundingNegative(),
    FundingExtreme(),
)
