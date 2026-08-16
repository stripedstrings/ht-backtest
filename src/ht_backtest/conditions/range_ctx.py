"""Asia range context conditions."""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool


class AsiaRangeTight:
    id = "asia_range_tight"
    category = "range"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_asia_range_tight" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_asia_range_tight"], bars["feat_asia_range_defined"])


class AsiaRangeWide:
    id = "asia_range_wide"
    category = "range"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_asia_range_wide" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_asia_range_wide"], bars["feat_asia_range_defined"])


class PriceAboveAsiaMid:
    id = "price_above_asia_mid"
    category = "range"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_price_above_asia_mid" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_price_above_asia_mid"], bars["feat_asia_mid_defined"])


class PriceBelowAsiaMid:
    id = "price_below_asia_mid"
    category = "range"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_price_below_asia_mid" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_price_below_asia_mid"], bars["feat_asia_mid_defined"])


RANGE_CONDITIONS = (
    AsiaRangeTight(),
    AsiaRangeWide(),
    PriceAboveAsiaMid(),
    PriceBelowAsiaMid(),
)
