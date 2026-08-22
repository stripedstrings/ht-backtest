"""Liquidation conditions (completed 15m bins only).

Requires ``liq_long_qty`` / ``liq_short_qty`` / ``liq_qty`` from
``attach_liquidations``. Undefined (NaN) → None.

A bar at open T sees only the bin that closed at T (events in [T-15m, T)).
Liquidations inside [T, T+15m) are invisible to this bar.
"""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool
from ht_backtest.data.liq import liq_spike_mask


class RecentLongLiq:
    id = "recent_long_liq"
    category = "liq"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "liq_long_qty" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        qty = bars["liq_long_qty"].astype(float)
        return as_object_bool(qty > 0, qty.notna())


class RecentShortLiq:
    id = "recent_short_liq"
    category = "liq"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "liq_short_qty" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        qty = bars["liq_short_qty"].astype(float)
        return as_object_bool(qty > 0, qty.notna())


class LiqSpike:
    id = "liq_spike"
    category = "liq"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "liq_qty" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        qty = bars["liq_qty"].astype(float)
        spike, defined = liq_spike_mask(qty)
        return as_object_bool(spike, defined)


LIQ_CONDITIONS = (
    RecentLongLiq(),
    RecentShortLiq(),
    LiqSpike(),
)
