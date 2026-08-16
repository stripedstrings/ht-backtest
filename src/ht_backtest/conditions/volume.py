"""Volume conditions vs prior rolling history (causal)."""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool


class VolumeHigh:
    id = "volume_high"
    category = "volume"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_volume_high" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_volume_high"], bars["feat_volume_defined"])


class VolumeLow:
    id = "volume_low"
    category = "volume"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_volume_low" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_volume_low"], bars["feat_volume_defined"])


class VolumeSpike:
    id = "volume_spike"
    category = "volume"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_volume_spike" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_volume_spike"], bars["feat_volume_defined"])


VOLUME_CONDITIONS = (VolumeHigh(), VolumeLow(), VolumeSpike())
