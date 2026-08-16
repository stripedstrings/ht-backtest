"""4h HTF trend conditions (requires htf_4h_* columns from attach_4h_features)."""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool


class Above4hEma20:
    id = "above_4h_ema20"
    category = "htf"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_above_4h_ema20" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_above_4h_ema20"], bars["feat_4h_ema_defined"])


class Below4hEma20:
    id = "below_4h_ema20"
    category = "htf"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_below_4h_ema20" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_below_4h_ema20"], bars["feat_4h_ema_defined"])


class H4HhHl:
    id = "4h_hh_hl"
    category = "htf"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_4h_hh_hl" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_4h_hh_hl"], bars["feat_4h_structure_defined"])


class H4LhLl:
    id = "4h_lh_ll"
    category = "htf"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        if "feat_4h_lh_ll" not in bars.columns:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(bars["feat_4h_lh_ll"], bars["feat_4h_structure_defined"])


HTF_CONDITIONS = (Above4hEma20(), Below4hEma20(), H4HhHl(), H4LhLl())
