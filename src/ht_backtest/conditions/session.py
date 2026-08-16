"""Session-timing conditions (Europe/London clock)."""

from __future__ import annotations

import pandas as pd

from ht_backtest.conditions.base import as_object_bool


def _col(bars: pd.DataFrame, name: str) -> pd.Series | None:
    return bars[name] if name in bars.columns else None


class LondonOpen30m:
    id = "london_open_30m"
    category = "session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        s = _col(bars, "feat_london_open_30m")
        if s is None:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(s.astype(bool), pd.Series(True, index=bars.index))


class NyOpen30m:
    id = "ny_open_30m"
    category = "session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        s = _col(bars, "feat_ny_open_30m")
        if s is None:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(s.astype(bool), pd.Series(True, index=bars.index))


class LondonSession:
    id = "london_session"
    category = "session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        s = _col(bars, "feat_london_session")
        if s is None:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(s.astype(bool), pd.Series(True, index=bars.index))


class NySession:
    id = "ny_session"
    category = "session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        s = _col(bars, "feat_ny_session")
        if s is None:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(s.astype(bool), pd.Series(True, index=bars.index))


class AsiaSession:
    id = "asia_session"
    category = "session"
    version = 1

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        s = _col(bars, "feat_asia_session")
        if s is None:
            return pd.Series([None] * len(bars), index=bars.index, dtype=object)
        return as_object_bool(s.astype(bool), pd.Series(True, index=bars.index))


SESSION_CONDITIONS = (
    LondonOpen30m(),
    NyOpen30m(),
    LondonSession(),
    NySession(),
    AsiaSession(),
)
