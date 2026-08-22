"""Unit tests for condition library — True / False / None on synthetic features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ht_backtest.conditions.funding import FundingExtreme, FundingNegative, FundingPositive
from ht_backtest.conditions.htf import Above4hEma20, Below4hEma20, H4HhHl, H4LhLl
from ht_backtest.conditions.prior_session import (
    LondonRaidedHigh,
    LondonRaidedLow,
    PriorSessionOppositeDirection,
    PriorSessionSameDirection,
)
from ht_backtest.conditions.range_ctx import (
    AsiaRangeTight,
    AsiaRangeWide,
    PriceAboveAsiaMid,
    PriceBelowAsiaMid,
)
from ht_backtest.conditions.registry import ALL_CONDITIONS, MUTEX_PAIRS, is_mutex_combo
from ht_backtest.conditions.session import (
    AsiaSession,
    LondonOpen30m,
    LondonSession,
    NyOpen30m,
    NySession,
)
from ht_backtest.conditions.volume import VolumeHigh, VolumeLow, VolumeSpike
from ht_backtest.data.htf_4h import TF_4H_MS, attach_4h_features, closed_4h_feature_table, ema


def _obj(vals):
    return list(vals)


def test_library_size_and_mutex():
    assert len(ALL_CONDITIONS) == 29
    assert is_mutex_combo(["london_session", "ny_session"])
    assert is_mutex_combo(["london_session", "london_raided_high"])
    assert is_mutex_combo(["london_session", "london_raided_low"])
    assert is_mutex_combo(["oi_rising", "oi_falling"])
    assert not is_mutex_combo(["recent_long_liq", "recent_short_liq"])
    assert not is_mutex_combo(["london_session", "volume_high"])
    assert len(MUTEX_PAIRS) == 15


def test_session_conditions():
    bars = pd.DataFrame(
        {
            "feat_london_open_30m": [True, False, False],
            "feat_ny_open_30m": [False, True, False],
            "feat_london_session": [True, False, False],
            "feat_ny_session": [False, True, False],
            "feat_asia_session": [False, False, True],
        }
    )
    assert _obj(LondonOpen30m().eval(bars)) == [True, False, False]
    assert _obj(NyOpen30m().eval(bars)) == [False, True, False]
    assert _obj(LondonSession().eval(bars)) == [True, False, False]
    assert _obj(NySession().eval(bars)) == [False, True, False]
    assert _obj(AsiaSession().eval(bars)) == [False, False, True]
    assert LondonSession().eval(pd.DataFrame({"x": [1]})).iloc[0] is None


def test_volume_conditions():
    bars = pd.DataFrame(
        {
            "feat_volume_high": [True, False, False],
            "feat_volume_low": [False, True, False],
            "feat_volume_spike": [True, False, False],
            "feat_volume_defined": [True, True, False],
        }
    )
    assert _obj(VolumeHigh().eval(bars)) == [True, False, None]
    assert _obj(VolumeLow().eval(bars)) == [False, True, None]
    assert _obj(VolumeSpike().eval(bars)) == [True, False, None]


def test_range_conditions():
    bars = pd.DataFrame(
        {
            "feat_asia_range_tight": [True, False, False],
            "feat_asia_range_wide": [False, True, False],
            "feat_price_above_asia_mid": [True, False, False],
            "feat_price_below_asia_mid": [False, True, False],
            "feat_asia_range_defined": [True, True, False],
            "feat_asia_mid_defined": [True, True, False],
        }
    )
    assert _obj(AsiaRangeTight().eval(bars)) == [True, False, None]
    assert _obj(AsiaRangeWide().eval(bars)) == [False, True, None]
    assert _obj(PriceAboveAsiaMid().eval(bars)) == [True, False, None]
    assert _obj(PriceBelowAsiaMid().eval(bars)) == [False, True, None]


def test_htf_conditions():
    bars = pd.DataFrame(
        {
            "feat_above_4h_ema20": [True, False, False],
            "feat_below_4h_ema20": [False, True, False],
            "feat_4h_hh_hl": [True, False, False],
            "feat_4h_lh_ll": [False, True, False],
            "feat_4h_ema_defined": [True, True, False],
            "feat_4h_structure_defined": [True, True, False],
        }
    )
    assert _obj(Above4hEma20().eval(bars)) == [True, False, None]
    assert _obj(Below4hEma20().eval(bars)) == [False, True, None]
    assert _obj(H4HhHl().eval(bars)) == [True, False, None]
    assert _obj(H4LhLl().eval(bars)) == [False, True, None]


def test_prior_and_funding_conditions():
    bars = pd.DataFrame(
        {
            "feat_london_raided_high": [1.0, 0.0, np.nan],
            "feat_london_raided_low": [0.0, 1.0, np.nan],
            "feat_prior_session_same": [1.0, 0.0, np.nan],
            "feat_prior_session_opp": [0.0, 1.0, np.nan],
            "funding_rate": [0.0002, -0.0001, np.nan],
        }
    )
    assert _obj(LondonRaidedHigh().eval(bars)) == [True, False, None]
    assert _obj(LondonRaidedLow().eval(bars)) == [False, True, None]
    assert _obj(PriorSessionSameDirection().eval(bars)) == [True, False, None]
    assert _obj(PriorSessionOppositeDirection().eval(bars)) == [False, True, None]
    assert _obj(FundingPositive().eval(bars)) == [True, False, None]
    assert _obj(FundingNegative().eval(bars)) == [False, True, None]
    assert FundingExtreme().eval(bars).iloc[0] is True


def test_4h_merge_uses_closed_bar_only():
    # 4h bars: 00:00-04:00 and 04:00-08:00
    h4 = pd.DataFrame(
        {
            "timestamp": [
                int(pd.Timestamp("2024-03-01T00:00:00Z").timestamp() * 1000),
                int(pd.Timestamp("2024-03-01T04:00:00Z").timestamp() * 1000),
                int(pd.Timestamp("2024-03-01T08:00:00Z").timestamp() * 1000),
            ],
            "open": [100.0, 110.0, 120.0],
            "high": [105.0, 115.0, 125.0],
            "low": [99.0, 109.0, 119.0],
            "close": [104.0, 114.0, 124.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    # Enough history for EMA: pad with flat bars before
    pad = []
    t0 = int(pd.Timestamp("2024-02-20T00:00:00Z").timestamp() * 1000)
    for i in range(30):
        ts = t0 + i * TF_4H_MS
        pad.append({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
    h4 = pd.concat([pd.DataFrame(pad), h4], ignore_index=True)

    bars = pd.DataFrame(
        {
            "timestamp": [
                int(pd.Timestamp("2024-03-01T09:00:00Z").timestamp() * 1000),  # forming 08:00 bar
                int(pd.Timestamp("2024-03-01T12:00:00Z").timestamp() * 1000),  # 08:00 closed
            ],
            "open": [200.0, 201.0],
            "high": [201.0, 202.0],
            "low": [199.0, 200.0],
            "close": [200.5, 201.5],
            "volume": [1.0, 1.0],
        }
    )
    merged = attach_4h_features(bars, h4)
    feats = closed_4h_feature_table(h4)
    # At 09:00, last closed 4h is 04:00-08:00 (close_time=08:00), NOT 08:00-12:00
    row_0800 = feats.loc[feats["open_time_ms"] == int(pd.Timestamp("2024-03-01T04:00:00Z").timestamp() * 1000)].iloc[0]
    assert merged.iloc[0]["htf_4h_close"] == pytest.approx(row_0800["close"])
    # Manual EMA from closes with close_time <= 09:00
    closed = feats[feats["close_time_ms"] <= bars.iloc[0]["timestamp"]]
    manual = float(ema(closed["close"]).iloc[-1])
    assert merged.iloc[0]["htf_4h_ema20"] == pytest.approx(manual, rel=1e-9)
