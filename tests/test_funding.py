"""Unit tests for funding merge causality and condition eval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ht_backtest.conditions.funding import FundingExtreme, FundingNegative, FundingPositive
from ht_backtest.data.funding import (
    EXTREME_ABS,
    flag_funding_anomalies,
    merge_funding_onto_bars,
)


def test_merge_asof_no_lookahead_and_stable_rows():
    # Bars every 15m around an 08:00 settlement
    opens = [
        "2024-03-01T07:45:00Z",
        "2024-03-01T08:00:00Z",
        "2024-03-01T09:00:00Z",
        "2024-03-01T16:00:00Z",
    ]
    bars = pd.DataFrame(
        {
            "timestamp": [int(pd.Timestamp(t).timestamp() * 1000) for t in opens],
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": [
                int(pd.Timestamp("2024-03-01T00:00:00Z").timestamp() * 1000),
                int(pd.Timestamp("2024-03-01T08:00:00Z").timestamp() * 1000),
                int(pd.Timestamp("2024-03-01T16:00:00Z").timestamp() * 1000),
            ],
            "funding_rate": [0.0001, -0.0002, 0.0003],
        }
    )
    merged = merge_funding_onto_bars(bars, funding)
    assert len(merged) == len(bars)
    assert list(merged["timestamp"]) == list(bars["timestamp"])
    # 07:45 → still on 00:00 print
    assert merged.iloc[0]["funding_rate"] == pytest.approx(0.0001)
    # 08:00 bar open == settlement → may carry 08:00 (settled at open)
    assert merged.iloc[1]["funding_rate"] == pytest.approx(-0.0002)
    # 09:00 → 08:00 settlement, NOT 16:00
    assert merged.iloc[2]["funding_rate"] == pytest.approx(-0.0002)
    # 16:00 → 16:00 settlement
    assert merged.iloc[3]["funding_rate"] == pytest.approx(0.0003)


def test_anomaly_flag_outside_point_one_percent():
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "funding_rate": [0.0001, 0.002, -0.0015],  # mid and last are anomalies (>0.1%)
        }
    )
    rep = flag_funding_anomalies(df, "BTC")
    assert rep.n_anomalies == 2


def test_funding_conditions():
    bars = pd.DataFrame(
        {
            "funding_rate": [0.0002, -0.0001, 0.0, np.nan, 0.00005],
        }
    )
    pos = list(FundingPositive().eval(bars))
    neg = list(FundingNegative().eval(bars))
    ext = list(FundingExtreme().eval(bars))
    assert pos == [True, False, False, None, True]
    assert neg == [False, True, False, None, False]
    assert ext[0] is True  # 0.0002 > 0.0001
    assert ext[1] is False  # 0.0001 is not > EXTREME_ABS
    assert abs(-0.0001) > EXTREME_ABS or ext[1] is False
    assert ext[3] is None
    assert ext[4] is False  # 0.00005 < 0.0001
