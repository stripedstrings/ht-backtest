"""Unit tests for OI merge causality and condition eval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ht_backtest.conditions.oi import OiExtreme, OiFalling, OiRising
from ht_backtest.data.oi import merge_oi_onto_bars, oi_extreme_mask


def _ms(iso: str) -> int:
    return int(pd.Timestamp(iso).timestamp() * 1000)


def test_merge_asof_no_lookahead_and_stable_rows():
    # Bars every 15m; OI snapshots at 08:00 and 08:15.
    opens = [
        "2024-03-01T07:45:00Z",
        "2024-03-01T08:00:00Z",
        "2024-03-01T08:15:00Z",
        "2024-03-01T09:00:00Z",
    ]
    bars = pd.DataFrame(
        {
            "timestamp": [_ms(t) for t in opens],
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )
    oi = pd.DataFrame(
        {
            "timestamp": [
                _ms("2024-03-01T07:00:00Z"),
                _ms("2024-03-01T08:00:00Z"),
                _ms("2024-03-01T08:15:00Z"),
            ],
            "open_interest": [100.0, 110.0, 999.0],
        }
    )
    merged = merge_oi_onto_bars(bars, oi)
    assert len(merged) == len(bars)
    assert list(merged["timestamp"]) == list(bars["timestamp"])
    # 07:45 → still on 07:00 print
    assert merged.iloc[0]["open_interest"] == pytest.approx(100.0)
    # 08:00 bar open == snapshot → may carry 08:00
    assert merged.iloc[1]["open_interest"] == pytest.approx(110.0)
    # 08:15 → 08:15 snapshot
    assert merged.iloc[2]["open_interest"] == pytest.approx(999.0)
    # 09:00 → 08:15, NOT a later print (none)
    assert merged.iloc[3]["open_interest"] == pytest.approx(999.0)


def test_future_oi_print_invisible_at_earlier_bar_open():
    bars = pd.DataFrame({"timestamp": [_ms("2024-03-01T08:00:00Z")], "close": [1.0]})
    oi = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T08:15:00Z")],
            "open_interest": [999.0],
        }
    )
    merged = merge_oi_onto_bars(bars, oi)
    assert len(merged) == 1
    assert pd.isna(merged.iloc[0]["open_interest"])


def test_empty_oi_adds_nan_column_without_row_change():
    bars = pd.DataFrame({"timestamp": [1, 2, 3], "close": [1.0, 2.0, 3.0]})
    merged = merge_oi_onto_bars(bars, pd.DataFrame(columns=["timestamp", "open_interest"]))
    assert len(merged) == 3
    assert merged["open_interest"].isna().all()


def test_oi_rising_falling_none_safe():
    bars = pd.DataFrame({"open_interest": [10.0, 12.0, 11.0, np.nan, 11.0]})
    rising = list(OiRising().eval(bars))
    falling = list(OiFalling().eval(bars))
    assert rising[0] is None  # no prior
    assert rising[1] is True
    assert rising[2] is False
    assert rising[3] is None
    assert rising[4] is None  # prior was NaN
    assert falling[0] is None
    assert falling[1] is False
    assert falling[2] is True
    assert OiRising().eval(pd.DataFrame({"x": [1]})).iloc[0] is None


def test_oi_extreme_uses_prior_window_not_current_or_future():
    # Five modest values, spike at t=5, larger future spike at t=6.
    # Prior-only: t=5 compares 50 to p90 of five 10s → extreme.
    # Lookahead window {t-3..t+1} includes 1000 and would un-flag t=5.
    oi = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 50.0, 1000.0])
    extreme, defined = oi_extreme_mask(oi, lookback=5, min_periods=5)
    assert not bool(defined.iloc[4])
    assert bool(defined.iloc[5])
    assert bool(extreme.iloc[5]) is True
    lookahead_window = oi.iloc[2:7]  # 10,10,10,50,1000 — includes t+1
    leak_p90 = float(lookahead_window.quantile(0.90))
    assert not (50.0 > leak_p90)
    assert bool(extreme.iloc[5]) is True


def test_oi_extreme_eval_none_until_min_periods_then_spike():
    bars = pd.DataFrame({"open_interest": [10.0] * 100 + [1000.0]})
    out = list(OiExtreme().eval(bars))
    assert out[95] is None  # min_periods=96 prior bars → defined from index 96
    assert out[-1] is True


def test_real_oi_cache_merge_preserves_bar_count():
    """If universe OI was downloaded, merge onto a dense 15m index without row drift."""
    from pathlib import Path

    from ht_backtest.data.oi import attach_open_interest, merge_oi_onto_bars, oi_path

    path = oi_path("BTC/USDT:USDT", Path("data/oi"))
    if not path.exists():
        pytest.skip("no BTC OI cache")
    oi = pd.read_parquet(path)
    start = int(oi["timestamp"].iloc[0]) - 4 * 15 * 60 * 1000
    end = int(oi["timestamp"].iloc[-1]) + 4 * 15 * 60 * 1000
    step = 15 * 60 * 1000
    ts = list(range(start, end + step, step))
    bars = pd.DataFrame({"timestamp": ts, "close": [1.0] * len(ts)})
    n0 = len(bars)
    merged = merge_oi_onto_bars(bars, oi)
    assert len(merged) == n0
    assert list(merged["timestamp"]) == ts
    attached = attach_open_interest(bars, "BTC/USDT:USDT", oi_dir="data/oi")
    assert len(attached) == n0
    assert attached["open_interest"].notna().sum() > 0


def test_oi_extreme_condition_none_without_column():
    assert OiExtreme().eval(pd.DataFrame({"x": [1.0]})).iloc[0] is None
