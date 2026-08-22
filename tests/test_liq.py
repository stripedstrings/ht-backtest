"""Causal timestamp tests for liquidations — in-bar events must not leak.

A liquidation that fires inside the current 15m bar is invisible to that
bar's conditions. Only the completed bin that closed at the bar open attaches.
Empty interior windows are 0 (flow), not a ffill of the last event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ht_backtest.conditions.liq import LiqSpike, RecentLongLiq, RecentShortLiq
from ht_backtest.data.liq import liq_spike_mask, merge_liq_onto_bars


def _ms(iso: str) -> int:
    return int(pd.Timestamp(iso).timestamp() * 1000)


def test_liquidation_inside_current_bar_is_invisible():
    """THE Phase 1 lookahead test: in-bar force-order must not attach to that bar.

    Bar 08:00–08:15. Liquidation at 08:07:00 (inside the bar, side=SELL / longs).
    Bar open 08:00 must not see it. Next bar (08:15) may see it — the bin closed.
    """
    bars = pd.DataFrame(
        {
            "timestamp": [
                _ms("2024-03-01T08:00:00Z"),
                _ms("2024-03-01T08:15:00Z"),
            ],
            "close": [1.0, 2.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T08:07:00Z")],
            "side": ["SELL"],
            "qty": [10.0],
        }
    )
    merged = merge_liq_onto_bars(bars, events)
    assert len(merged) == 2
    assert list(merged["timestamp"]) == list(bars["timestamp"])
    # Current bar: in-bar liq is invisible
    assert pd.isna(merged.iloc[0]["liq_long_qty"])
    assert pd.isna(merged.iloc[0]["liq_qty"])
    # Next bar: bin [08:00, 08:15) has closed at 08:15 → visible
    assert merged.iloc[1]["liq_long_qty"] == pytest.approx(10.0)
    assert merged.iloc[1]["liq_short_qty"] == pytest.approx(0.0)
    assert merged.iloc[1]["liq_qty"] == pytest.approx(10.0)


def test_event_exactly_at_bar_open_is_inside_that_bar():
    """Event at T belongs to [T, T+15m) — invisible to the T bar, visible at T+15m."""
    bars = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T08:00:00Z"), _ms("2024-03-01T08:15:00Z")],
            "close": [1.0, 2.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T08:00:00Z")],
            "side": ["BUY"],
            "qty": [3.0],
        }
    )
    merged = merge_liq_onto_bars(bars, events)
    assert pd.isna(merged.iloc[0]["liq_short_qty"])
    assert merged.iloc[1]["liq_short_qty"] == pytest.approx(3.0)


def test_previous_bar_liquidation_is_visible():
    bars = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T08:00:00Z")],
            "close": [1.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T07:50:00Z")],
            "side": ["SELL"],
            "qty": [7.0],
        }
    )
    merged = merge_liq_onto_bars(bars, events)
    assert merged.iloc[0]["liq_long_qty"] == pytest.approx(7.0)


def test_empty_window_is_zero_not_ffill():
    """Flow semantics: a later empty 15m bin must not inherit the previous liq."""
    bars = pd.DataFrame(
        {
            "timestamp": [
                _ms("2024-03-01T08:15:00Z"),
                _ms("2024-03-01T08:30:00Z"),
                _ms("2024-03-01T08:45:00Z"),
            ],
            "close": [1.0, 2.0, 3.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T08:07:00Z"), _ms("2024-03-01T08:40:00Z")],
            "side": ["SELL", "BUY"],
            "qty": [10.0, 5.0],
        }
    )
    merged = merge_liq_onto_bars(bars, events)
    assert merged.iloc[0]["liq_long_qty"] == pytest.approx(10.0)
    assert merged.iloc[1]["liq_qty"] == pytest.approx(0.0)
    assert merged.iloc[2]["liq_short_qty"] == pytest.approx(5.0)
    assert merged.iloc[2]["liq_long_qty"] == pytest.approx(0.0)


def test_empty_events_adds_nan_columns_without_row_change():
    bars = pd.DataFrame({"timestamp": [1, 2, 3], "close": [1.0, 2.0, 3.0]})
    merged = merge_liq_onto_bars(bars, pd.DataFrame(columns=["timestamp", "side", "qty"]))
    assert len(merged) == 3
    assert merged["liq_qty"].isna().all()


def test_recent_liq_conditions_none_safe():
    bars = pd.DataFrame(
        {
            "liq_long_qty": [np.nan, 10.0, 0.0],
            "liq_short_qty": [np.nan, 0.0, 4.0],
        }
    )
    longs = list(RecentLongLiq().eval(bars))
    shorts = list(RecentShortLiq().eval(bars))
    assert longs == [None, True, False]
    assert shorts == [None, False, True]
    assert RecentLongLiq().eval(pd.DataFrame({"x": [1]})).iloc[0] is None
    assert RecentShortLiq().eval(pd.DataFrame({"x": [1]})).iloc[0] is None


def test_in_bar_liq_does_not_fire_current_bar_conditions():
    """End-to-end: in-bar SELL must not make recent_long_liq True on that bar."""
    bars = pd.DataFrame(
        {
            "timestamp": [
                _ms("2024-03-01T08:00:00Z"),
                _ms("2024-03-01T08:15:00Z"),
            ],
            "close": [1.0, 2.0],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": [_ms("2024-03-01T08:07:00Z")],
            "side": ["SELL"],
            "qty": [10.0],
        }
    )
    merged = merge_liq_onto_bars(bars, events)
    longs = list(RecentLongLiq().eval(merged))
    assert longs[0] is None
    assert longs[1] is True


def test_liq_spike_uses_prior_window_not_current_or_future():
    qty = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 1000.0])
    spike, defined = liq_spike_mask(qty, lookback=5, min_periods=5)
    assert not bool(defined.iloc[4])
    assert bool(defined.iloc[5])
    assert bool(spike.iloc[5]) is True
    lookahead_window = qty.iloc[2:7]
    leak_p90 = float(lookahead_window.quantile(0.90))
    assert not (50.0 > leak_p90)
    assert bool(spike.iloc[5]) is True


def test_liq_spike_eval_none_until_min_periods():
    bars = pd.DataFrame({"liq_qty": [1.0] * 20 + [100.0]})
    out = list(LiqSpike().eval(bars))
    assert out[19] is None  # min_periods=20 prior bars → defined from index 20
    assert out[-1] is True
    assert LiqSpike().eval(pd.DataFrame({"x": [1.0]})).iloc[0] is None


def test_real_liq_cache_merge_preserves_bar_count():
    from pathlib import Path

    from ht_backtest.data.liq import attach_liquidations, liq_path

    path = liq_path("BTC/USDT:USDT", Path("data/liq"))
    if not path.exists():
        pytest.skip("no BTC liq cache")
    events = pd.read_parquet(path)
    if events.empty:
        pytest.skip("empty BTC liq cache")
    start = int(events["timestamp"].iloc[0]) - 4 * 15 * 60 * 1000
    end = int(events["timestamp"].iloc[-1]) + 4 * 15 * 60 * 1000
    step = 15 * 60 * 1000
    ts = list(range(start, end + step, step))
    bars = pd.DataFrame({"timestamp": ts, "close": [1.0] * len(ts)})
    n0 = len(bars)
    merged = merge_liq_onto_bars(bars, events)
    assert len(merged) == n0
    assert list(merged["timestamp"]) == ts
    attached = attach_liquidations(bars, "BTC/USDT:USDT", liq_dir="data/liq")
    assert len(attached) == n0
    assert attached["liq_qty"].notna().sum() > 0
