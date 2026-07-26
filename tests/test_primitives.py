import numpy as np
import pandas as pd
import pytest

from ht_backtest.gates.primitives import (
    asia_range_pools,
    compute_atr,
    confirmed_fractal_high,
    confirmed_fractal_low,
    daily_bias_pivots,
    daily_prev_high_low,
    efficiency_ratio,
    fractal_high,
    fractal_low,
    pool_swing_events,
    protected_swing_levels,
    session_tags,
    utc_day_tags,
)

TF_MS = 15 * 60_000


def _mk_ts(n, start="2026-01-05 00:00:00"):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    return pd.Series(start_ms + np.arange(n) * TF_MS)


def _brute_force_fractal_high(high: pd.Series, n: int) -> pd.Series:
    v = high.to_numpy()
    out = np.zeros(len(v), dtype=bool)
    for j in range(n, len(v) - n):
        window = np.concatenate([v[j - n : j], v[j + 1 : j + n + 1]])
        out[j] = bool(np.all(v[j] > window))
    return pd.Series(out, index=high.index)


def _brute_force_fractal_low(low: pd.Series, n: int) -> pd.Series:
    v = low.to_numpy()
    out = np.zeros(len(v), dtype=bool)
    for j in range(n, len(v) - n):
        window = np.concatenate([v[j - n : j], v[j + 1 : j + n + 1]])
        out[j] = bool(np.all(v[j] < window))
    return pd.Series(out, index=low.index)


def test_fractal_high_matches_brute_force():
    rng = np.random.default_rng(1)
    high = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    for n in (2, 5):
        assert (fractal_high(high, n) == _brute_force_fractal_high(high, n)).all()


def test_fractal_low_matches_brute_force():
    rng = np.random.default_rng(2)
    low = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    for n in (2, 5):
        assert (fractal_low(low, n) == _brute_force_fractal_low(low, n)).all()


def test_fractal_high_simple_peak():
    # index:      0  1  2  3  4  5  6
    high = pd.Series([1, 2, 3, 10, 3, 2, 1.0])
    assert fractal_high(high, 3).tolist() == [False, False, False, True, False, False, False]


def test_confirmed_fractal_is_shifted_by_n():
    high = pd.Series([1, 2, 3, 10, 3, 2, 1.0])
    n = 3
    raw = fractal_high(high, n)
    confirmed = confirmed_fractal_high(high, n)
    assert raw.iloc[3]
    assert confirmed.iloc[3 + n]
    assert not confirmed.iloc[: 3 + n].any()


def test_protected_swing_levels_persist_between_fractals():
    high = pd.Series([1, 2, 3, 10.0, 3, 2, 1, 1, 1, 1, 1, 1])
    low = pd.Series([5, 4, 3, 1.0, 3, 4, 5, 5, 5, 5, 5, 5])
    last_hi, last_lo = protected_swing_levels(pd.DataFrame({"high": high, "low": low}), int_len=2)
    # the fractal high of 10 at index 3 confirms at index 5 and should persist forward
    assert last_hi.iloc[5] == 10.0
    assert last_hi.iloc[11] == 10.0
    assert last_lo.iloc[5] == 1.0
    assert last_lo.iloc[11] == 1.0


def test_atr_matches_manual_wilder_rma():
    df = pd.DataFrame(
        {
            "high": [10, 11, 10.5, 12, 11.5, 13, 12.5, 14, 13.5, 15, 14.5, 16, 15.5, 17, 16.5],
            "low": [9, 10, 9.5, 11, 10.5, 12, 11.5, 13, 12.5, 14, 13.5, 15, 14.5, 16, 15.5],
            "close": [9.5, 10.5, 10, 11.5, 11, 12.5, 12, 13.5, 13, 14.5, 14, 15.5, 15, 16.5, 16],
        }
    )
    atr = compute_atr(df, length=5)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    manual = tr.ewm(alpha=1 / 5, adjust=False, min_periods=5).mean()
    pd.testing.assert_series_equal(atr, manual)


def test_session_tags_summer_bst():
    # 2026-07-15 is BST (UTC+1). London 07:00 local = 06:00 UTC.
    ts = pd.Series(
        [
            int(pd.Timestamp("2026-07-15 06:00:00", tz="UTC").timestamp() * 1000),  # 07:00 BST -> in London
            int(pd.Timestamp("2026-07-15 08:59:00", tz="UTC").timestamp() * 1000),  # 09:59 BST -> in London
            int(pd.Timestamp("2026-07-15 05:59:00", tz="UTC").timestamp() * 1000),  # 06:59 BST -> not yet
            int(pd.Timestamp("2026-07-15 11:30:00", tz="UTC").timestamp() * 1000),  # 12:30 BST -> in NY kz
        ]
    )
    tags = session_tags(ts)
    assert tags["in_london"].tolist() == [True, True, False, False]
    assert tags["in_ny"].tolist() == [False, False, False, True]


def test_session_tags_winter_gmt():
    # 2026-01-15 is GMT (UTC+0). London 07:00 local = 07:00 UTC.
    ts = pd.Series(
        [
            int(pd.Timestamp("2026-01-15 07:00:00", tz="UTC").timestamp() * 1000),
            int(pd.Timestamp("2026-01-15 06:59:00", tz="UTC").timestamp() * 1000),
        ]
    )
    tags = session_tags(ts)
    assert tags["in_london"].tolist() == [True, False]


def test_session_start_flags_fire_once():
    ts = _mk_ts(8, start="2026-01-15 06:45:00")  # crosses 07:00 UTC = London open in GMT
    tags = session_tags(ts)
    assert tags["london_start"].sum() == 1
    assert tags["london_start"].idxmax() == tags["in_london"].idxmax()


def test_utc_day_boundary_three_bars():
    # 2026-07-15 23:45 UTC = 2026-07-16 00:45 BST local. new_day (UTC) must NOT
    # flip at London midnight; it flips at UTC midnight only.
    ts = pd.Series(
        [
            int(pd.Timestamp("2026-07-15 23:30:00", tz="UTC").timestamp() * 1000),
            int(pd.Timestamp("2026-07-15 23:45:00", tz="UTC").timestamp() * 1000),
            int(pd.Timestamp("2026-07-16 00:00:00", tz="UTC").timestamp() * 1000),
        ]
    )
    tags = utc_day_tags(ts)
    assert tags["new_day"].tolist() == [True, False, True]


def test_daily_prev_high_low_no_lookahead_and_correct_values():
    ts = _mk_ts(96 * 3, start="2026-01-05 00:00:00")  # 3 full UTC days
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.1, len(ts)))
    high = close + rng.uniform(0, 1, len(ts))
    low = close - rng.uniform(0, 1, len(ts))
    df = pd.DataFrame({"timestamp": ts, "high": high, "low": low, "close": close})

    pdh, pdl = daily_prev_high_low(df)
    utc_date = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    day0, day1, day2 = sorted(utc_date.unique())

    assert pdh[utc_date == day0].isna().all()
    day0_high = df.loc[utc_date == day0, "high"].max()
    day0_low = df.loc[utc_date == day0, "low"].min()
    assert (pdh[utc_date == day1] == day0_high).all()
    assert (pdl[utc_date == day1] == day0_low).all()

    day1_high = df.loc[utc_date == day1, "high"].max()
    assert (pdh[utc_date == day2] == day1_high).all()


def test_pool_swing_events_eqh_within_tolerance():
    # two swing highs of 10.0 and 10.02 separated by a swing low; ATR ~1.0 so
    # 0.02 move is well within 0.10*ATR tolerance -> second one tags EQH.
    high = pd.Series([1, 2, 3, 10.0, 3, 2, 1, 2, 3, 10.02, 3, 2, 1])
    low = pd.Series([9, 8, 7, 1.0, 7, 8, 9, 8, 7, 1.0, 7, 8, 9])
    df = pd.DataFrame({"high": high, "low": low, "close": (high + low) / 2})
    atr = pd.Series(1.0, index=df.index)
    out = pool_swing_events(df, atr, sw_len=3, eq_tol_mult=0.10)
    assert out["pool_up"].sum() == 2
    eqh_rows = out.index[out["is_eqh"]]
    assert len(eqh_rows) == 1
    assert out.loc[eqh_rows[0], "pool_up_price"] == 10.02


def test_pool_swing_events_no_eqh_outside_tolerance():
    high = pd.Series([1, 2, 3, 10.0, 3, 2, 1, 2, 3, 15.0, 3, 2, 1])
    low = pd.Series([9, 8, 7, 1.0, 7, 8, 9, 8, 7, 1.0, 7, 8, 9])
    df = pd.DataFrame({"high": high, "low": low, "close": (high + low) / 2})
    atr = pd.Series(1.0, index=df.index)
    out = pool_swing_events(df, atr, sw_len=3, eq_tol_mult=0.10)
    assert out["is_eqh"].sum() == 0


def test_asia_range_pools_freezes_running_extremes():
    ts = _mk_ts(96, start="2026-01-15 00:00:00")  # full UTC day, London=GMT so Asia=UTC too
    high = pd.Series(np.full(96, 100.0))
    low = pd.Series(np.full(96, 90.0))
    # spike the range strictly within Asia window (00:00-06:00 = bars 0..23)
    high.iloc[10] = 150.0
    low.iloc[15] = 50.0
    df = pd.DataFrame({"timestamp": ts, "high": high, "low": low, "close": (high + low) / 2})
    sessions = session_tags(df["timestamp"])
    out = asia_range_pools(df, sessions)
    asia_end_idx = sessions.index[sessions["asia_end"]][0]
    assert out.loc[asia_end_idx, "asia_pool_high"] == 150.0
    assert out.loc[asia_end_idx, "asia_pool_low"] == 50.0
    # only one freeze event in a single day
    assert out["asia_pool_high"].notna().sum() == 1


def test_daily_bias_pivots_smoke():
    ts = _mk_ts(96 * 20, start="2026-01-05 00:00:00")
    rng = np.random.default_rng(4)
    close = 100 + np.cumsum(rng.normal(0, 0.3, len(ts)))
    high = close + rng.uniform(0, 1, len(ts))
    low = close - rng.uniform(0, 1, len(ts))
    df = pd.DataFrame({"timestamp": ts, "high": high, "low": low, "close": close})
    out = daily_bias_pivots(df, d_piv=2)
    assert len(out) == len(df)
    assert {"H1", "H2", "L1", "L2", "str_bull", "str_bear", "eq", "premium", "bias_bull", "bias_bear"}.issubset(
        out.columns
    )
    # every trade always gets a direction -- never both/neither
    assert (out["bias_bull"] != out["bias_bear"]).all()


def test_daily_bias_premium_uses_intraday_close_not_daily():
    ts = _mk_ts(96 * 10, start="2026-01-05 00:00:00")
    close = pd.Series(100.0, index=range(len(ts)))
    high = close + 1
    low = close - 1
    df = pd.DataFrame({"timestamp": ts, "high": high, "low": low, "close": close})
    out = daily_bias_pivots(df, d_piv=2)
    # flat price throughout -> eq settles at 100 (own high/low fallback) ->
    # close==eq everywhere -> premium (strict >) is False for a flat market
    assert (~out["premium"]).all()


def test_efficiency_ratio_low_in_range_high_in_trend():
    n = 100
    ctx = 40
    # pure range: oscillates but nets to ~0 over ctx_bars -> low ER
    ranging = pd.Series(100 + 2 * np.sin(np.arange(n) * 0.5))
    # pure trend: monotonic -> net move == sum of abs moves -> ER == 1
    trending = pd.Series(100 + np.arange(n) * 0.5)

    er_range = efficiency_ratio(pd.DataFrame({"close": ranging}), ctx_bars=ctx)
    er_trend = efficiency_ratio(pd.DataFrame({"close": trending}), ctx_bars=ctx)

    assert er_trend.iloc[-1] == pytest.approx(1.0)
    assert er_range.iloc[-1] < 0.3
