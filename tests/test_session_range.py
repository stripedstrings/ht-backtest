import numpy as np
import pandas as pd

from ht_backtest.gates.session_range import _PoolBook, run_session_range_engine


def test_pool_book_nearest_prefers_closest_and_caps_at_max_liq():
    book = _PoolBook(max_liq=2)
    book.add(100.0, origin_bar=0, kind=0)
    book.add(105.0, origin_bar=1, kind=0)
    book.add(110.0, origin_bar=2, kind=0)  # evicts the 100.0 entry (oldest)
    assert [p.price for p in book.pools] == [105.0, 110.0]
    nearest = book.nearest_above(101.0)
    assert nearest.price == 105.0


def test_pool_book_prune_removes_swept_levels():
    book = _PoolBook(max_liq=8)
    book.add(100.0, 0, 0)
    book.add(110.0, 0, 0)
    book.prune_swept_above(high=105.0)  # sweeps the 100.0 level only
    assert [p.price for p in book.pools] == [110.0]


def _empty_frames(n):
    idx = pd.RangeIndex(n)
    sessions = pd.DataFrame(
        {
            "ny_start": np.zeros(n, dtype=bool),
            "london_start": np.zeros(n, dtype=bool),
            "in_killzone": np.zeros(n, dtype=bool),
        },
        index=idx,
    )
    day_tags = pd.DataFrame({"new_day": np.zeros(n, dtype=bool)}, index=idx)
    pdh = pd.Series(np.nan, index=idx)
    pdl = pd.Series(np.nan, index=idx)
    pool_events = pd.DataFrame(
        {
            "pool_up": np.zeros(n, dtype=bool),
            "pool_dn": np.zeros(n, dtype=bool),
            "pool_up_price": np.full(n, np.nan),
            "pool_dn_price": np.full(n, np.nan),
            "is_eqh": np.zeros(n, dtype=bool),
            "is_eql": np.zeros(n, dtype=bool),
        },
        index=idx,
    )
    asia_pools = pd.DataFrame({"asia_pool_high": np.full(n, np.nan), "asia_pool_low": np.full(n, np.nan)}, index=idx)
    return sessions, day_tags, pdh, pdl, pool_events, asia_pools


def test_one_raid_per_side_per_session_and_gate_selects_nearest_unswept():
    n = 6
    df = pd.DataFrame({"high": [100.0] * n, "low": [90.0] * n, "close": [95.0] * n})
    atr = pd.Series(1.0, index=df.index)
    sessions, day_tags, pdh, pdl, pool_events, asia_pools = _empty_frames(n)

    # Two swing-high pools already resting above close=95 at bar 0's fractal
    # confirmation (bar 0), one closer (105) one farther (120).
    pool_events.loc[0, ["pool_up", "pool_up_price"]] = [True, 105.0]
    pool_events.loc[1, ["pool_up", "pool_up_price"]] = [True, 120.0]
    sessions.loc[2, ["london_start", "in_killzone"]] = [True, True]
    sessions.loc[3, "in_killzone"] = True
    sessions.loc[4, "in_killzone"] = True

    df.loc[2, "high"] = 100.0  # no raid yet
    df.loc[3, "high"] = 106.0  # first raid: sweeps the 105 range -> grab_up
    df.loc[4, "high"] = 130.0  # would sweep 120 too, but one_raid blocks a 2nd grab this session

    out = run_session_range_engine(df, atr, sessions, day_tags, pdh, pdl, pool_events, asia_pools, sw_len=0)

    assert out.loc[2, "kz_hi"] == 105.0  # nearest unswept pool above close chosen at killzone open
    assert out.loc[3, "grab_up"] == True
    assert out.loc[4, "grab_up"] == False  # one-raid-per-side-per-session


def test_grab_seq_resets_on_new_utc_day():
    n = 4
    df = pd.DataFrame({"high": [100.0, 106.0, 100.0, 106.0], "low": [90.0] * n, "close": [95.0] * n})
    atr = pd.Series(1.0, index=df.index)
    sessions, day_tags, pdh, pdl, pool_events, asia_pools = _empty_frames(n)
    pool_events.loc[0, ["pool_up", "pool_up_price"]] = [True, 105.0]
    sessions.loc[0, ["london_start", "in_killzone"]] = [True, True]
    sessions.loc[1, "in_killzone"] = True
    day_tags.loc[2, "new_day"] = True
    pool_events.loc[2, ["pool_up", "pool_up_price"]] = [True, 105.0]
    sessions.loc[2, ["london_start", "in_killzone"]] = [True, True]
    sessions.loc[3, "in_killzone"] = True

    out = run_session_range_engine(df, atr, sessions, day_tags, pdh, pdl, pool_events, asia_pools, sw_len=0)
    assert out.loc[1, "grab_up"] == True
    assert out.loc[1, "grab_seq"] == 1
    assert out.loc[3, "grab_up"] == True
    assert out.loc[3, "grab_seq"] == 1  # reset by new_day, not carried over as 2


def test_prevday_and_asia_pools_feed_into_gate():
    n = 3
    df = pd.DataFrame({"high": [100.0, 121.0, 100.0], "low": [90.0] * n, "close": [95.0] * n})
    atr = pd.Series(1.0, index=df.index)
    sessions, day_tags, pdh, pdl, pool_events, asia_pools = _empty_frames(n)
    day_tags.loc[0, "new_day"] = True
    pdh.loc[0] = 120.0
    pdl.loc[0] = 80.0
    sessions.loc[0, ["london_start", "in_killzone"]] = [True, True]
    sessions.loc[1, "in_killzone"] = True

    out = run_session_range_engine(df, atr, sessions, day_tags, pdh, pdl, pool_events, asia_pools, sw_len=0)
    assert out.loc[0, "kz_hi"] == 120.0
    assert out.loc[0, "kz_hi_type"] == 3  # PREVDAY
    assert out.loc[1, "grab_up"] == True
