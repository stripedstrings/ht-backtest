import numpy as np
import pandas as pd

from ht_backtest.trades.forward_tracker import track_forward_reach

TARGETS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def _df(n, base=100.0):
    return pd.DataFrame(
        {
            "open": [base] * n,
            "high": [base] * n,
            "low": [base] * n,
            "close": [base] * n,
        }
    )


def _trade(direction, entry_bar, entry_price, risk):
    return {"direction": direction, "entry_bar": entry_bar, "entry_price": entry_price, "risk": risk}


def test_reaches_2r_before_stop_short():
    # short, entry=100, risk=10 -> 1R stop at high=110; 2R favorable at low=80
    n = 10
    df = _df(n)
    df.loc[1, "low"] = 85.0   # 1.5R favorable
    df.loc[2, "low"] = 80.0   # 2R favorable
    df.loc[3, "high"] = 110.0  # 1R adverse -> stop
    trades = pd.DataFrame([_trade("short", 0, 100.0, 10.0)])
    out = track_forward_reach(trades, df, mfe_win=100, targets=TARGETS)
    r = out.iloc[0]
    assert r["shadow_resolution"] == "stop"
    assert r["reach_0.5R"] and r["reach_1.0R"] and r["reach_1.5R"] and r["reach_2.0R"]
    assert not r["reach_2.5R"] and not r["reach_3.0R"]
    assert r["intrabar_ambiguous"] == False


def test_intrabar_ambiguous_bar_does_not_credit_same_bar_new_high():
    # short, entry=100, risk=10. bar1: low=90 (1R favorable, clean). bar2: BOTH
    # high=110 (1R adverse/stop) AND low=70 (would be 3R favorable) in the SAME
    # bar -> conservative rule must NOT credit anything beyond what bar1 gave.
    n = 10
    df = _df(n)
    df.loc[1, "low"] = 90.0
    df.loc[2, "high"] = 110.0
    df.loc[2, "low"] = 70.0
    trades = pd.DataFrame([_trade("short", 0, 100.0, 10.0)])
    out = track_forward_reach(trades, df, mfe_win=100, targets=TARGETS)
    r = out.iloc[0]
    assert r["shadow_resolution"] == "stop"
    assert r["intrabar_ambiguous"] == True
    assert r["intrabar_ambiguous_bar"] == 2
    assert r["reach_1.0R"] == True   # from bar 1, strictly before the ambiguous bar
    assert r["reach_1.5R"] == False  # would need the ambiguous bar's move -- conservatively denied
    assert r["reach_3.0R"] == False
    assert r["shadow_mfe_r"] == 1.0  # frozen at the pre-ambiguous-bar value


def test_no_flag_when_stop_bar_adds_no_new_information():
    # the stop-bar's own favorable side doesn't exceed what's already banked,
    # so there's nothing ambiguous about it.
    n = 10
    df = _df(n)
    df.loc[1, "low"] = 70.0   # 3R favorable already banked
    df.loc[2, "high"] = 110.0  # stop bar; its low stays at 100 (no new favorable move)
    trades = pd.DataFrame([_trade("short", 0, 100.0, 10.0)])
    out = track_forward_reach(trades, df, mfe_win=100, targets=TARGETS)
    r = out.iloc[0]
    assert r["intrabar_ambiguous"] == False
    assert r["reach_3.0R"] == True


def test_timeout_with_sufficient_data():
    n = 105
    df = _df(n)
    df.loc[1, "low"] = 95.0  # 0.5R favorable only, never near 1R stop
    trades = pd.DataFrame([_trade("short", 0, 100.0, 10.0)])
    out = track_forward_reach(trades, df, mfe_win=100, targets=TARGETS)
    r = out.iloc[0]
    assert r["shadow_resolution"] == "timeout"
    assert r["reach_0.5R"] == True
    assert r["reach_1.0R"] == False
    assert not np.isnan(r["shadow_end_r"])


def test_insufficient_data_preserves_partial_reach_but_nans_the_rest():
    # only 5 forward bars exist (dataset ends), well short of mfe_win=100
    n = 6
    df = _df(n)
    df.loc[1, "low"] = 90.0  # 1R favorable achieved within the available bars
    trades = pd.DataFrame([_trade("short", 0, 100.0, 10.0)])
    out = track_forward_reach(trades, df, mfe_win=100, targets=TARGETS)
    r = out.iloc[0]
    assert r["shadow_resolution"] == "insufficient_data"
    assert r["reach_1.0R"] == True   # genuinely observed within available data
    assert np.isnan(r["reach_1.5R"])  # never observed, but window was truncated -- unknown, not False


def test_zero_forward_bars_all_unknown():
    n = 1  # trade enters on the very last bar
    df = _df(n)
    trades = pd.DataFrame([_trade("short", 0, 100.0, 10.0)])
    out = track_forward_reach(trades, df, mfe_win=100, targets=TARGETS)
    r = out.iloc[0]
    assert r["shadow_resolution"] == "insufficient_data"
    for T in TARGETS:
        assert np.isnan(r[f"reach_{T}R"])


def test_long_direction_mirrors_short():
    n = 10
    df = _df(n)
    df.loc[1, "high"] = 115.0  # 1.5R favorable for a long
    df.loc[2, "low"] = 90.0    # 1R adverse -> stop
    trades = pd.DataFrame([_trade("long", 0, 100.0, 10.0)])
    out = track_forward_reach(trades, df, mfe_win=100, targets=TARGETS)
    r = out.iloc[0]
    assert r["shadow_resolution"] == "stop"
    assert r["reach_1.5R"] == True
    assert r["reach_2.0R"] == False
