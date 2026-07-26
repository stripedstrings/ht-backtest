import numpy as np
import pandas as pd

from ht_backtest.trades.state_machine import GateParams, run_trade_state_machine


def _series(vals):
    return pd.Series(vals, dtype=float)


def _bool_series(vals):
    return pd.Series(vals, dtype=bool)


def _const_sessions(n, label="LONDON", hour=8):
    return pd.DataFrame(
        {
            "in_ny": [label == "NY"] * n,
            "in_london": [label == "LONDON"] * n,
            "in_asia": [label == "ASIA"] * n,
            "hour_london": [hour] * n,
        }
    )


def _const_daily_bias(n, bias_bull, premium):
    return pd.DataFrame(
        {
            "bias_bull": [bias_bull] * n,
            "bias_bear": [not bias_bull] * n,
            "premium": [premium] * n,
        }
    )


def test_full_short_trade_end_to_end():
    # bar: open, high, low, close
    ohlc = [
        (100, 102, 99, 101),   # 0: grab bar
        (110, 115, 90, 112),   # 1: new high -> be.hi=115, be.body=112; no confirm yet
        (105, 108, 104, 105),  # 2: close<110 -> confirm; wick=3; run_lo=104; be.bar=2
        (100, 100, 95, 95),    # 3: no MSS yet
        (90, 88, 84, 85),      # 4: close<90 -> MSS; run_lo=84; FVG found -> retest state
        (85, 85, 83, 84),      # 5: no retest yet (high<88)
        (84, 90, 83, 89),      # 6: high>=88 -> entry at 88
        (89, 90, 95, 91),      # 7: no stop/target yet
        (90, 92, 78, 79),      # 8: low<=80 -> target hit
    ]
    n = len(ohlc)
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.RangeIndex(n) * 900_000

    atr = _series([1.0] * n)
    last_int_hi = _series([np.nan] * n)
    last_int_lo = _series([90.0] * n)
    grab_up = _bool_series([True] + [False] * (n - 1))
    grab_dn = _bool_series([False] * n)
    grab_up_price = _series([110.0] * n)
    grab_dn_price = _series([np.nan] * n)
    grab_up_type = _series([0] * n)
    grab_dn_type = _series([-1] * n)
    grab_up_age = _series([5] * n)
    grab_dn_age = _series([0] * n)
    kz_hi = _series([np.nan] * n)
    kz_lo = _series([80.0] * n)
    range_width_atr = _series([2.0] * n)
    efficiency_ratio = _series([0.5] * n)
    grab_seq = _series([1] * n)
    sessions = _const_sessions(n, "LONDON", 8)
    daily_bias = _const_daily_bias(n, bias_bull=False, premium=True)  # bear bias + premium favors the short

    trades = run_trade_state_machine(
        df, atr, last_int_hi, last_int_lo,
        grab_up, grab_dn, grab_up_price, grab_dn_price,
        grab_up_type, grab_dn_type, grab_up_age, grab_dn_age,
        kz_hi, kz_lo, range_width_atr, efficiency_ratio, grab_seq,
        sessions, daily_bias, params=GateParams(),
    )

    assert len(trades) == 1
    t = trades[0]
    assert t["direction"] == "short"
    assert t["grab_bar"] == 0
    assert t["grab_price"] == 110.0
    assert t["mss_bar"] == 4
    assert t["protected_level"] == 90.0
    assert t["leg_high"] == 115.0
    assert t["leg_low"] == 84.0
    assert t["fvg_top"] == 104.0
    assert t["fvg_bottom"] == 88.0
    assert abs(t["fvg_retrace_depth"] - 12 / 31) < 1e-9
    assert t["entry_bar"] == 6
    assert t["entry_price"] == 88.0
    assert t["stop_price"] == 115.0
    assert t["target_price"] == 80.0
    assert abs(t["planned_rr"] - 8 / 27) < 1e-9
    assert t["exit_bar"] == 8
    assert t["exit_reason"] == "target"
    assert abs(t["r_multiple"] - 8 / 27) < 1e-9

    # tag-raw-value capture
    assert t["range_width_atr"] == 2.0
    assert np.isnan(t["efficiency_ratio"])  # grab is at bar 0 -- no prior bar to sample erV[1] from
    assert t["grab_seq"] == 1
    assert t["first_grab_of_day"] is True
    assert abs(t["wick_atr"] - 3.0) < 1e-9  # (be.hi=115 - be.body=112)/atr(1.0)
    assert t["body_atr"] == abs(85 - 90) / 1.0  # bar4 open=90,close=85
    assert t["imbalance_count"] == 1
    assert t["stacked_imbalance"] is False
    assert t["valid_ob"] is False
    assert np.isnan(t["coil_atr"])  # insufficient history before the leg
    assert t["ote"] is False  # 0.387 retrace not in [0.62, 0.79)
    assert t["session"] == "LONDON"
    assert t["hour_london"] == 8
    assert t["with_daily_bias"] is True
    assert t["right_premium_discount"] is True


def test_full_long_trade_end_to_end():
    ohlc = [
        (100, 101, 98, 99),     # 0: grab bar
        (90, 92, 85, 88),       # 1: new low -> bu.lo=85, bu.body=88; no confirm yet
        (95, 96, 95, 95),       # 2: close>90 -> confirm; wick=3; run_hi=96; bu.bar=2
        (100, 100, 90, 105),    # 3: no MSS yet
        (110, 112, 104, 115),   # 4: close>110 -> MSS; run_hi=112; FVG found -> retest state
        (108, 106, 106, 107),   # 5: no retest yet (low>104)
        (105, 108, 100, 103),   # 6: low<=104 -> entry at 104
        (103, 110, 95, 108),    # 7: no stop/target yet
        (108, 132, 100, 130),   # 8: high>=130 -> target hit
    ]
    n = len(ohlc)
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.RangeIndex(n) * 900_000

    atr = _series([1.0] * n)
    last_int_hi = _series([110.0] * n)
    last_int_lo = _series([np.nan] * n)
    grab_up = _bool_series([False] * n)
    grab_dn = _bool_series([True] + [False] * (n - 1))
    grab_up_price = _series([np.nan] * n)
    grab_dn_price = _series([90.0] * n)
    grab_up_type = _series([-1] * n)
    grab_dn_type = _series([0] * n)
    grab_up_age = _series([0] * n)
    grab_dn_age = _series([5] * n)
    kz_hi = _series([130.0] * n)
    kz_lo = _series([np.nan] * n)
    range_width_atr = _series([3.0] * n)
    efficiency_ratio = _series([0.4] * n)
    grab_seq = _series([2] * n)
    sessions = _const_sessions(n, "NY", 13)
    daily_bias = _const_daily_bias(n, bias_bull=True, premium=False)  # bull bias + discount favors the long

    trades = run_trade_state_machine(
        df, atr, last_int_hi, last_int_lo,
        grab_up, grab_dn, grab_up_price, grab_dn_price,
        grab_up_type, grab_dn_type, grab_up_age, grab_dn_age,
        kz_hi, kz_lo, range_width_atr, efficiency_ratio, grab_seq,
        sessions, daily_bias, params=GateParams(),
    )

    assert len(trades) == 1
    t = trades[0]
    assert t["direction"] == "long"
    assert t["grab_bar"] == 0
    assert t["grab_price"] == 90.0
    assert t["mss_bar"] == 4
    assert t["protected_level"] == 110.0
    assert t["leg_high"] == 112.0
    assert t["leg_low"] == 85.0
    assert t["fvg_top"] == 104.0
    assert t["fvg_bottom"] == 96.0
    assert abs(t["fvg_retrace_depth"] - 12 / 27) < 1e-9
    assert t["entry_bar"] == 6
    assert t["entry_price"] == 104.0
    assert t["stop_price"] == 85.0
    assert t["target_price"] == 130.0
    assert abs(t["planned_rr"] - 26 / 19) < 1e-9
    assert t["exit_bar"] == 8
    assert t["exit_reason"] == "target"
    assert abs(t["r_multiple"] - 26 / 19) < 1e-9

    assert t["range_width_atr"] == 3.0
    assert np.isnan(t["efficiency_ratio"])  # grab is at bar 0 -- no prior bar to sample erV[1] from
    assert t["grab_seq"] == 2
    assert t["first_grab_of_day"] is False
    assert abs(t["wick_atr"] - 3.0) < 1e-9  # (bu.body=88 - bu.lo=85)/atr(1.0)
    assert t["body_atr"] == abs(115 - 110) / 1.0  # bar4 open=110,close=115
    assert t["imbalance_count"] == 1
    assert t["stacked_imbalance"] is False
    assert t["valid_ob"] is False
    assert np.isnan(t["coil_atr"])
    assert t["ote"] is False  # 0.444 retrace not in [0.62, 0.79)
    assert t["session"] == "NY"
    assert t["hour_london"] == 13
    assert t["with_daily_bias"] is True
    assert t["right_premium_discount"] is True


def test_second_grab_ignored_while_short_setup_in_progress():
    ohlc = [
        (100, 102, 99, 101),
        (110, 115, 90, 112),
        (105, 108, 104, 105),
        (100, 100, 95, 95),
        (90, 88, 84, 85),
        (85, 85, 83, 84),
        (84, 90, 83, 89),
        (89, 90, 95, 91),
        (90, 92, 78, 79),
    ]
    n = len(ohlc)
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.RangeIndex(n) * 900_000

    atr = _series([1.0] * n)
    last_int_hi = _series([np.nan] * n)
    last_int_lo = _series([90.0] * n)
    # a second grab_up fires at bar 1, while the first setup is already armed
    grab_up = _bool_series([True, True] + [False] * (n - 2))
    grab_dn = _bool_series([False] * n)
    grab_up_price = _series([110.0, 999.0] + [np.nan] * (n - 2))
    grab_dn_price = _series([np.nan] * n)
    grab_up_type = _series([0] * n)
    grab_dn_type = _series([-1] * n)
    grab_up_age = _series([5] * n)
    grab_dn_age = _series([0] * n)
    kz_hi = _series([np.nan] * n)
    kz_lo = _series([80.0] * n)
    range_width_atr = _series([2.0] * n)
    efficiency_ratio = _series([0.5] * n)
    grab_seq = _series([1] * n)
    sessions = _const_sessions(n, "LONDON", 8)
    daily_bias = _const_daily_bias(n, bias_bull=False, premium=True)

    trades = run_trade_state_machine(
        df, atr, last_int_hi, last_int_lo,
        grab_up, grab_dn, grab_up_price, grab_dn_price,
        grab_up_type, grab_dn_type, grab_up_age, grab_dn_age,
        kz_hi, kz_lo, range_width_atr, efficiency_ratio, grab_seq,
        sessions, daily_bias, params=GateParams(),
    )
    assert len(trades) == 1
    assert trades[0]["grab_price"] == 110.0  # the second grab (999.0) was ignored


def test_stacked_imbalance_counts_multiple_gaps():
    # extend the leg so more than one bearish FVG exists inside it.
    ohlc = [
        (100, 102, 99, 101),   # 0: grab bar
        (110, 130, 90, 112),   # 1: high update -> be.hi=130
        (105, 108, 120, 105),  # 2: confirm close<110; run_lo=120; be.bar=2
        (100, 100, 95, 95),    # 3
        (95, 95, 60, 92),      # 4
        (90, 88, 40, 85),      # 5: close<90 MSS; run_lo=min(...,40)=40
    ]
    n = len(ohlc)
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.RangeIndex(n) * 900_000

    atr = _series([1.0] * n)
    last_int_hi = _series([np.nan] * n)
    last_int_lo = _series([90.0] * n)
    grab_up = _bool_series([True] + [False] * (n - 1))
    grab_dn = _bool_series([False] * n)
    grab_up_price = _series([110.0] * n)
    grab_dn_price = _series([np.nan] * n)
    grab_up_type = _series([0] * n)
    grab_dn_type = _series([-1] * n)
    grab_up_age = _series([5] * n)
    grab_dn_age = _series([0] * n)
    kz_hi = _series([np.nan] * n)
    kz_lo = _series([10.0] * n)
    range_width_atr = _series([2.0] * n)
    efficiency_ratio = _series([0.5] * n)
    grab_seq = _series([1] * n)
    sessions = _const_sessions(n, "LONDON", 8)
    daily_bias = _const_daily_bias(n, bias_bull=False, premium=True)

    # at i=5 (MSS bar), be.bar=2, leg_len=min(5-2+2,60)=5
    # j=0: t=low[3]=95, b=high[5]=88 -> 95>88 valid
    # j=1: t=low[2]=120, b=high[4]=95 -> 120>95 valid
    # j=2: t=low[1]=90, b=high[3]=100 -> 90>100 invalid
    # j=3: t=low[0]=99, b=high[2]=108 -> invalid
    trades = run_trade_state_machine(
        df, atr, last_int_hi, last_int_lo,
        grab_up, grab_dn, grab_up_price, grab_dn_price,
        grab_up_type, grab_dn_type, grab_up_age, grab_dn_age,
        kz_hi, kz_lo, range_width_atr, efficiency_ratio, grab_seq,
        sessions, daily_bias, params=GateParams(),
    )
    # setup should reach state 3 (retest) with 2 stacked imbalances recorded;
    # we don't need it to fill to inspect the count, so check via a longer run
    # isn't necessary here -- but since it never retests in this short frame,
    # no trade is emitted. Assert no crash and, separately, unit-test the
    # finder function directly for the count.
    from ht_backtest.trades.state_machine import _find_bear_fvg
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    t, b, d, n_found, j = _find_bear_fvg(low, high, i=5, rng_lo=40.0, rng=130 - 40, leg_len=5)
    assert n_found == 2
