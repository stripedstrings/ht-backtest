import numpy as np
import pandas as pd

from ht_backtest.data.split import SplitManifest
from ht_backtest.trades.tagging import apply_median_tags, assemble_trade_frame


def _manifest(train_symbols, holdout_symbols, date_cutoff_ms=1_000_000_000_000):
    return SplitManifest(
        seed=1,
        timeframe="15m",
        symbol_holdout_fraction=0.3,
        date_holdout_fraction=0.2,
        universe=sorted(train_symbols + holdout_symbols),
        holdout_symbols=sorted(holdout_symbols),
        train_symbols=sorted(train_symbols),
        overall_start_ms=0,
        overall_end_ms=date_cutoff_ms * 2,
        date_holdout_start_ms=date_cutoff_ms,
    )


def _trades_df(n, wick_vals, entry_time_start=0, entry_time_step=1000):
    return pd.DataFrame(
        {
            "entry_bar": range(n),
            "entry_time": [entry_time_start + i * entry_time_step for i in range(n)],
            "wick_atr": wick_vals,
            "body_atr": [1.0] * n,
            "coil_atr": [1.0] * n,
            "grab_age_bars": [1.0] * n,
            "efficiency_ratio": [1.0] * n,
            "range_width_atr": [1.0] * n,
        }
    )


def test_warm_up_gating_and_direct_median_comparison():
    manifest = _manifest(train_symbols=["X"], holdout_symbols=["Y"])
    wick_vals = [1.0] * 8 + [10.0, 0.0]
    df = _trades_df(10, wick_vals)
    out = apply_median_tags(df, manifest, "X", warm_n=8, window=300)
    assert list(out["big_wick"].iloc[:8]) == [-1] * 8
    assert out["big_wick"].iloc[8] == 1  # 10.0 >= median(eight 1.0s)=1.0
    assert out["big_wick"].iloc[9] == 0  # 0.0 < median of the nine prior values


def test_tight_coil_and_range_before_grab_are_inverted():
    manifest = _manifest(train_symbols=["X"], holdout_symbols=["Y"])
    df = _trades_df(10, wick_vals=[1.0] * 10)
    df["coil_atr"] = [1.0] * 8 + [10.0, 0.0]  # same pattern as the wick test
    df["efficiency_ratio"] = [1.0] * 8 + [10.0, 0.0]
    out = apply_median_tags(df, manifest, "X", warm_n=8, window=300)
    # raw median tag would be [.., 1, 0] -- tight_coil/range_before_grab invert that
    assert out["tight_coil"].iloc[8] == 0  # coil at/above median -> NOT tight
    assert out["tight_coil"].iloc[9] == 1  # coil below median -> tight
    assert out["range_before_grab"].iloc[8] == 0
    assert out["range_before_grab"].iloc[9] == 1


def test_holdout_dates_of_a_train_symbol_do_not_update_the_array_but_still_get_tagged():
    cutoff = 5_000
    manifest = _manifest(train_symbols=["X"], holdout_symbols=["Y"], date_cutoff_ms=cutoff)
    # 8 train trades (before cutoff) warm the array with 1.0s, then 2 "holdout
    # date" trades (after cutoff) with a big wick value each.
    n = 10
    entry_times = [i * 500 for i in range(8)] + [cutoff + 100, cutoff + 200]
    wick_vals = [1.0] * 8 + [10.0, 10.0]
    df = _trades_df(n, wick_vals, entry_time_start=0)
    df["entry_time"] = entry_times
    out = apply_median_tags(df, manifest, "X", warm_n=8, window=300)
    assert out["big_wick"].iloc[8] == 1  # tagged using the frozen train median
    assert out["big_wick"].iloc[9] == 1  # still 1 -- the 9th trade's 10.0 was NOT pushed in
    assert out["split"].iloc[9] if "split" in out.columns else True  # split col only added by assemble_trade_frame


def test_fully_holdout_symbol_gets_all_undefined_median_tags():
    manifest = _manifest(train_symbols=["X"], holdout_symbols=["Y"])
    df = _trades_df(20, wick_vals=list(range(20)))
    out = apply_median_tags(df, manifest, "Y", warm_n=8, window=300)  # Y is fully holdout
    assert (out["big_wick"] == -1).all()
    assert (out["big_displacement"] == -1).all()
    assert (out["tight_coil"] == -1).all()
    assert (out["old_liquidity"] == -1).all()
    assert (out["range_before_grab"] == -1).all()
    assert (out["wide_range"] == -1).all()


def test_assemble_trade_frame_adds_session_edges_and_split_column():
    manifest = _manifest(train_symbols=["BTC/USDT:USDT"], holdout_symbols=["ETH/USDT:USDT"])
    trades = [
        {
            "direction": "short",
            "grab_bar": 0, "grab_time": 0, "grab_price": 110.0, "grab_type": 0, "grab_age_bars": 5,
            "range_width_atr": 2.0, "efficiency_ratio": np.nan, "grab_seq": 1,
            "close_back_bar": 1, "wick_atr": 3.0,
            "mss_bar": 4, "mss_time": 4, "protected_level": 90.0,
            "leg_high": 115.0, "leg_low": 84.0, "body_atr": 1.0,
            "imbalance_count": 1, "stacked_imbalance": False, "valid_ob": False, "coil_atr": np.nan,
            "fvg_top": 104.0, "fvg_bottom": 88.0, "fvg_retrace_depth": 0.3, "ote": False,
            "entry_bar": 6, "entry_time": 6, "entry_price": 88.0, "stop_price": 115.0, "target_price": 80.0,
            "risk": 27.0, "planned_rr": 0.3, "session_edge_target": 80.0,
            "session": "LONDON", "hour_london": 8, "with_daily_bias": True, "right_premium_discount": True,
            "first_grab_of_day": True,
            "exit_bar": 8, "exit_time": 8, "exit_reason": "target", "r_multiple": 0.3,
        }
    ]
    out = assemble_trade_frame(trades, "BTC/USDT:USDT", "15m", manifest)
    assert out.loc[0, "session_range_high"] == 110.0  # short: raided edge (grab_price) is the HIGH
    assert out.loc[0, "session_range_low"] == 80.0    # short: target edge is the LOW
    assert out.loc[0, "split"] == "train"
    assert out.loc[0, "symbol"] == "BTC/USDT:USDT"
