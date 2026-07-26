import numpy as np
import pandas as pd

from ht_backtest.data.validator import validate_ohlcv

TF_MS = 15 * 60_000


def _base_df(n=300, start=1_700_000_000_000, price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    ts = start + np.arange(n) * TF_MS
    close = price + np.cumsum(rng.normal(0, 0.05, n))
    open_ = close - rng.normal(0, 0.02, n)
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.05, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.05, n)
    vol = rng.uniform(10, 20, n)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_clean_data_flags_nothing():
    df = _base_df()
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.duplicates) == 0
    assert len(report.gaps) == 0
    assert len(report.impossible_ohlc) == 0
    assert len(report.non_positive_price) == 0
    assert len(report.extreme_jumps) == 0
    assert len(report.outlier_wicks) == 0


def test_duplicate_timestamp_detected():
    df = _base_df()
    dup_row = df.iloc[[10]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.duplicates) == 2
    assert report.total_bars == len(df) - 1


def test_gap_detected():
    df = _base_df()
    df = df.drop(df.index[[50, 51, 52]]).reset_index(drop=True)
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.gaps) == 1
    assert report.gaps.iloc[0]["missing_bars"] == 3
    assert report.missing_bar_count() == 3


def test_impossible_ohlc_detected():
    df = _base_df()
    df.loc[20, "high"] = df.loc[20, "low"] - 1.0  # high < low
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.impossible_ohlc) == 1
    assert report.impossible_ohlc.iloc[0]["timestamp"] == df.loc[20, "timestamp"]


def test_non_positive_price_detected():
    df = _base_df()
    df.loc[15, "low"] = 0.0
    df.loc[16, "close"] = -5.0
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.non_positive_price) == 2


def test_extreme_jump_detected():
    df = _base_df(n=250)
    df.loc[200, "close"] = df.loc[199, "close"] * 1.5  # +50% in one bar
    df.loc[200, "high"] = df.loc[200, "close"] + 0.05
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.extreme_jumps) >= 1
    assert df.loc[200, "timestamp"] in report.extreme_jumps["timestamp"].values


def test_isolated_outlier_wick_detected():
    df = _base_df(n=100)
    spike_idx = 60
    df.loc[spike_idx, "high"] = df.loc[spike_idx, "close"] + 5.0  # ~50-100x normal wick
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.outlier_wicks) >= 1
    assert df.loc[spike_idx, "timestamp"] in report.outlier_wicks["timestamp"].values


def test_dead_volume_run_detected():
    df = _base_df(n=200)
    dead_start, dead_len = 150, 20
    df.loc[dead_start : dead_start + dead_len - 1, "volume"] = 0.0
    last_close = df.loc[dead_start - 1, "close"]
    for col in ("open", "high", "low", "close"):
        df.loc[dead_start : dead_start + dead_len - 1, col] = last_close
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.dead_runs) == 1
    assert report.dead_runs.iloc[0]["bars"] == dead_len
    assert report.dead_bar_count() == dead_len


def test_short_zero_volume_blip_not_flagged_as_dead_run():
    df = _base_df(n=100)
    df.loc[40, "volume"] = 0.0  # single quiet bar, below dead_run_min_bars
    report = validate_ohlcv(df, "TEST/USDT", "15m")
    assert len(report.dead_runs) == 0
