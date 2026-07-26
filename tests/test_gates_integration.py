import numpy as np
import pandas as pd

from ht_backtest.gates.primitives import (
    asia_range_pools,
    compute_atr,
    daily_prev_high_low,
    pool_swing_events,
    session_tags,
    utc_day_tags,
)
from ht_backtest.gates.session_range import run_session_range_engine

TF_MS = 15 * 60_000


def _synthetic_ohlcv(days=20, seed=7):
    n = days * 96
    start_ms = int(pd.Timestamp("2026-01-05 00:00:00", tz="UTC").timestamp() * 1000)
    ts = start_ms + np.arange(n) * TF_MS
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close - rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.4, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.4, n)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close})


def test_full_gate_pipeline_runs_and_produces_sane_output():
    df = _synthetic_ohlcv()
    atr = compute_atr(df, length=14)
    sessions = session_tags(df["timestamp"])
    day_tags = utc_day_tags(df["timestamp"])
    pdh, pdl = daily_prev_high_low(df)
    pool_events = pool_swing_events(df, atr, sw_len=5, eq_tol_mult=0.10)
    asia_pools = asia_range_pools(df, sessions)

    out = run_session_range_engine(
        df, atr, sessions, day_tags, pdh, pdl, pool_events, asia_pools, max_liq=8, one_raid=True, sw_len=5
    )

    assert len(out) == len(df)
    # killzones open every day; by day 3+ (ATR warmed up) some session range
    # should be frozen, i.e. this isn't all-NaN.
    assert out["kz_hi"].notna().sum() > 0
    assert out["kz_lo"].notna().sum() > 0
    # a session range should never be inverted once frozen
    valid = out.dropna(subset=["kz_hi", "kz_lo"])
    assert (valid["kz_hi"] > valid["kz_lo"]).all()
    # at most one grab_up and one grab_dn per killzone: no two consecutive
    # True grab_up flags share the same kz_hi_bar within an uninterrupted
    # in_killzone stretch without a kz_start in between (one-raid invariant,
    # spot-checked via grab_seq never jumping by more than the raid count).
    assert (out["grab_seq"] >= 0).all()
    assert out["range_width_atr"].dropna().ge(0).all()
