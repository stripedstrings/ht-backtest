"""Wires the gate primitives, session-range engine, and trade state machine
together for a single symbol's OHLCV frame."""

from __future__ import annotations

import pandas as pd

from ht_backtest.gates.primitives import (
    asia_range_pools,
    compute_atr,
    daily_bias_pivots,
    daily_prev_high_low,
    efficiency_ratio,
    pool_swing_events,
    protected_swing_levels,
    session_tags,
    utc_day_tags,
)
from ht_backtest.gates.session_range import run_session_range_engine
from ht_backtest.trades.state_machine import GateParams, run_trade_state_machine


def generate_trades(
    df: pd.DataFrame,
    params: GateParams = GateParams(),
    sw_len: int = 5,
    int_len: int = 2,
    eq_tol_mult: float = 0.10,
    max_liq: int = 8,
    one_raid: bool = True,
    ctx_bars: int = 40,
    d_piv: int = 2,
) -> tuple[list[dict], pd.DataFrame]:
    """df must have columns timestamp, open, high, low, close (sorted
    ascending). Returns (trades, session_range_frame) -- the session-range
    frame is returned too since deliverable 5's tagging needs it."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    atr = compute_atr(df, length=14)
    sessions = session_tags(df["timestamp"])
    day_tags = utc_day_tags(df["timestamp"])
    pdh, pdl = daily_prev_high_low(df)
    pool_events = pool_swing_events(df, atr, sw_len=sw_len, eq_tol_mult=eq_tol_mult)
    asia_pools = asia_range_pools(df, sessions)
    last_int_hi, last_int_lo = protected_swing_levels(df, int_len=int_len)
    er = efficiency_ratio(df, ctx_bars=ctx_bars)
    daily_bias = daily_bias_pivots(df, d_piv=d_piv)

    session_range = run_session_range_engine(
        df, atr, sessions, day_tags, pdh, pdl, pool_events, asia_pools,
        max_liq=max_liq, one_raid=one_raid, sw_len=sw_len,
    )

    trades = run_trade_state_machine(
        df,
        atr=atr,
        last_int_hi=last_int_hi,
        last_int_lo=last_int_lo,
        grab_up=session_range["grab_up"],
        grab_dn=session_range["grab_dn"],
        grab_up_price=session_range["grab_up_price"],
        grab_dn_price=session_range["grab_dn_price"],
        grab_up_type=session_range["grab_up_type"],
        grab_dn_type=session_range["grab_dn_type"],
        grab_up_age=session_range["grab_up_age"],
        grab_dn_age=session_range["grab_dn_age"],
        kz_hi=session_range["kz_hi"],
        kz_lo=session_range["kz_lo"],
        range_width_atr=session_range["range_width_atr"],
        efficiency_ratio=er,
        grab_seq=session_range["grab_seq"],
        sessions=sessions,
        daily_bias=daily_bias,
        params=params,
    )
    return trades, session_range
