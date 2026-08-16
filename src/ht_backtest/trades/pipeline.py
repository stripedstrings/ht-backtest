"""Wires the gate primitives, session-range engine, and trade state machine
together for a single symbol's OHLCV frame."""

from __future__ import annotations

import time

import pandas as pd

from ht_backtest.gates.primitive_cache import SymbolPrimitives, compute_symbol_primitives
from ht_backtest.gates.session_range import run_session_range_engine
from ht_backtest.profiling import StageTimings
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
    primitives: SymbolPrimitives | None = None,
    timings: StageTimings | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """df must have columns timestamp, open, high, low, close (sorted
    ascending). Returns (trades, session_range_frame).

    If `primitives` is provided, ATR/sessions/swings/etc. are reused instead
    of recomputed — gate/FSM logic is unchanged.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    t_prim = time.perf_counter()
    if primitives is None:
        primitives = compute_symbol_primitives(
            df,
            sw_len=sw_len,
            int_len=int_len,
            eq_tol_mult=eq_tol_mult,
            ctx_bars=ctx_bars,
            d_piv=d_piv,
        )
    prim_s = time.perf_counter() - t_prim

    t_gate = time.perf_counter()
    session_range = run_session_range_engine(
        df,
        primitives.atr,
        primitives.sessions,
        primitives.day_tags,
        primitives.pdh,
        primitives.pdl,
        primitives.pool_events,
        primitives.asia_pools,
        max_liq=max_liq,
        one_raid=one_raid,
        sw_len=sw_len,
    )

    trades = run_trade_state_machine(
        df,
        atr=primitives.atr,
        last_int_hi=primitives.last_int_hi,
        last_int_lo=primitives.last_int_lo,
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
        efficiency_ratio=primitives.efficiency_ratio,
        grab_seq=session_range["grab_seq"],
        sessions=primitives.sessions,
        daily_bias=primitives.daily_bias,
        params=params,
    )
    gate_s = time.perf_counter() - t_gate

    if timings is not None:
        timings.primitives_s += prim_s
        timings.gate_fsm_s += gate_s
        timings.notes["primitives_cache_hit"] = bool(primitives.cache_hit)

    return trades, session_range
