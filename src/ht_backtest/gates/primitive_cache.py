"""Shared per-symbol primitive bundle (ATR, sessions, swings, PDH/PDL, …).

Computed once per (exchange, symbol, timeframe, param-hash) and reused across
strategies. Does not change gate logic — only avoids recomputing identical
inputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

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

DEFAULT_PRIM_PARAMS = {
    "atr_len": 14,
    "sw_len": 5,
    "int_len": 2,
    "eq_tol_mult": 0.10,
    "ctx_bars": 40,
    "d_piv": 2,
}


def primitives_param_hash(params: dict | None = None) -> str:
    p = {**DEFAULT_PRIM_PARAMS, **(params or {})}
    blob = json.dumps(p, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class SymbolPrimitives:
    atr: pd.Series
    sessions: pd.DataFrame
    day_tags: pd.DataFrame
    pdh: pd.Series
    pdl: pd.Series
    pool_events: pd.DataFrame
    asia_pools: pd.DataFrame
    last_int_hi: pd.Series
    last_int_lo: pd.Series
    efficiency_ratio: pd.Series
    daily_bias: pd.DataFrame
    param_hash: str
    compute_s: float = 0.0
    cache_hit: bool = False


def compute_symbol_primitives(
    df: pd.DataFrame,
    *,
    atr_len: int = 14,
    sw_len: int = 5,
    int_len: int = 2,
    eq_tol_mult: float = 0.10,
    ctx_bars: int = 40,
    d_piv: int = 2,
) -> SymbolPrimitives:
    t0 = time.perf_counter()
    atr = compute_atr(df, length=atr_len)
    sessions = session_tags(df["timestamp"])
    day_tags = utc_day_tags(df["timestamp"])
    pdh, pdl = daily_prev_high_low(df)
    pool_events = pool_swing_events(df, atr, sw_len=sw_len, eq_tol_mult=eq_tol_mult)
    asia_pools = asia_range_pools(df, sessions)
    last_int_hi, last_int_lo = protected_swing_levels(df, int_len=int_len)
    er = efficiency_ratio(df, ctx_bars=ctx_bars)
    daily_bias = daily_bias_pivots(df, d_piv=d_piv)
    ph = primitives_param_hash(
        {
            "atr_len": atr_len,
            "sw_len": sw_len,
            "int_len": int_len,
            "eq_tol_mult": eq_tol_mult,
            "ctx_bars": ctx_bars,
            "d_piv": d_piv,
        }
    )
    return SymbolPrimitives(
        atr=atr,
        sessions=sessions,
        day_tags=day_tags,
        pdh=pdh,
        pdl=pdl,
        pool_events=pool_events,
        asia_pools=asia_pools,
        last_int_hi=last_int_hi,
        last_int_lo=last_int_lo,
        efficiency_ratio=er,
        daily_bias=daily_bias,
        param_hash=ph,
        compute_s=time.perf_counter() - t0,
        cache_hit=False,
    )


def _cache_path(
    cache_dir: str | Path,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    param_hash: str,
    n_bars: int,
    first_ts: int,
    last_ts: int,
) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    name = f"{n_bars}_{first_ts}_{last_ts}_{param_hash}.pkl"
    return Path(cache_dir) / exchange_id / safe / timeframe / name


def load_or_compute_primitives(
    df: pd.DataFrame,
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    cache_dir: str | Path = "data/cache/primitives",
    use_cache: bool = True,
    **prim_kwargs,
) -> SymbolPrimitives:
    ph = primitives_param_hash({**DEFAULT_PRIM_PARAMS, **prim_kwargs})
    if df.empty:
        return compute_symbol_primitives(df, **{k: prim_kwargs.get(k, DEFAULT_PRIM_PARAMS[k]) for k in DEFAULT_PRIM_PARAMS})

    path = _cache_path(
        cache_dir,
        exchange_id,
        symbol,
        timeframe,
        ph,
        len(df),
        int(df["timestamp"].iloc[0]),
        int(df["timestamp"].iloc[-1]),
    )
    if use_cache and path.exists():
        t0 = time.perf_counter()
        obj = pd.read_pickle(path)
        obj.cache_hit = True
        obj.compute_s = time.perf_counter() - t0
        return obj

    kwargs = {k: prim_kwargs.get(k, DEFAULT_PRIM_PARAMS[k]) for k in DEFAULT_PRIM_PARAMS}
    prim = compute_symbol_primitives(df, **kwargs)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        # store without mutating caller's view of compute_s semantics
        pd.to_pickle(prim, path)
    return prim
