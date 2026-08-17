"""Shared raid/reclaim helpers for independent hypothesis strategies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ht_backtest.gates.primitive_cache import SymbolPrimitives, compute_symbol_primitives
from ht_backtest.gates.session_range import run_session_range_engine
from ht_backtest.strategies.base import StrategyContext, TradeCandidate

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SOL = "SOL/USDT:USDT"

RECLAIM_WIN = 3
FAIL_WIN = 20
VOL_LOOKBACK = 50
ASIA_WIDTH_DAYS = 20
SMT_LOOKBACK = 20
TARGET_R = 2.0


def hash_params(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def ensure_primitives(bars: pd.DataFrame, ctx: StrategyContext) -> SymbolPrimitives:
    if ctx.primitives is not None:
        return ctx.primitives
    return compute_symbol_primitives(bars)


def session_range_frame(
    bars: pd.DataFrame,
    prim: SymbolPrimitives,
    *,
    sw_len: int = 5,
    one_raid: bool = True,
    max_liq: int = 8,
) -> pd.DataFrame:
    return run_session_range_engine(
        bars,
        prim.atr,
        prim.sessions,
        prim.day_tags,
        prim.pdh,
        prim.pdl,
        prim.pool_events,
        prim.asia_pools,
        max_liq=max_liq,
        one_raid=one_raid,
        sw_len=sw_len,
    )


@dataclass(frozen=True)
class RaidEvent:
    grab_bar: int
    reclaim_bar: int | None
    direction: str  # "up" | "dn"
    level: float
    extreme: float  # sweep extreme (bar high for up, bar low for dn)
    session: str  # LONDON | NY | OTHER
    first_of_kz: bool
    first_30: bool
    utc_day: int
    kz_hi: float
    kz_lo: float
    grab_volume: float
    vol_median: float
    vol_p40: float


def _session_name(in_london: bool, in_ny: bool) -> str:
    if in_london:
        return "LONDON"
    if in_ny:
        return "NY"
    return "OTHER"


def _in_first_30(in_london: bool, in_ny: bool, hour: int, minute: int) -> bool:
    if in_london and hour == 7 and minute < 30:
        return True
    if in_ny and hour == 12 and minute >= 30:
        return True
    return False


def reclaim_bar(close: np.ndarray, grab_i: int, direction: str, level: float, max_bars: int) -> int | None:
    n = len(close)
    for k in range(1, max_bars + 1):
        j = grab_i + k
        if j >= n:
            return None
        if direction == "up" and close[j] < level:
            return j
        if direction == "dn" and close[j] > level:
            return j
    return None


def collect_raid_events(bars: pd.DataFrame, prim: SymbolPrimitives, sr: pd.DataFrame) -> list[RaidEvent]:
    """Session-range edge raids with optional reclaim within RECLAIM_WIN."""
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    vol = bars["volume"].to_numpy() if "volume" in bars.columns else np.ones(len(bars))
    ts = bars["timestamp"].to_numpy()
    grab_up = sr["grab_up"].to_numpy()
    grab_dn = sr["grab_dn"].to_numpy()
    grab_up_px = sr["grab_up_price"].to_numpy()
    grab_dn_px = sr["grab_dn_price"].to_numpy()
    kz_hi = sr["kz_hi"].to_numpy()
    kz_lo = sr["kz_lo"].to_numpy()
    sessions = prim.sessions
    kz_start = (sessions["london_start"] | sessions["ny_start"]).to_numpy()
    in_london = sessions["in_london"].to_numpy()
    in_ny = sessions["in_ny"].to_numpy()
    hour = sessions["hour_london"].to_numpy()
    minute = pd.to_datetime(ts, unit="ms", utc=True).tz_convert("Europe/London").minute.to_numpy()

    n = len(bars)
    first_raid_of_kz = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not kz_start[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not kz_start[j]:
            j += 1
        for k in range(i, j):
            if grab_up[k] or grab_dn[k]:
                first_raid_of_kz[k] = True
                break
        i = j

    vol_med = pd.Series(vol).rolling(VOL_LOOKBACK, min_periods=10).median().to_numpy()
    vol_p40 = pd.Series(vol).rolling(VOL_LOOKBACK, min_periods=10).quantile(0.40).to_numpy()

    out: list[RaidEvent] = []
    for i in range(n):
        if grab_up[i]:
            direction, level, extreme = "up", float(grab_up_px[i]), float(high[i])
        elif grab_dn[i]:
            direction, level, extreme = "dn", float(grab_dn_px[i]), float(low[i])
        else:
            continue
        rec = reclaim_bar(close, i, direction, level, RECLAIM_WIN)
        out.append(
            RaidEvent(
                grab_bar=i,
                reclaim_bar=rec,
                direction=direction,
                level=level,
                extreme=extreme,
                session=_session_name(bool(in_london[i]), bool(in_ny[i])),
                first_of_kz=bool(first_raid_of_kz[i]),
                first_30=_in_first_30(bool(in_london[i]), bool(in_ny[i]), int(hour[i]), int(minute[i])),
                utc_day=int(ts[i]) // 86_400_000,
                kz_hi=float(kz_hi[i]) if not np.isnan(kz_hi[i]) else float("nan"),
                kz_lo=float(kz_lo[i]) if not np.isnan(kz_lo[i]) else float("nan"),
                grab_volume=float(vol[i]),
                vol_median=float(vol_med[i]) if not np.isnan(vol_med[i]) else float("nan"),
                vol_p40=float(vol_p40[i]) if not np.isnan(vol_p40[i]) else float("nan"),
            )
        )
    return out


def make_reclaim_trade(
    bars: pd.DataFrame,
    ctx: StrategyContext,
    strategy_id: str,
    event: RaidEvent,
    *,
    target_r: float = TARGET_R,
    use_far_edge: bool = True,
) -> TradeCandidate | None:
    """Enter at reclaim close; stop beyond sweep extreme; target far edge or target_r * R."""
    if event.reclaim_bar is None:
        return None
    i = event.reclaim_bar
    entry = float(bars["close"].iloc[i])
    if event.direction == "up":
        direction = "short"
        stop = event.extreme
        if not (stop > entry):
            return None
        risk = stop - entry
        far = event.kz_lo
        if use_far_edge and not np.isnan(far) and far < entry:
            planned = far
            if (entry - planned) < 0.5 * risk:
                planned = entry - target_r * risk
        else:
            planned = entry - target_r * risk
    else:
        direction = "long"
        stop = event.extreme
        if not (stop < entry):
            return None
        risk = entry - stop
        far = event.kz_hi
        if use_far_edge and not np.isnan(far) and far > entry:
            planned = far
            if (planned - entry) < 0.5 * risk:
                planned = entry + target_r * risk
        else:
            planned = entry + target_r * risk
    if risk <= 0 or np.isnan(risk):
        return None
    return TradeCandidate(
        direction=direction,
        entry_bar=i,
        entry_price=entry,
        stop_price=stop,
        risk=risk,
        strategy_id=strategy_id,
        symbol=ctx.symbol,
        planned_target=planned,
        extras={
            "entry_time": int(bars["timestamp"].iloc[i]),
            "grab_bar": event.grab_bar,
            "raid_direction": event.direction,
            "session": event.session,
            "grab_level": event.level,
        },
    )


def ffill_asia_levels(prim: SymbolPrimitives) -> tuple[np.ndarray, np.ndarray]:
    """Carry last completed Asia high/low forward onto subsequent bars."""
    hi = prim.asia_pools["asia_pool_high"].to_numpy(dtype=float).copy()
    lo = prim.asia_pools["asia_pool_low"].to_numpy(dtype=float).copy()
    last_hi, last_lo = np.nan, np.nan
    for i in range(len(hi)):
        if not np.isnan(hi[i]):
            last_hi, last_lo = hi[i], lo[i]
        hi[i], lo[i] = last_hi, last_lo
    return hi, lo


def asia_width_is_tight(asia_hi: np.ndarray, asia_lo: np.ndarray) -> np.ndarray:
    """True on bars whose current (ffilled) Asia width is ≤ 40th pct of last ASIA_WIDTH_DAYS completions."""
    n = len(asia_hi)
    tight = np.zeros(n, dtype=bool)
    # Completion events: where raw asia pool was set (before ffill we need ends)
    # Use change-points in ffilled series as proxy for new Asia freeze.
    widths: list[float] = []
    completion_idx: list[int] = []
    prev = (np.nan, np.nan)
    for i in range(n):
        cur = (asia_hi[i], asia_lo[i])
        if np.isnan(cur[0]) or np.isnan(cur[1]):
            continue
        if cur != prev and (np.isnan(prev[0]) or cur[0] != prev[0] or cur[1] != prev[1]):
            w = cur[0] - cur[1]
            if w > 0:
                widths.append(w)
                completion_idx.append(i)
            prev = cur

    # Map each bar to whether its Asia width is tight vs prior completions
    width_at = {}
    for k, idx in enumerate(completion_idx):
        hist = widths[max(0, k - ASIA_WIDTH_DAYS) : k]  # prior only, no look-ahead
        if len(hist) < 5:
            continue
        p40 = float(np.quantile(hist, 0.40))
        width_at[idx] = widths[k] <= p40

    last = False
    comp_set = {completion_idx[k]: width_at.get(completion_idx[k], False) for k in range(len(completion_idx))}
    for i in range(n):
        if i in comp_set:
            last = comp_set[i]
        tight[i] = last
    return tight


def first_fade_in_session(
    close: np.ndarray,
    in_sess: np.ndarray,
    start_i: int,
    direction: str,
    level: float,
) -> int | None:
    n = len(close)
    i = start_i
    while i < n and not in_sess[i]:
        i += 1
    if i >= n:
        return None
    while i < n and in_sess[i]:
        if direction == "up" and close[i] < level:
            return i
        if direction == "dn" and close[i] > level:
            return i
        i += 1
    return None


def make_fade_trade(
    bars: pd.DataFrame,
    ctx: StrategyContext,
    strategy_id: str,
    *,
    entry_bar: int,
    direction: str,
    level: float,
    extreme: float,
    target_r: float = TARGET_R,
) -> TradeCandidate | None:
    entry = float(bars["close"].iloc[entry_bar])
    if direction == "short":
        stop = max(extreme, level)
        if not (stop > entry):
            return None
        risk = stop - entry
        planned = entry - target_r * risk
    else:
        stop = min(extreme, level)
        if not (stop < entry):
            return None
        risk = entry - stop
        planned = entry + target_r * risk
    if risk <= 0:
        return None
    return TradeCandidate(
        direction=direction,
        entry_bar=entry_bar,
        entry_price=entry,
        stop_price=stop,
        risk=risk,
        strategy_id=strategy_id,
        symbol=ctx.symbol,
        planned_target=planned,
        extras={"entry_time": int(bars["timestamp"].iloc[entry_bar]), "fade_level": level},
    )


def smt_divergences(
    sweeper: pd.DataFrame,
    holder: pd.DataFrame,
    lookback: int = SMT_LOOKBACK,
) -> list[dict[str, Any]]:
    """Bars where sweeper makes a lookback extreme and holder fails to confirm."""
    s_low = sweeper["low"].to_numpy()
    s_high = sweeper["high"].to_numpy()
    h_low = holder["low"].to_numpy()
    h_high = holder["high"].to_numpy()
    n = min(len(sweeper), len(holder))
    out: list[dict[str, Any]] = []
    for i in range(lookback - 1, n):
        window_s_low = s_low[i - lookback + 1 : i + 1]
        window_s_high = s_high[i - lookback + 1 : i + 1]
        if np.any(np.isnan(window_s_low)) or np.any(np.isnan(h_low[i])) or np.any(np.isnan(h_high[i])):
            continue
        if s_low[i] <= np.min(window_s_low) + 1e-12 and h_low[i] > s_low[i]:
            out.append(
                {
                    "bar": i,
                    "side": "low",
                    "sweep_level": float(s_low[i]),
                    "holder_extreme": float(h_low[i]),
                }
            )
        if s_high[i] >= np.max(window_s_high) - 1e-12 and h_high[i] < s_high[i]:
            out.append(
                {
                    "bar": i,
                    "side": "high",
                    "sweep_level": float(s_high[i]),
                    "holder_extreme": float(h_high[i]),
                }
            )
    return out
