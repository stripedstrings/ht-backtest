"""Feature enrichment for the condition library (causal, bar-local)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ht_backtest.data.funding import attach_funding_rate
from ht_backtest.data.htf_4h import attach_4h_for_symbol
from ht_backtest.data.liq import attach_liquidations
from ht_backtest.data.oi import attach_open_interest
from ht_backtest.gates.primitive_cache import SymbolPrimitives, load_or_compute_primitives
from ht_backtest.gates.primitives import LONDON_TZ
from ht_backtest.strategies.hypothesis_helpers import (
    collect_raid_events,
    ffill_asia_levels,
    session_range_frame,
)

VOL_LOOKBACK = 50
ASIA_WIDTH_HIST = 20
ASIA_TIGHT_Q = 0.40
ASIA_WIDE_Q = 0.60


def _asia_width_flags(asia_hi: np.ndarray, asia_lo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tight, wide, defined) using prior ASIA_WIDTH_HIST completions only."""
    n = len(asia_hi)
    tight = np.zeros(n, dtype=bool)
    wide = np.zeros(n, dtype=bool)
    defined = np.zeros(n, dtype=bool)
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

    width_tight: dict[int, bool] = {}
    width_wide: dict[int, bool] = {}
    width_ok: dict[int, bool] = {}
    for k, idx in enumerate(completion_idx):
        hist = widths[max(0, k - ASIA_WIDTH_HIST) : k]
        if len(hist) < 5:
            continue
        p_lo = float(np.quantile(hist, ASIA_TIGHT_Q))
        p_hi = float(np.quantile(hist, ASIA_WIDE_Q))
        width_tight[idx] = widths[k] <= p_lo
        width_wide[idx] = widths[k] >= p_hi
        width_ok[idx] = True

    last_t = last_w = last_ok = False
    for i in range(n):
        if i in width_ok:
            last_t = width_tight.get(i, False)
            last_w = width_wide.get(i, False)
            last_ok = True
        tight[i] = last_t
        wide[i] = last_w
        defined[i] = last_ok
    return tight, wide, defined


def _prior_session_features(
    bars: pd.DataFrame, prim: SymbolPrimitives, sr: pd.DataFrame
) -> pd.DataFrame:
    """Build london_raided_* and prior_session_* columns (NaN = undefined)."""
    n = len(bars)
    london_hi = np.full(n, np.nan)
    london_lo = np.full(n, np.nan)
    prior_same = np.full(n, np.nan)
    prior_opp = np.full(n, np.nan)

    events = collect_raid_events(bars, prim, sr)
    sessions = prim.sessions
    london_start = sessions["london_start"].to_numpy()
    in_london = sessions["in_london"].to_numpy()

    # Within each London instance: accumulate raid+reclaim; at London exit freeze True/False.
    saw_hi = False
    saw_lo = False
    in_lon = False
    frozen_hi = np.nan
    frozen_lo = np.nan
    reclaim_up = {
        e.reclaim_bar
        for e in events
        if e.session == "LONDON" and e.direction == "up" and e.reclaim_bar is not None
    }
    reclaim_dn = {
        e.reclaim_bar
        for e in events
        if e.session == "LONDON" and e.direction == "dn" and e.reclaim_bar is not None
    }

    for i in range(n):
        if london_start[i]:
            saw_hi = False
            saw_lo = False
            in_lon = True
            frozen_hi = np.nan
            frozen_lo = np.nan
        if in_lon:
            if i in reclaim_up:
                saw_hi = True
            if i in reclaim_dn:
                saw_lo = True
            if not in_london[i]:
                frozen_hi = 1.0 if saw_hi else 0.0
                frozen_lo = 1.0 if saw_lo else 0.0
                in_lon = False
        if in_lon:
            london_hi[i] = np.nan
            london_lo[i] = np.nan
        else:
            london_hi[i] = frozen_hi
            london_lo[i] = frozen_lo

    kz_start = (sessions["london_start"] | sessions["ny_start"]).to_numpy()
    kz_id = np.full(n, -1, dtype=int)
    kid = -1
    for i in range(n):
        if kz_start[i]:
            kid += 1
        if bool(sessions["in_killzone"].iloc[i]):
            kz_id[i] = kid

    first_dir_by_kz: dict[int, str] = {}
    known_bar_by_kz: dict[int, int] = {}
    for e in events:
        if e.reclaim_bar is None:
            continue
        k = int(kz_id[e.grab_bar]) if 0 <= e.grab_bar < n else -1
        if k < 0:
            continue
        if k not in first_dir_by_kz:
            first_dir_by_kz[k] = e.direction
            known_bar_by_kz[k] = int(e.reclaim_bar)

    for i in range(n):
        k = int(kz_id[i])
        if k < 0 or k not in first_dir_by_kz or known_bar_by_kz[k] > i:
            continue
        prev = k - 1
        while prev >= 0 and prev not in first_dir_by_kz:
            prev -= 1
        if prev < 0 or known_bar_by_kz[prev] > i:
            continue
        same = first_dir_by_kz[k] == first_dir_by_kz[prev]
        prior_same[i] = 1.0 if same else 0.0
        prior_opp[i] = 0.0 if same else 1.0

    return pd.DataFrame(
        {
            "feat_london_raided_high": london_hi,
            "feat_london_raided_low": london_lo,
            "feat_prior_session_same": prior_same,
            "feat_prior_session_opp": prior_opp,
        },
        index=bars.index,
    )


def enrich_condition_features(
    bars: pd.DataFrame,
    symbol: str,
    *,
    prim: SymbolPrimitives | None = None,
    cache_dir: str | Path = "data/raw",
    funding_dir: str | Path = "data/funding",
    oi_dir: str | Path = "data/oi",
    liq_dir: str | Path = "data/liq",
    primitives_cache_dir: str | Path = "data/cache/primitives",
    use_primitive_cache: bool = True,
    attach_funding: bool = True,
    attach_htf: bool = True,
    attach_oi: bool = True,
    attach_liq: bool = True,
) -> pd.DataFrame:
    """Return a copy of bars with columns required by the condition library."""
    out = bars.copy()
    if attach_funding:
        out = attach_funding_rate(out, symbol, funding_dir=funding_dir)
    if attach_oi:
        out = attach_open_interest(out, symbol, oi_dir=oi_dir)
    if attach_liq:
        out = attach_liquidations(out, symbol, liq_dir=liq_dir)
    if attach_htf:
        out = attach_4h_for_symbol(out, symbol, cache_dir=cache_dir)

    if prim is None:
        prim = load_or_compute_primitives(
            out,
            exchange_id="binanceusdm",
            symbol=symbol,
            timeframe="15m",
            cache_dir=primitives_cache_dir,
            use_cache=use_primitive_cache,
        )

    sessions = prim.sessions
    local = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.tz_convert(LONDON_TZ)
    minute_of_day = local.dt.hour * 60 + local.dt.minute

    out["feat_london_session"] = sessions["in_london"].to_numpy()
    out["feat_ny_session"] = sessions["in_ny"].to_numpy()
    out["feat_asia_session"] = sessions["in_asia"].to_numpy()
    out["feat_london_open_30m"] = (minute_of_day >= 7 * 60) & (minute_of_day < 7 * 60 + 30)
    out["feat_ny_open_30m"] = (minute_of_day >= 12 * 60 + 30) & (minute_of_day < 13 * 60)

    vol = out["volume"].astype(float)
    prior = vol.shift(1)
    roll = prior.rolling(VOL_LOOKBACK, min_periods=20)
    p70 = roll.quantile(0.70)
    p30 = roll.quantile(0.30)
    med = roll.median()
    out["feat_volume_high"] = vol >= p70
    out["feat_volume_low"] = vol <= p30
    out["feat_volume_spike"] = vol >= (2.0 * med)
    out["feat_volume_defined"] = p70.notna() & vol.notna()

    asia_hi, asia_lo = ffill_asia_levels(prim)
    mid = (asia_hi + asia_lo) / 2.0
    open_px = out["open"].astype(float).to_numpy()
    out["feat_asia_hi"] = asia_hi
    out["feat_asia_lo"] = asia_lo
    out["feat_asia_mid"] = mid
    out["feat_price_above_asia_mid"] = open_px > mid
    out["feat_price_below_asia_mid"] = open_px < mid
    out["feat_asia_mid_defined"] = ~np.isnan(mid)

    tight, wide, asia_w_def = _asia_width_flags(asia_hi, asia_lo)
    out["feat_asia_range_tight"] = tight
    out["feat_asia_range_wide"] = wide
    out["feat_asia_range_defined"] = asia_w_def

    # HTF already attached as htf_4h_*
    out["feat_above_4h_ema20"] = out["open"].astype(float) > out["htf_4h_ema20"]
    out["feat_below_4h_ema20"] = out["open"].astype(float) < out["htf_4h_ema20"]
    out["feat_4h_ema_defined"] = out["htf_4h_ema20"].notna()
    out["feat_4h_hh_hl"] = out["htf_4h_hh_hl"].fillna(False).astype(bool)
    out["feat_4h_lh_ll"] = out["htf_4h_lh_ll"].fillna(False).astype(bool)
    # Structure defined once we have an ema (enough history); swings may still be False
    out["feat_4h_structure_defined"] = out["htf_4h_ema20"].notna()

    sr = session_range_frame(out, prim)
    prior_df = _prior_session_features(out, prim, sr)
    for c in prior_df.columns:
        out[c] = prior_df[c].to_numpy()

    return out
