"""The v1.0 gate: freeze the nearest unswept pool above/below at killzone
open, and only a raid of THAT range can arm a setup. Pool bookkeeping (swing
fractals, equal H/L, Asia range, prev-day H/L) is inherently sequential --
pools are added, capped at maxLiq per side, and pruned the moment price
sweeps them -- so this is a direct bar-by-bar port of the Pine state rather
than a vectorized one, exactly mirroring live execution order per bar:
add this bar's new pools -> freeze range at killzone open -> prune any pool
price sweeps through -> apply the session-range gate override.

Only rangeMode="Freeze at session open" (the Pine default) is implemented.
"Develop with session" is out of scope until a caller actually needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SWING, EQUAL, ASIA, PREVDAY = 0, 1, 2, 3


@dataclass
class _Pool:
    price: float
    origin_bar: int
    kind: int


@dataclass
class _PoolBook:
    max_liq: int
    pools: list = field(default_factory=list)

    def add(self, price: float, origin_bar: int, kind: int) -> None:
        self.pools.append(_Pool(price, origin_bar, kind))
        if len(self.pools) > self.max_liq:
            self.pools.pop(0)

    def nearest_above(self, ref: float) -> _Pool | None:
        candidates = [p for p in self.pools if p.price > ref]
        return min(candidates, key=lambda p: p.price) if candidates else None

    def nearest_below(self, ref: float) -> _Pool | None:
        candidates = [p for p in self.pools if p.price < ref]
        return max(candidates, key=lambda p: p.price) if candidates else None

    def prune_swept_above(self, high: float) -> None:
        self.pools = [p for p in self.pools if not (p.price < high)]

    def prune_swept_below(self, low: float) -> None:
        self.pools = [p for p in self.pools if not (p.price > low)]


def run_session_range_engine(
    df: pd.DataFrame,
    atr: pd.Series,
    sessions: pd.DataFrame,
    day_tags: pd.DataFrame,
    pdh: pd.Series,
    pdl: pd.Series,
    pool_events: pd.DataFrame,
    asia_pools: pd.DataFrame,
    max_liq: int = 8,
    one_raid: bool = True,
    sw_len: int = 5,
) -> pd.DataFrame:
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr_v = atr.to_numpy()

    kz_start = (sessions["ny_start"] | sessions["london_start"]).to_numpy()
    in_kz = sessions["in_killzone"].to_numpy()
    new_day = day_tags["new_day"].to_numpy()
    pdh_v, pdl_v = pdh.to_numpy(), pdl.to_numpy()
    pool_up = pool_events["pool_up"].to_numpy()
    pool_dn = pool_events["pool_dn"].to_numpy()
    pool_up_price = pool_events["pool_up_price"].to_numpy()
    pool_dn_price = pool_events["pool_dn_price"].to_numpy()
    is_eqh = pool_events["is_eqh"].to_numpy()
    is_eql = pool_events["is_eql"].to_numpy()
    asia_hi = asia_pools["asia_pool_high"].to_numpy()
    asia_lo = asia_pools["asia_pool_low"].to_numpy()

    hi_book = _PoolBook(max_liq=max_liq)
    lo_book = _PoolBook(max_liq=max_liq)

    kz_hi = np.full(n, np.nan)
    kz_lo = np.full(n, np.nan)
    kz_hi_bar = np.full(n, -1, dtype=int)
    kz_lo_bar = np.full(n, -1, dtype=int)
    kz_hi_ty = np.full(n, -1, dtype=int)
    kz_lo_ty = np.full(n, -1, dtype=int)
    grab_up = np.zeros(n, dtype=bool)
    grab_dn = np.zeros(n, dtype=bool)
    grab_up_price = np.full(n, np.nan)
    grab_dn_price = np.full(n, np.nan)
    grab_up_type = np.full(n, -1, dtype=int)
    grab_dn_type = np.full(n, -1, dtype=int)
    grab_up_age = np.zeros(n, dtype=int)
    grab_dn_age = np.zeros(n, dtype=int)
    grab_seq = np.zeros(n, dtype=int)

    cur_kz_hi = cur_kz_lo = np.nan
    cur_kz_hi_bar = cur_kz_lo_bar = -1
    cur_kz_hi_ty = cur_kz_lo_ty = -1
    kz_used_up = kz_used_dn = False
    seq = 0

    for i in range(n):
        if pool_up[i]:
            hi_book.add(pool_up_price[i], i - sw_len, EQUAL if is_eqh[i] else SWING)
        if pool_dn[i]:
            lo_book.add(pool_dn_price[i], i - sw_len, EQUAL if is_eql[i] else SWING)
        if new_day[i] and not np.isnan(pdh_v[i]):
            hi_book.add(pdh_v[i], i, PREVDAY)
            lo_book.add(pdl_v[i], i, PREVDAY)
        if not np.isnan(asia_hi[i]):
            hi_book.add(asia_hi[i], i, ASIA)
        if not np.isnan(asia_lo[i]):
            lo_book.add(asia_lo[i], i, ASIA)

        if kz_start[i]:
            hp = hi_book.nearest_above(close[i])
            lp = lo_book.nearest_below(close[i])
            cur_kz_hi = hp.price if hp else np.nan
            cur_kz_hi_bar = hp.origin_bar if hp else -1
            cur_kz_hi_ty = hp.kind if hp else -1
            cur_kz_lo = lp.price if lp else np.nan
            cur_kz_lo_bar = lp.origin_bar if lp else -1
            cur_kz_lo_ty = lp.kind if lp else -1
            kz_used_up = False
            kz_used_dn = False

        kz_hi[i], kz_lo[i] = cur_kz_hi, cur_kz_lo
        kz_hi_bar[i], kz_lo_bar[i] = cur_kz_hi_bar, cur_kz_lo_bar
        kz_hi_ty[i], kz_lo_ty[i] = cur_kz_hi_ty, cur_kz_lo_ty

        hi_book.prune_swept_above(high[i])
        lo_book.prune_swept_below(low[i])

        gu = (not np.isnan(cur_kz_hi)) and in_kz[i] and high[i] > cur_kz_hi and not (one_raid and kz_used_up)
        gd = (not np.isnan(cur_kz_lo)) and in_kz[i] and low[i] < cur_kz_lo and not (one_raid and kz_used_dn)
        if gu:
            kz_used_up = True
        if gd:
            kz_used_dn = True

        grab_up[i], grab_dn[i] = gu, gd
        if gu:
            grab_up_price[i] = cur_kz_hi
            grab_up_type[i] = cur_kz_hi_ty
            grab_up_age[i] = i - cur_kz_hi_bar
        if gd:
            grab_dn_price[i] = cur_kz_lo
            grab_dn_type[i] = cur_kz_lo_ty
            grab_dn_age[i] = i - cur_kz_lo_bar

        if new_day[i]:
            seq = 0
        if gu or gd:
            seq += 1
        grab_seq[i] = seq

    range_width_atr = np.where(
        (~np.isnan(kz_hi)) & (~np.isnan(kz_lo)) & (atr_v > 0), (kz_hi - kz_lo) / atr_v, np.nan
    )

    return pd.DataFrame(
        {
            "kz_hi": kz_hi,
            "kz_lo": kz_lo,
            "kz_hi_bar": kz_hi_bar,
            "kz_lo_bar": kz_lo_bar,
            "kz_hi_type": kz_hi_ty,
            "kz_lo_type": kz_lo_ty,
            "grab_up": grab_up,
            "grab_dn": grab_dn,
            "grab_up_price": grab_up_price,
            "grab_dn_price": grab_dn_price,
            "grab_up_type": grab_up_type,
            "grab_dn_type": grab_dn_type,
            "grab_up_age": grab_up_age,
            "grab_dn_age": grab_dn_age,
            "range_width_atr": range_width_atr,
            "grab_seq": grab_seq,
        },
        index=df.index,
    )
