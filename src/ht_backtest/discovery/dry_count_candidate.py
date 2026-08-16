"""Dry-count allowlisted candidate methods on train universe (crypto 15m)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import SplitManifest
from ht_backtest.gates.primitive_cache import load_or_compute_primitives
from ht_backtest.gates.session_range import run_session_range_engine
_FAR_PAST = 0
_FAR_FUTURE = 4_102_444_800_000
RECLAIM_WIN = 3
FAIL_WIN = 20
VOL_LOOKBACK = 50


def _reclaim_bar(close: np.ndarray, grab_i: int, direction: str, level: float, max_bars: int) -> int | None:
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


def _first_fade_in_session(
    close: np.ndarray, in_sess: np.ndarray, start_i: int, direction: str, level: float
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


def _count_symbol(df: pd.DataFrame, symbol: str, split: SplitManifest, method: str, params: dict[str, Any]) -> int:
    if df.empty:
        return 0
    reclaim_win = int(params.get("reclaim_bars", RECLAIM_WIN))
    prim = load_or_compute_primitives(
        df,
        exchange_id="binanceusdm",
        symbol=symbol,
        timeframe="15m",
        cache_dir="data/cache/primitives",
        use_cache=True,
    )
    # load_or_compute may need different signature - check
    sr = run_session_range_engine(
        df,
        prim.atr,
        prim.sessions,
        prim.day_tags,
        prim.pdh,
        prim.pdl,
        prim.pool_events,
        prim.asia_pools,
        max_liq=8,
        one_raid=True,
        sw_len=5,
    )
    close = df["close"].to_numpy()
    ts = df["timestamp"].to_numpy()
    vol = df["volume"].to_numpy() if "volume" in df.columns else np.ones(len(df))
    grab_up = sr["grab_up"].to_numpy()
    grab_dn = sr["grab_dn"].to_numpy()
    grab_up_px = sr["grab_up_price"].to_numpy()
    grab_dn_px = sr["grab_dn_price"].to_numpy()
    kz_start = (prim.sessions["london_start"] | prim.sessions["ny_start"]).to_numpy()
    in_london = prim.sessions["in_london"].to_numpy()
    in_ny = prim.sessions["in_ny"].to_numpy()
    hour = prim.sessions["hour_london"].to_numpy()
    minute = pd.to_datetime(ts, unit="ms", utc=True).tz_convert("Europe/London").minute.to_numpy()
    vol_med = pd.Series(vol).rolling(VOL_LOOKBACK, min_periods=10).median().to_numpy()

    n = len(df)
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

    def is_train(bar: int) -> bool:
        return split.classify(symbol, int(ts[bar])) == "train"

    def in_first_30(bar: int) -> bool:
        h, m = int(hour[bar]), int(minute[bar])
        if in_london[bar] and h == 7 and m < 30:
            return True
        if in_ny[bar] and h == 12 and m >= 30:
            return True
        return False

    events = []
    failed = []
    for i in range(n):
        if grab_up[i]:
            direction, level = "up", float(grab_up_px[i])
        elif grab_dn[i]:
            direction, level = "dn", float(grab_dn_px[i])
        else:
            continue
        sess = "LONDON" if in_london[i] else ("NY" if in_ny[i] else "OTHER")
        utc_day = int(ts[i]) // 86_400_000
        rec = _reclaim_bar(close, i, direction, level, reclaim_win)
        vmed = float(vol_med[i]) if not np.isnan(vol_med[i]) else float("nan")
        if rec is not None:
            events.append(
                {
                    "grab_bar": i,
                    "reclaim_bar": rec,
                    "direction": direction,
                    "session": sess,
                    "utc_day": utc_day,
                    "first_of_kz": bool(first_raid_of_kz[i]),
                    "first_30": in_first_30(i),
                    "vol": float(vol[i]),
                    "vol_med": vmed,
                }
            )
        elif _reclaim_bar(close, i, direction, level, FAIL_WIN) is None:
            failed.append({"grab_bar": i, "direction": direction, "level": level, "session": sess})

    if method == "kz_first_raid_reclaim":
        return sum(1 for e in events if e["first_of_kz"] and is_train(e["reclaim_bar"]))
    if method == "kz_first_30m_raid_reclaim":
        return sum(1 for e in events if e["first_of_kz"] and e["first_30"] and is_train(e["reclaim_bar"]))
    if method == "raid_reclaim_all":
        return sum(1 for e in events if is_train(e["reclaim_bar"]))
    if method == "raid_reclaim_vol_high":
        return sum(
            1
            for e in events
            if is_train(e["reclaim_bar"]) and not np.isnan(e["vol_med"]) and e["vol"] >= e["vol_med"]
        )
    if method == "raid_reclaim_vol_low":
        return sum(
            1
            for e in events
            if is_train(e["reclaim_bar"]) and not np.isnan(e["vol_med"]) and e["vol"] <= e["vol_med"]
        )
    if method == "london_ny_same_direction":
        n13 = 0
        lon = [e for e in events if e["session"] == "LONDON"]
        ny = [e for e in events if e["session"] == "NY"]
        for L in lon:
            if not is_train(L["reclaim_bar"]):
                continue
            for N in ny:
                if N["utc_day"] != L["utc_day"] or N["direction"] != L["direction"]:
                    continue
                if N["grab_bar"] <= L["reclaim_bar"]:
                    continue
                if is_train(N["reclaim_bar"]):
                    n13 += 1
                    break
        return n13
    if method == "failed_raid_next_session_fade":
        n16 = 0
        for f in failed:
            nxt = None
            for j in range(f["grab_bar"] + 1, n):
                if kz_start[j]:
                    nxt = j
                    break
            if nxt is None:
                continue
            if prim.sessions["london_start"].iloc[nxt]:
                in_sess = in_london
            elif prim.sessions["ny_start"].iloc[nxt]:
                in_sess = in_ny
            else:
                continue
            fade = _first_fade_in_session(close, in_sess, nxt, f["direction"], f["level"])
            if fade is not None and is_train(fade):
                n16 += 1
        return n16
    raise ValueError(f"unknown dry_count method: {method}")


def dry_count_candidate(
    candidate: dict[str, Any],
    *,
    split_path: str | Path = "specs/splits/v1.json",
    cache_dir: str = "data/raw",
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Return {n, method, min_n, ok, reason}."""
    dc = candidate.get("dry_count") or {}
    method = dc.get("method")
    min_n = int(dc.get("min_n") or 200)
    params = dict(dc.get("params") or {})
    if not method:
        return {"n": 0, "method": None, "min_n": min_n, "ok": False, "reason": "untranslatable_dry_count"}

    split = SplitManifest.load(split_path)
    dl = OHLCVDownloader(exchange_id="binanceusdm", cache_dir=cache_dir)
    symbols = list(split.train_symbols)
    if max_symbols is not None:
        symbols = symbols[: max_symbols]

    total = 0
    scanned = 0
    for symbol in symbols:
        df = dl.cached_range(symbol, "15m", _FAR_PAST, _FAR_FUTURE)
        if df.empty:
            continue
        scanned += 1
        total += _count_symbol(df, symbol, split, method, params)

    if scanned == 0:
        return {
            "n": 0,
            "method": method,
            "min_n": min_n,
            "ok": False,
            "reason": "insufficient_data:no_cached_ohlcv",
            "symbols_scanned": 0,
        }
    ok = total >= min_n
    return {
        "n": int(total),
        "method": method,
        "min_n": min_n,
        "ok": ok,
        "reason": "" if ok else "n_too_low",
        "symbols_scanned": scanned,
    }
