"""Step 1 dry-count only: expected train n for hypotheses 1, 3, 13, 16.

Definitions (locked, loosest defensible):
  1  — first raid of a killzone instance + reclaim (close back inside) within 3 bars
  3  — same as 1, but raid bar in first 30 minutes of London (07:00-07:30) or NY (12:30-13:00) London time
  13 — London raid+reclaim, then same UTC day NY raid+reclaim in same direction
  16 — raid with NO reclaim within 20 bars, then next killzone: first close through
       the failed edge opposite the raid direction

Train only: train symbols, timestamps < date_holdout_start_ms.
Does not write strategy code or modify the engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import SplitManifest
from ht_backtest.gates.primitive_cache import compute_symbol_primitives
from ht_backtest.gates.session_range import run_session_range_engine

SWEEP_WIN = 3
FAIL_WIN = 20
_FAR_PAST = 0
_FAR_FUTURE = 4_102_444_800_000


def _reclaim_bar(close: np.ndarray, grab_i: int, direction: str, level: float, max_bars: int) -> int | None:
    """Return bar index of reclaim, or None. direction 'up' = raided high (need close < level)."""
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
    close: np.ndarray,
    in_sess: np.ndarray,
    start_i: int,
    direction: str,
    level: float,
) -> int | None:
    """First bar of the next session stretch that closes through level opposite the raid."""
    n = len(close)
    # find start of a session stretch at or after start_i
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


def count_symbol(df: pd.DataFrame, symbol: str, split: SplitManifest) -> dict[str, int]:
    prim = compute_symbol_primitives(df)
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
    grab_up = sr["grab_up"].to_numpy()
    grab_dn = sr["grab_dn"].to_numpy()
    grab_up_px = sr["grab_up_price"].to_numpy()
    grab_dn_px = sr["grab_dn_price"].to_numpy()
    kz_start = (prim.sessions["london_start"] | prim.sessions["ny_start"]).to_numpy()
    in_london = prim.sessions["in_london"].to_numpy()
    in_ny = prim.sessions["in_ny"].to_numpy()
    hour = prim.sessions["hour_london"].to_numpy()
    minute = pd.to_datetime(ts, unit="ms", utc=True).tz_convert("Europe/London").minute.to_numpy()

    n = len(df)
    # Mark first raid bar index within each killzone instance (from kz_start to next kz_start)
    first_raid_of_kz = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not kz_start[i]:
            i += 1
            continue
        # session stretch until next kz_start
        j = i + 1
        while j < n and not kz_start[j]:
            j += 1
        # first grab in [i, j)
        for k in range(i, j):
            if grab_up[k] or grab_dn[k]:
                first_raid_of_kz[k] = True
                break
        i = j

    def is_train(bar: int) -> bool:
        return split.classify(symbol, int(ts[bar])) == "train"

    def in_first_30(bar: int) -> bool:
        # London 07:00-07:30 or NY 12:30-13:00 Europe/London (15m bars: 07:00,07:15 / 12:30,12:45)
        h, m = int(hour[bar]), int(minute[bar])
        if in_london[bar] and h == 7 and m < 30:
            return True
        if in_ny[bar] and h == 12 and m >= 30:
            return True
        return False

    # Collect completed raid+reclaim events: (reclaim_bar, direction, session, grab_bar, level, utc_day)
    events: list[dict] = []
    failed: list[dict] = []

    for i in range(n):
        if grab_up[i]:
            direction, level = "up", float(grab_up_px[i])
        elif grab_dn[i]:
            direction, level = "dn", float(grab_dn_px[i])
        else:
            continue
        sess = "LONDON" if in_london[i] else ("NY" if in_ny[i] else "OTHER")
        utc_day = int(ts[i]) // 86_400_000
        rec = _reclaim_bar(close, i, direction, level, SWEEP_WIN)
        if rec is not None:
            events.append(
                {
                    "grab_bar": i,
                    "reclaim_bar": rec,
                    "direction": direction,
                    "session": sess,
                    "level": level,
                    "utc_day": utc_day,
                    "first_of_kz": bool(first_raid_of_kz[i]),
                    "first_30": in_first_30(i),
                }
            )
        else:
            # no reclaim in 3 bars — check fail window of 20 for strategy 16
            rec20 = _reclaim_bar(close, i, direction, level, FAIL_WIN)
            if rec20 is None:
                failed.append(
                    {
                        "grab_bar": i,
                        "direction": direction,
                        "session": sess,
                        "level": level,
                        "utc_day": utc_day,
                    }
                )

    n1 = n3 = 0
    for e in events:
        if not e["first_of_kz"]:
            continue
        if not is_train(e["reclaim_bar"]):
            continue
        n1 += 1
        if e["first_30"]:
            n3 += 1

    # 13: London reclaim then NY same direction same UTC day
    n13 = 0
    lon = [e for e in events if e["session"] == "LONDON"]
    ny = [e for e in events if e["session"] == "NY"]
    for L in lon:
        if not is_train(L["reclaim_bar"]):
            continue
        for N in ny:
            if N["utc_day"] != L["utc_day"]:
                continue
            if N["direction"] != L["direction"]:
                continue
            if N["grab_bar"] <= L["reclaim_bar"]:
                continue
            if not is_train(N["reclaim_bar"]):
                continue
            n13 += 1
            break  # one pair per London event (loosest: count London legs that get a NY confirm)

    # 16: failed raid -> next killzone fade
    n16 = 0
    for f in failed:
        # next killzone start after grab
        nxt = None
        for j in range(f["grab_bar"] + 1, n):
            if kz_start[j]:
                nxt = j
                break
        if nxt is None:
            continue
        # which session starts?
        if prim.sessions["london_start"].iloc[nxt]:
            in_sess = in_london
        elif prim.sessions["ny_start"].iloc[nxt]:
            in_sess = in_ny
        else:
            continue
        fade = _first_fade_in_session(close, in_sess, nxt, f["direction"], f["level"])
        if fade is not None and is_train(fade):
            n16 += 1

    return {"1": n1, "3": n3, "13": n13, "16": n16}


def main() -> int:
    split = SplitManifest.load(ROOT / "specs" / "splits" / "v1.json")
    dl = OHLCVDownloader(exchange_id="binanceusdm", cache_dir=str(ROOT / "data" / "raw"))
    totals = {"1": 0, "3": 0, "13": 0, "16": 0}
    print("=" * 72)
    print("STEP 1 DRY-COUNT — train n only (split v1)")
    print(f"train symbols: {len(split.train_symbols)}")
    print(f"date cutoff:   {split.to_dict()['date_holdout_start']}")
    print("=" * 72)
    print(f"{'symbol':<22} {'n1':>6} {'n3':>6} {'n13':>6} {'n16':>6}")
    for i, symbol in enumerate(split.train_symbols, 1):
        df = dl.cached_range(symbol, "15m", _FAR_PAST, _FAR_FUTURE)
        if df.empty:
            print(f"[{i}/{len(split.train_symbols)}] {symbol}: no data")
            continue
        c = count_symbol(df, symbol, split)
        for k in totals:
            totals[k] += c[k]
        print(f"{symbol:<22} {c['1']:6d} {c['3']:6d} {c['13']:6d} {c['16']:6d}")

    print("=" * 72)
    print("TOTALS (train)")
    print(f"  strategy 1  first raid + reclaim (3 bars):           n = {totals['1']}")
    print(f"  strategy 3  first raid in first 30m + reclaim:      n = {totals['3']}")
    print(f"  strategy 13 London then NY same-dir same UTC day:   n = {totals['13']}")
    print(f"  strategy 16 failed raid -> next session fade:       n = {totals['16']}")
    print("=" * 72)
    print("n >= 200 gate:")
    for k, label in [
        ("1", "strategy 1"),
        ("3", "strategy 3"),
        ("13", "strategy 13"),
        ("16", "strategy 16"),
    ]:
        status = "SURVIVE" if totals[k] >= 200 else "KILL (<200)"
        print(f"  {label}: {totals[k]}  ->  {status}")
    print("=" * 72)
    print("STOP — awaiting confirmation before Step 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
