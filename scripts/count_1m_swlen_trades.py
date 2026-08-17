"""Count kz_first_raid_reclaim trades on 1m for sw_len=10 and sw_len=20.

Session windows unchanged (Europe/London). Stop after printing counts —
no reach tables / grid.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.data.downloader import OHLCVDownloader  # noqa: E402
from ht_backtest.data.split import SplitManifest  # noqa: E402
from ht_backtest.gates.primitive_cache import compute_symbol_primitives  # noqa: E402
from ht_backtest.strategies.hypothesis_helpers import (  # noqa: E402
    collect_raid_events,
    session_range_frame,
)

SW_LENS = (10, 20)
# reclaim window in bars — scale with timeframe: 3 bars on 15m ≈ 45m;
# on 1m use 45 bars so reclaim horizon stays ~45 minutes of clock time.
RECLAIM_WIN_1M = 45


def count_first_reclaims(bars: pd.DataFrame, sw_len: int) -> int:
    prim = compute_symbol_primitives(bars, sw_len=sw_len)
    # pool_events inside prim use sw_len from compute; session_range must match
    sr = session_range_frame(bars, prim, sw_len=sw_len)
    # temporarily patch reclaim look via local copy of collect logic
    from ht_backtest.strategies import hypothesis_helpers as hh

    old = hh.RECLAIM_WIN
    hh.RECLAIM_WIN = RECLAIM_WIN_1M
    try:
        events = collect_raid_events(bars, prim, sr)
    finally:
        hh.RECLAIM_WIN = old
    n = 0
    for e in events:
        if e.first_of_kz and e.reclaim_bar is not None:
            n += 1
    return n


def main() -> int:
    split = SplitManifest.load(ROOT / "specs" / "splits" / "v1.json")
    assert "1m" in split.supported_timeframes()
    dl = OHLCVDownloader(cache_dir=ROOT / "data" / "raw", limit=1500)

    # Span matching split calendar (all symbols — frequency target is universe-wide)
    since = split.overall_start_ms
    until = split.overall_end_ms

    results = {sw: {"total": 0, "by_symbol": {}} for sw in SW_LENS}
    t0 = time.perf_counter()

    for i, sym in enumerate(split.universe, 1):
        bars = dl.cached_range(sym, "1m", since, until)
        if bars.empty:
            print(f"[{i}/30] {sym}: NO 1m DATA")
            continue
        print(f"[{i}/30] {sym}: 1m bars={len(bars):,}")
        for sw in SW_LENS:
            n_all = count_first_reclaims(bars, sw) if len(bars) > 500 else 0
            results[sw]["total"] += n_all
            results[sw]["by_symbol"][sym] = n_all
            print(f"       sw_len={sw}: trades={n_all}")

    # Calendar days in span
    days = (until - since) / (86_400_000)
    print()
    print("=" * 72)
    print("1m session-range base trade counts (first raid + reclaim)")
    print(f"reclaim_win={RECLAIM_WIN_1M} bars (~45m clock); sessions = London/NY unchanged")
    print(f"span days ≈ {days:.1f}")
    print("=" * 72)
    for sw in SW_LENS:
        total = results[sw]["total"]
        per_day = total / days if days else float("nan")
        print(
            f"sw_len={sw:2d}  (~{sw}-minute swing):  "
            f"trades={total:,}  ({per_day:.1f}/day across 30 symbols)"
        )
        if 50 <= per_day <= 200:
            print("  -> within target band 50-200 trades/day")
        elif per_day < 50:
            print("  -> BELOW target (too sparse)")
        else:
            print("  -> ABOVE target (too busy)")
    print(f"elapsed {time.perf_counter()-t0:.1f}s")
    print("Stop: no reach / grid runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
