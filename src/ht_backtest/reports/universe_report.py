"""Pools trades across the whole symbol universe for the evidence tables.

Callers are responsible for filtering to a `split` column value ("train" or
"holdout") before analysis — this module just generates and tags, it does
not decide which half anyone is allowed to look at.

Strategies plug in via the Strategy protocol; the shared forward reach
tracker and reach-vs-RW tables stay strategy-agnostic.
"""

from __future__ import annotations

import time

import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import SplitManifest
from ht_backtest.strategies.base import Strategy, StrategyContext, assemble_symbol_trades
from ht_backtest.trades.forward_tracker import track_forward_reach

_FAR_PAST_MS = 0
_FAR_FUTURE_MS = 4_102_444_800_000  # 2100-01-01


def generate_pooled_trades(
    strategy: Strategy,
    split: SplitManifest,
    timeframe: str,
    exchange_id: str = "binanceusdm",
    cache_dir: str = "data/raw",
    mfe_win: int = 100,
    log_fn=print,
) -> pd.DataFrame:
    meta = strategy.metadata()
    dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)
    frames: list[pd.DataFrame] = []
    for i, symbol in enumerate(split.universe, 1):
        df = dl.cached_range(symbol, timeframe, _FAR_PAST_MS, _FAR_FUTURE_MS)
        if df.empty:
            log_fn(f"[{i}/{len(split.universe)}] {symbol}: no cached data, skipping")
            continue
        t0 = time.time()
        ctx = StrategyContext(
            symbol=symbol,
            timeframe=timeframe,
            split=split,
            exchange_id=exchange_id,
        )
        candidates = strategy.generate_trades(df, ctx)
        tagged = assemble_symbol_trades(strategy, candidates, df, symbol, timeframe, split)
        if not tagged.empty:
            tagged = tagged.copy()
            tagged["strategy_id"] = meta.id
            tagged["strategy_version"] = meta.version
            tagged["strategy_parameter_hash"] = meta.parameter_hash
        tracked = track_forward_reach(tagged, df, mfe_win=mfe_win)
        frames.append(tracked)
        n_train = int((tracked["split"] == "train").sum()) if not tracked.empty else 0
        n_holdout = len(tracked) - n_train
        log_fn(
            f"[{i}/{len(split.universe)}] {symbol}: {len(candidates)} trades "
            f"(train={n_train} holdout={n_holdout}) in {time.time()-t0:.1f}s"
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
