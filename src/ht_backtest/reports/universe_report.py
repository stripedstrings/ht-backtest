"""Pools trades across the whole symbol universe for the evidence tables.
Callers are responsible for filtering to a `split` column value ("train" or
"holdout") before analysis -- this module just generates and tags, it does
not decide which half anyone is allowed to look at."""

from __future__ import annotations

import time

import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import SplitManifest
from ht_backtest.trades.forward_tracker import track_forward_reach
from ht_backtest.trades.pipeline import generate_trades
from ht_backtest.trades.state_machine import GateParams
from ht_backtest.trades.tagging import assemble_trade_frame

_FAR_PAST_MS = 0
_FAR_FUTURE_MS = 4_102_444_800_000  # 2100-01-01


def generate_pooled_trades(
    split: SplitManifest,
    timeframe: str,
    exchange_id: str = "binanceusdm",
    cache_dir: str = "data/raw",
    params: GateParams = GateParams(),
    mfe_win: int = 100,
    log_fn=print,
) -> pd.DataFrame:
    dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)
    frames = []
    for i, symbol in enumerate(split.universe, 1):
        df = dl.cached_range(symbol, timeframe, _FAR_PAST_MS, _FAR_FUTURE_MS)
        if df.empty:
            log_fn(f"[{i}/{len(split.universe)}] {symbol}: no cached data, skipping")
            continue
        t0 = time.time()
        trades, _ = generate_trades(df, params=params)
        tagged = assemble_trade_frame(trades, symbol, timeframe, split)
        tracked = track_forward_reach(tagged, df, mfe_win=mfe_win)
        frames.append(tracked)
        n_train = int((tracked["split"] == "train").sum()) if not tracked.empty else 0
        n_holdout = len(tracked) - n_train
        log_fn(
            f"[{i}/{len(split.universe)}] {symbol}: {len(trades)} trades "
            f"(train={n_train} holdout={n_holdout}) in {time.time()-t0:.1f}s"
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
