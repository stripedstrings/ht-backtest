"""Pools trades across the whole symbol universe for the evidence tables.

Callers are responsible for filtering to a `split` column value ("train" or
"holdout") before analysis — this module just generates and tags, it does
not decide which half anyone is allowed to look at.

Strategies plug in via the Strategy protocol; the shared forward reach
tracker and reach-vs-RW tables stay strategy-agnostic.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import SplitManifest
from ht_backtest.strategies.base import Strategy, StrategyContext, assemble_symbol_trades
from ht_backtest.trades.forward_tracker import track_forward_reach

_FAR_PAST_MS = 0
_FAR_FUTURE_MS = 4_102_444_800_000  # 2100-01-01


def _run_one_symbol(payload: dict[str, Any]) -> tuple[str, pd.DataFrame, float, int]:
    """Picklable worker: strategy by registry name + split loaded from path."""
    from ht_backtest.strategies.registry import get_strategy

    t0 = time.time()
    strategy = get_strategy(payload["strategy_name"])
    meta = strategy.metadata()
    split = SplitManifest.load(payload["split_path"])
    symbol = payload["symbol"]
    timeframe = payload["timeframe"]
    dl = OHLCVDownloader(exchange_id=payload["exchange_id"], cache_dir=payload["cache_dir"])
    df = dl.cached_range(symbol, timeframe, _FAR_PAST_MS, _FAR_FUTURE_MS)
    if df.empty:
        return symbol, pd.DataFrame(), time.time() - t0, 0
    ctx = StrategyContext(
        symbol=symbol,
        timeframe=timeframe,
        split=split,
        exchange_id=payload["exchange_id"],
    )
    candidates = strategy.generate_trades(df, ctx)
    tagged = assemble_symbol_trades(strategy, candidates, df, symbol, timeframe, split)
    if not tagged.empty:
        tagged = tagged.copy()
        tagged["strategy_id"] = meta.id
        tagged["strategy_version"] = meta.version
        tagged["strategy_parameter_hash"] = meta.parameter_hash
    tracked = track_forward_reach(tagged, df, mfe_win=payload["mfe_win"])
    return symbol, tracked, time.time() - t0, len(candidates)


def generate_pooled_trades(
    strategy: Strategy,
    split: SplitManifest,
    timeframe: str,
    exchange_id: str = "binanceusdm",
    cache_dir: str = "data/raw",
    mfe_win: int = 100,
    workers: int = 1,
    strategy_name: str | None = None,
    split_path: str | Path | None = None,
    log_fn=print,
) -> pd.DataFrame:
    """Generate pooled trades for one strategy across the split universe.

    `workers>1` uses a process pool over symbols. Parallel mode requires
    `strategy_name` (registry key) and `split_path` so workers can rebuild
    state after spawn on Windows.
    """
    meta = strategy.metadata()
    workers = max(1, int(workers))

    if workers == 1:
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

    if not strategy_name:
        raise ValueError("workers>1 requires strategy_name (registry key) for pickling")
    if not split_path:
        # Persist a temp split path under cache so workers can load it
        split_path = Path(cache_dir).parent / "runs" / "_tmp_splits" / f"{meta.id}_{os.getpid()}.json"
        Path(split_path).parent.mkdir(parents=True, exist_ok=True)
        split.save(split_path)
    split_path = str(Path(split_path).resolve())
    cache_dir = str(Path(cache_dir).resolve())

    payloads = [
        {
            "strategy_name": strategy_name,
            "split_path": split_path,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange_id": exchange_id,
            "cache_dir": cache_dir,
            "mfe_win": mfe_win,
        }
        for symbol in split.universe
    ]

    frames = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one_symbol, p): p["symbol"] for p in payloads}
        for fut in as_completed(futures):
            symbol = futures[fut]
            done += 1
            try:
                sym, tracked, elapsed, n_cand = fut.result()
            except Exception as exc:  # noqa: BLE001 — surface per-symbol failure, continue batch
                log_fn(f"[{done}/{len(payloads)}] {symbol}: ERROR {exc}")
                continue
            if tracked.empty and n_cand == 0:
                log_fn(f"[{done}/{len(payloads)}] {sym}: no cached data or no trades in {elapsed:.1f}s")
            else:
                n_train = int((tracked["split"] == "train").sum()) if not tracked.empty else 0
                n_holdout = len(tracked) - n_train
                log_fn(
                    f"[{done}/{len(payloads)}] {sym}: {n_cand} trades "
                    f"(train={n_train} holdout={n_holdout}) in {elapsed:.1f}s"
                )
            if not tracked.empty:
                frames.append(tracked)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
