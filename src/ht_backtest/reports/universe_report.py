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
from ht_backtest.data.funding import attach_funding_rate
from ht_backtest.data.split import SplitManifest
from ht_backtest.gates.primitive_cache import load_or_compute_primitives
from ht_backtest.profiling import StageTimings
from ht_backtest.reports.reach import reach_vs_random_walk
from ht_backtest.strategies.base import (
    Strategy,
    StrategyContext,
    align_aux_bars_to_primary,
    assemble_symbol_trades,
    strategy_requires_symbols,
)
from ht_backtest.trades.forward_tracker import track_forward_reach

_FAR_PAST_MS = 0
_FAR_FUTURE_MS = 4_102_444_800_000  # 2100-01-01


def _load_aux_bars(
    strategy: Strategy,
    primary_symbol: str,
    primary_df: pd.DataFrame,
    timeframe: str,
    exchange_id: str,
    cache_dir: str,
) -> dict[str, pd.DataFrame] | None:
    """Load and timestamp-align auxiliary frames if strategy.requires_symbols is set.

    HT and other single-symbol strategies return None (no aux load).
    """
    needed = strategy_requires_symbols(strategy)
    if not needed:
        return None
    dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)
    raw: dict[str, pd.DataFrame] = {}
    for sym in needed:
        if sym == primary_symbol:
            continue
        adf = dl.cached_range(sym, timeframe, _FAR_PAST_MS, _FAR_FUTURE_MS)
        if not adf.empty:
            raw[sym] = adf
    if not raw:
        return {}
    return align_aux_bars_to_primary(primary_df, raw)


def _process_symbol_timed(
    strategy: Strategy,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    split: SplitManifest,
    exchange_id: str,
    mfe_win: int,
    *,
    use_primitive_cache: bool,
    primitives_cache_dir: str,
    cache_dir: str = "data/raw",
    funding_dir: str = "data/funding",
    attach_funding: bool = True,
) -> tuple[pd.DataFrame, int, StageTimings]:
    timings = StageTimings(symbols=1)
    t_all = time.perf_counter()

    if attach_funding:
        df = attach_funding_rate(df, symbol, funding_dir=funding_dir)

    t0 = time.perf_counter()
    prim = load_or_compute_primitives(
        df,
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        cache_dir=primitives_cache_dir,
        use_cache=use_primitive_cache,
    )
    timings.primitives_s = time.perf_counter() - t0
    timings.notes["primitives_cache_hit"] = prim.cache_hit

    aux_bars = _load_aux_bars(strategy, symbol, df, timeframe, exchange_id, cache_dir)
    if aux_bars is not None:
        timings.notes["aux_symbols"] = list(aux_bars.keys())

    ctx = StrategyContext(
        symbol=symbol,
        timeframe=timeframe,
        split=split,
        exchange_id=exchange_id,
        primitives=prim,
        aux_bars=aux_bars,
    )

    t0 = time.perf_counter()
    # HT pipeline records gate_fsm inside generate_trades when timings passed;
    # baselines fold signal generation into gate_fsm_s for the stage budget.
    from ht_backtest.trades.pipeline import generate_trades as _ht_pipe
    from ht_backtest.strategies.holy_trinity_v10 import HolyTrinityV10Strategy

    if isinstance(strategy, HolyTrinityV10Strategy):
        stage = StageTimings()
        trades_list, _ = _ht_pipe(
            df,
            params=strategy.params,
            primitives=prim,
            timings=stage,
            **strategy._pipeline_kwargs(),
        )
        from ht_backtest.strategies.base import TradeCandidate

        candidates = [
            TradeCandidate.from_legacy_trade(t, strategy_id=strategy.metadata().id, symbol=symbol)
            for t in trades_list
        ]
        timings.gate_fsm_s = stage.gate_fsm_s
        # primitives already timed above (cache path); don't double-count pipeline prim
    else:
        candidates = strategy.generate_trades(df, ctx)
        timings.gate_fsm_s = time.perf_counter() - t0

    meta = strategy.metadata()
    t0 = time.perf_counter()
    tagged = assemble_symbol_trades(strategy, candidates, df, symbol, timeframe, split)
    if not tagged.empty:
        tagged = tagged.copy()
        tagged["strategy_id"] = meta.id
        tagged["strategy_version"] = meta.version
        tagged["strategy_parameter_hash"] = meta.parameter_hash
    timings.assemble_tag_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tracked = track_forward_reach(tagged, df, mfe_win=mfe_win)
    timings.forward_tracker_s = time.perf_counter() - t0
    timings.trades = len(candidates)
    timings.total_s = time.perf_counter() - t_all
    return tracked, len(candidates), timings


def _run_one_symbol(payload: dict[str, Any]) -> tuple[str, pd.DataFrame, float, int, dict]:
    """Picklable worker: strategy by registry name + split loaded from path."""
    from ht_backtest.strategies.registry import get_strategy

    t_load0 = time.perf_counter()
    strategy = get_strategy(payload["strategy_name"])
    split = SplitManifest.load(payload["split_path"])
    symbol = payload["symbol"]
    timeframe = payload["timeframe"]
    dl = OHLCVDownloader(exchange_id=payload["exchange_id"], cache_dir=payload["cache_dir"])
    df = dl.cached_range(symbol, timeframe, _FAR_PAST_MS, _FAR_FUTURE_MS)
    load_s = time.perf_counter() - t_load0
    if df.empty:
        empty_t = StageTimings(parquet_load_s=load_s, total_s=load_s)
        return symbol, pd.DataFrame(), load_s, 0, empty_t.to_dict()

    tracked, n_cand, timings = _process_symbol_timed(
        strategy,
        df,
        symbol,
        timeframe,
        split,
        payload["exchange_id"],
        payload["mfe_win"],
        use_primitive_cache=payload.get("use_primitive_cache", True),
        primitives_cache_dir=payload.get("primitives_cache_dir", "data/cache/primitives"),
        cache_dir=payload["cache_dir"],
        funding_dir=payload.get("funding_dir", "data/funding"),
        attach_funding=payload.get("attach_funding", True),
    )
    timings.parquet_load_s = load_s
    timings.total_s += load_s
    return symbol, tracked, timings.total_s, n_cand, timings.to_dict()


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
    use_primitive_cache: bool = True,
    primitives_cache_dir: str = "data/cache/primitives",
    funding_dir: str = "data/funding",
    attach_funding: bool = True,
    log_fn=print,
) -> tuple[pd.DataFrame, StageTimings]:
    """Generate pooled trades for one strategy across the split universe.

    Returns (trades_df, StageTimings). `workers>1` uses a process pool over
    symbols and requires `strategy_name` + `split_path`.

    When ``attach_funding`` is True, each symbol's OHLCV frame gains a causal
    ``funding_rate`` column (row count unchanged).
    """
    meta = strategy.metadata()
    workers = max(1, int(workers))
    agg = StageTimings()
    t_run = time.perf_counter()

    if workers == 1:
        dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)
        frames: list[pd.DataFrame] = []
        for i, symbol in enumerate(split.universe, 1):
            t0 = time.perf_counter()
            df = dl.cached_range(symbol, timeframe, _FAR_PAST_MS, _FAR_FUTURE_MS)
            load_s = time.perf_counter() - t0
            if df.empty:
                log_fn(f"[{i}/{len(split.universe)}] {symbol}: no cached data, skipping")
                continue
            tracked, n_cand, timings = _process_symbol_timed(
                strategy,
                df,
                symbol,
                timeframe,
                split,
                exchange_id,
                mfe_win,
                use_primitive_cache=use_primitive_cache,
                primitives_cache_dir=primitives_cache_dir,
                cache_dir=cache_dir,
                funding_dir=funding_dir,
                attach_funding=attach_funding,
            )
            timings.parquet_load_s = load_s
            timings.total_s += load_s
            agg = agg.add(timings)
            frames.append(tracked)
            n_train = int((tracked["split"] == "train").sum()) if not tracked.empty else 0
            n_holdout = len(tracked) - n_train
            log_fn(
                f"[{i}/{len(split.universe)}] {symbol}: {n_cand} trades "
                f"(train={n_train} holdout={n_holdout}) in {timings.total_s:.1f}s "
                f"[load={load_s:.2f} prim={timings.primitives_s:.2f} "
                f"gate={timings.gate_fsm_s:.2f} track={timings.forward_tracker_s:.2f}"
                f"{' cache_hit' if timings.notes.get('primitives_cache_hit') else ''}]"
            )
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        t_agg0 = time.perf_counter()
        if not out.empty:
            _ = reach_vs_random_walk(out[out["split"] == "train"] if "split" in out.columns else out)
        agg.reach_aggregation_s = time.perf_counter() - t_agg0
        agg.total_s = time.perf_counter() - t_run
        agg.notes["strategy_id"] = meta.id
        agg.notes["use_primitive_cache"] = use_primitive_cache
        return out, agg

    if not strategy_name:
        raise ValueError("workers>1 requires strategy_name (registry key) for pickling")
    if not split_path:
        split_path = Path(cache_dir).parent / "runs" / "_tmp_splits" / f"{meta.id}_{os.getpid()}.json"
        Path(split_path).parent.mkdir(parents=True, exist_ok=True)
        split.save(split_path)
    split_path = str(Path(split_path).resolve())
    cache_dir = str(Path(cache_dir).resolve())
    primitives_cache_dir = str(Path(primitives_cache_dir).resolve())

    payloads = [
        {
            "strategy_name": strategy_name,
            "split_path": split_path,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange_id": exchange_id,
            "cache_dir": cache_dir,
            "mfe_win": mfe_win,
            "use_primitive_cache": use_primitive_cache,
            "primitives_cache_dir": primitives_cache_dir,
            "funding_dir": str(Path(funding_dir).resolve()),
            "attach_funding": attach_funding,
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
                sym, tracked, elapsed, n_cand, timing_dict = fut.result()
            except Exception as exc:  # noqa: BLE001
                log_fn(f"[{done}/{len(payloads)}] {symbol}: ERROR {exc}")
                continue
            st = StageTimings(**{k: timing_dict[k] for k in timing_dict if k in StageTimings.__dataclass_fields__})
            # notes may be nested
            if "notes" in timing_dict:
                st.notes = timing_dict["notes"]
            agg = agg.add(st)
            if tracked.empty and n_cand == 0:
                log_fn(f"[{done}/{len(payloads)}] {sym}: no cached data or no trades in {elapsed:.1f}s")
            else:
                n_train = int((tracked["split"] == "train").sum()) if not tracked.empty else 0
                n_holdout = len(tracked) - n_train
                log_fn(
                    f"[{done}/{len(payloads)}] {sym}: {n_cand} trades "
                    f"(train={n_train} holdout={n_holdout}) in {elapsed:.1f}s "
                    f"[load={st.parquet_load_s:.2f} prim={st.primitives_s:.2f} "
                    f"gate={st.gate_fsm_s:.2f} track={st.forward_tracker_s:.2f}"
                    f"{' cache_hit' if st.notes.get('primitives_cache_hit') else ''}]"
                )
            if not tracked.empty:
                frames.append(tracked)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    t_agg0 = time.perf_counter()
    if not out.empty:
        _ = reach_vs_random_walk(out[out["split"] == "train"] if "split" in out.columns else out)
    agg.reach_aggregation_s += time.perf_counter() - t_agg0
    agg.total_s = time.perf_counter() - t_run
    agg.notes["strategy_id"] = meta.id
    agg.notes["use_primitive_cache"] = use_primitive_cache
    agg.notes["workers"] = workers
    return out, agg
