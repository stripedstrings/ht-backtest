"""Unit tests for the Strategy protocol and HT adapter (no full-universe run)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ht_backtest.data.split import SplitManifest
from ht_backtest.strategies.base import StrategyContext, TradeCandidate, assemble_symbol_trades
from ht_backtest.strategies.holy_trinity_v10 import HolyTrinityV10Strategy
from ht_backtest.strategies.registry import get_strategy, list_strategies
from ht_backtest.trades.pipeline import generate_trades as pipeline_generate_trades


def _tiny_ohlcv(n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = int(pd.Timestamp("2024-01-02 00:00:00", tz="UTC").timestamp() * 1000)
    ts = start + np.arange(n) * 15 * 60_000
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    open_ = close - rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.3, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.3, n)
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": np.ones(n)}
    )


def _split_for(symbol: str, df: pd.DataFrame) -> SplitManifest:
    start = int(df["timestamp"].iloc[0])
    end = int(df["timestamp"].iloc[-1])
    mid = start + (end - start) // 5
    return SplitManifest(
        seed=1,
        timeframe="15m",
        symbol_holdout_fraction=0.3,
        date_holdout_fraction=0.2,
        universe=[symbol],
        holdout_symbols=[],
        train_symbols=[symbol],
        overall_start_ms=start,
        overall_end_ms=end,
        date_holdout_start_ms=mid,
    )


def test_registry_lists_ht():
    names = list_strategies()
    assert "ht_v10" in names
    assert "holy_trinity_v10" in names
    s = get_strategy("ht_v10")
    meta = s.metadata()
    assert meta.id == "holy_trinity_v10"
    assert meta.version == "1.0.0"
    assert len(meta.parameter_hash) == 16
    assert "session range" in meta.description.lower() or "killzone" in meta.description.lower()


def test_trade_candidate_requires_positive_risk():
    try:
        TradeCandidate(
            direction="long",
            entry_bar=1,
            entry_price=100.0,
            stop_price=99.0,
            risk=0.0,
            strategy_id="x",
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ht_adapter_matches_pipeline_trade_count_and_core_fields():
    df = _tiny_ohlcv()
    symbol = "BTC/USDT:USDT"
    split = _split_for(symbol, df)
    ctx = StrategyContext(symbol=symbol, timeframe="15m", split=split)

    direct, _ = pipeline_generate_trades(df)
    strategy = HolyTrinityV10Strategy()
    candidates = strategy.generate_trades(df, ctx)

    assert len(candidates) == len(direct)
    if not candidates:
        return

    for c, d in zip(candidates, direct):
        assert c.direction == d["direction"]
        assert c.entry_bar == d["entry_bar"]
        assert c.entry_price == d["entry_price"]
        assert c.stop_price == d["stop_price"]
        assert c.risk == d["risk"]
        legacy = c.to_legacy_dict()
        for key in ("grab_bar", "mss_bar", "wick_atr", "body_atr", "target_price", "entry_time"):
            assert key in legacy
            if d[key] is None or (isinstance(d[key], float) and np.isnan(d[key])):
                continue
            assert legacy[key] == d[key] or (isinstance(legacy[key], float) and np.isclose(legacy[key], d[key]))


def test_ht_assemble_preserves_median_tag_columns():
    df = _tiny_ohlcv(n=2000, seed=3)
    symbol = "ETH/USDT:USDT"
    split = _split_for(symbol, df)
    ctx = StrategyContext(symbol=symbol, timeframe="15m", split=split)
    strategy = HolyTrinityV10Strategy()
    candidates = strategy.generate_trades(df, ctx)
    frame = assemble_symbol_trades(strategy, candidates, df, symbol, "15m", split)
    if frame.empty:
        return
    for col in ("big_wick", "big_displacement", "wide_range", "split", "session_range_high"):
        assert col in frame.columns
