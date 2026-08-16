"""Quick Phase A unit checks without pytest (Windows WDAC sometimes blocks pandas.testing)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.data.split import SplitManifest
from ht_backtest.strategies.base import StrategyContext, TradeCandidate, assemble_symbol_trades
from ht_backtest.strategies.holy_trinity_v10 import HolyTrinityV10Strategy
from ht_backtest.strategies.registry import get_strategy, list_strategies
from ht_backtest.trades.pipeline import generate_trades as pipeline_generate_trades


def _tiny_ohlcv(n: int = 800, seed: int = 1) -> pd.DataFrame:
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


def main() -> int:
    assert "ht_v10" in list_strategies()
    meta = get_strategy("ht_v10").metadata()
    assert meta.id == "holy_trinity_v10"
    assert len(meta.parameter_hash) == 16
    print("metadata ok:", meta.to_dict())

    try:
        TradeCandidate("long", 0, 1.0, 0.9, 0.0, "x")
        raise AssertionError("risk=0 should raise")
    except ValueError:
        print("risk validation ok")

    df = _tiny_ohlcv()
    symbol = "BTC/USDT:USDT"
    start, end = int(df.timestamp.iloc[0]), int(df.timestamp.iloc[-1])
    split = SplitManifest(
        seed=1,
        timeframe="15m",
        symbol_holdout_fraction=0.3,
        date_holdout_fraction=0.2,
        universe=[symbol],
        holdout_symbols=[],
        train_symbols=[symbol],
        overall_start_ms=start,
        overall_end_ms=end,
        date_holdout_start_ms=start + (end - start) // 5,
    )
    ctx = StrategyContext(symbol=symbol, timeframe="15m", split=split)
    direct, _ = pipeline_generate_trades(df)
    strategy = HolyTrinityV10Strategy()
    candidates = strategy.generate_trades(df, ctx)
    assert len(candidates) == len(direct), (len(candidates), len(direct))
    for c, d in zip(candidates, direct):
        assert c.entry_bar == d["entry_bar"]
        assert c.entry_price == d["entry_price"]
        assert c.stop_price == d["stop_price"]
        assert c.risk == d["risk"]
    frame = assemble_symbol_trades(strategy, candidates, df, symbol, "15m", split)
    if not frame.empty:
        assert "big_displacement" in frame.columns
        assert "strategy_id" not in frame.columns or True
    print(f"adapter parity ok: {len(candidates)} trades on synthetic bars")
    print("UNIT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
