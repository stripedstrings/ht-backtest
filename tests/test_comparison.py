"""Unit tests for multi-strategy comparison (no full-universe I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ht_backtest.reports.comparison import strategy_comparison_table
from ht_backtest.strategies.registry import get_strategy, list_strategies


def test_registry_has_ten_plus_strategies():
    names = list_strategies()
    # ht aliases + 9 baselines; unique metadata ids should be >= 10
    assert "ht_v10" in names
    assert "london_open_long_atr1" in names
    assert "donchian_20_long" in names
    ids = {get_strategy(n).metadata().id for n in names}
    assert len(ids) >= 10


def test_baseline_emits_candidates_on_synthetic():
    n = 400
    start = int(pd.Timestamp("2024-06-03 00:00:00", tz="UTC").timestamp() * 1000)
    ts = start + np.arange(n) * 15 * 60_000
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.2, n))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n),
        }
    )
    from ht_backtest.strategies.base import StrategyContext

    ctx = StrategyContext(symbol="BTC/USDT:USDT", timeframe="15m")
    s = get_strategy("donchian_20_long")
    trades = s.generate_trades(df, ctx)
    assert isinstance(trades, list)
    meta = s.metadata()
    assert meta.id
    assert meta.parameter_hash
    assert meta.description


def test_comparison_table_marks_promotion_correctly():
    def _frame(reach_rate: float, n: int = 250) -> pd.DataFrame:
        hits = int(round(reach_rate * n))
        vals = [True] * hits + [False] * (n - hits)
        return pd.DataFrame(
            {
                "split": ["train"] * n,
                "reach_0.5R": vals,
                "reach_1.0R": vals,
                "reach_1.5R": vals,
                "reach_2.0R": vals,
                "reach_2.5R": vals,
                "reach_3.0R": vals,
                "strategy_version": ["1"] * n,
                "strategy_parameter_hash": ["abc"] * n,
                "strategy_description": ["demo"] * n,
            }
        )

    # ~80% reach at 1R vs RW 50% -> should promote at 1R with n=250
    table = strategy_comparison_table(
        {"edgey": _frame(0.80), "coiny": _frame(0.50)},
        min_edge_pp=5.0,
        min_n=200,
    )
    edge_1r = table[(table["strategy_id"] == "edgey") & (table["target_R"] == 1.0)].iloc[0]
    coin_1r = table[(table["strategy_id"] == "coiny") & (table["target_R"] == 1.0)].iloc[0]
    assert edge_1r["promoted"] is True or edge_1r["promoted"] == True
    assert coin_1r["promoted"] is False or coin_1r["promoted"] == False
