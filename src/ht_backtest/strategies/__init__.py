"""Strategy plugins for the reach-vs-random-walk engine."""

from ht_backtest.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyMetadata,
    TradeCandidate,
    align_aux_bars_to_primary,
    assemble_symbol_trades,
    default_assemble_symbol_frame,
    strategy_requires_symbols,
)
from ht_backtest.strategies.holy_trinity_v10 import HolyTrinityV10Strategy
from ht_backtest.strategies.registry import get_strategy, list_strategies, register_strategy

# Importing registry registers baseline strategies as a side effect.
__all__ = [
    "HolyTrinityV10Strategy",
    "Strategy",
    "StrategyContext",
    "StrategyMetadata",
    "TradeCandidate",
    "align_aux_bars_to_primary",
    "assemble_symbol_trades",
    "default_assemble_symbol_frame",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "strategy_requires_symbols",
]
