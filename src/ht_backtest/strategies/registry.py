"""Named strategy constructors for the CLI and batch runner."""

from __future__ import annotations

from collections.abc import Callable

from ht_backtest.strategies.base import Strategy
from ht_backtest.strategies.holy_trinity_v10 import HolyTrinityV10Strategy, default_holy_trinity_v10

StrategyFactory = Callable[[], Strategy]

_REGISTRY: dict[str, StrategyFactory] = {
    "ht_v10": default_holy_trinity_v10,
    "holy_trinity_v10": default_holy_trinity_v10,
}


def list_strategies() -> list[str]:
    return sorted(set(_REGISTRY))


def get_strategy(name: str) -> Strategy:
    key = name.strip()
    if key not in _REGISTRY:
        known = ", ".join(list_strategies())
        raise KeyError(f"unknown strategy {name!r}; known: {known}")
    return _REGISTRY[key]()


def register_strategy(name: str, factory: StrategyFactory) -> None:
    _REGISTRY[name] = factory


# Re-export for callers that want the class directly
__all__ = [
    "HolyTrinityV10Strategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]
