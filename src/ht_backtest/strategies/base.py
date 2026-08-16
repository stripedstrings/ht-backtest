"""Strategy ingestion contract.

Strategies emit TradeCandidate rows. They never score themselves against
random walk and never decide train vs holdout — those stay in the shared
engine and are non-negotiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import pandas as pd

from ht_backtest.data.split import SplitManifest

# Required TradeCandidate fields that must not live only inside extras.
_CORE_KEYS = frozenset(
    {
        "direction",
        "entry_bar",
        "entry_price",
        "stop_price",
        "risk",
        "strategy_id",
        "symbol",
        "planned_target",
        "extras",
    }
)


@dataclass(frozen=True)
class StrategyMetadata:
    """Identity payload written into every run artifact."""

    id: str
    version: str
    parameter_hash: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "version": self.version,
            "parameter_hash": self.parameter_hash,
            "description": self.description,
        }


@dataclass
class StrategyContext:
    symbol: str
    timeframe: str
    split: SplitManifest | None = None
    exchange_id: str = "binanceusdm"
    primitives: Any = None  # optional SymbolPrimitives; avoid circular import


@dataclass
class TradeCandidate:
    """Minimal trade the shared reach engine can score.

    Extra strategy-specific columns (HT tags raw values, grab bars, etc.)
    go in `extras` and are merged back when assembling a symbol frame.
    """

    direction: str
    entry_bar: int
    entry_price: float
    stop_price: float
    risk: float
    strategy_id: str
    symbol: str = ""
    planned_target: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {self.direction!r}")
        if self.risk is None or not (self.risk > 0):
            raise ValueError(f"risk must be > 0, got {self.risk!r}")
        if self.entry_bar < 0:
            raise ValueError(f"entry_bar must be >= 0, got {self.entry_bar!r}")

    def to_dict(self) -> dict[str, Any]:
        """Flat row: extras first, then core fields (core wins on collision)."""
        row = dict(self.extras)
        row.update(
            {
                "direction": self.direction,
                "entry_bar": self.entry_bar,
                "entry_price": self.entry_price,
                "stop_price": self.stop_price,
                "risk": self.risk,
                "strategy_id": self.strategy_id,
                "symbol": self.symbol,
                "planned_target": self.planned_target,
            }
        )
        # Preserve legacy HT column name used by assemble_trade_frame / live exit.
        if self.planned_target is not None and "target_price" not in row:
            row["target_price"] = self.planned_target
        return row

    def to_legacy_dict(self) -> dict[str, Any]:
        """Reconstruct a pre-Strategy trade dict for HT assemble_trade_frame."""
        row = dict(self.extras)
        row["direction"] = self.direction
        row["entry_bar"] = self.entry_bar
        row["entry_price"] = self.entry_price
        row["stop_price"] = self.stop_price
        row["risk"] = self.risk
        if self.planned_target is not None:
            row["target_price"] = self.planned_target
        elif "target_price" not in row:
            row["target_price"] = None
        return row

    @classmethod
    def from_legacy_trade(
        cls,
        trade: Mapping[str, Any],
        strategy_id: str,
        symbol: str = "",
    ) -> "TradeCandidate":
        extras = {k: v for k, v in trade.items() if k not in _CORE_KEYS}
        # target_price is HT's planned target; keep it in extras too for parity.
        planned = trade.get("planned_target", trade.get("target_price"))
        return cls(
            direction=str(trade["direction"]),
            entry_bar=int(trade["entry_bar"]),
            entry_price=float(trade["entry_price"]),
            stop_price=float(trade["stop_price"]),
            risk=float(trade["risk"]),
            strategy_id=strategy_id,
            symbol=symbol,
            planned_target=None if planned is None else float(planned),
            extras=extras,
        )


@runtime_checkable
class Strategy(Protocol):
    """Plug-in surface for any signal logic.

    Optional extension (not required by the Protocol, used by the runner when
    present): `assemble_symbol_frame(candidates, symbol, timeframe, split)`
    for strategies that need cross-trade tagging (e.g. HT median tags).
    """

    def metadata(self) -> StrategyMetadata:
        """id, version, parameter hash, and plain-English entry logic."""
        ...

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        """Emit candidates for one symbol's OHLCV frame. Never filters by split."""
        ...

    def tags(
        self,
        trade: TradeCandidate,
        bars: pd.DataFrame,
        ctx: StrategyContext,
    ) -> Mapping[str, Any]:
        """Per-trade analysis columns. Must never veto a setup."""
        ...


def default_assemble_symbol_frame(
    strategy: Strategy,
    candidates: list[TradeCandidate],
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    split: SplitManifest,
) -> pd.DataFrame:
    """Generic assemble: flatten candidates, merge strategy.tags, classify split."""
    if not candidates:
        return pd.DataFrame()
    ctx = StrategyContext(symbol=symbol, timeframe=timeframe, split=split)
    rows: list[dict[str, Any]] = []
    for c in candidates:
        row = c.to_dict()
        row["symbol"] = symbol
        row["timeframe"] = timeframe
        row.update(dict(strategy.tags(c, bars, ctx)))
        entry_time = row.get("entry_time")
        if entry_time is None:
            entry_time = int(bars["timestamp"].iloc[int(c.entry_bar)])
            row["entry_time"] = entry_time
        row["split"] = split.classify(symbol, int(entry_time))
        rows.append(row)
    return pd.DataFrame(rows)


def assemble_symbol_trades(
    strategy: Strategy,
    candidates: list[TradeCandidate],
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    split: SplitManifest,
) -> pd.DataFrame:
    """Prefer strategy-specific assemble when provided; else the generic path."""
    custom = getattr(strategy, "assemble_symbol_frame", None)
    if custom is not None:
        return custom(candidates, symbol, timeframe, split)
    return default_assemble_symbol_frame(strategy, candidates, bars, symbol, timeframe, split)
