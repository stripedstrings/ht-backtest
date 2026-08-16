"""Holy Trinity v1.0 session-range strategy, wrapped behind the Strategy protocol.

The gate logic remains in trades/pipeline.py and trades/state_machine.py.
This module only adapts that existing path so numbers cannot drift from the
pre-refactor FINDINGS.md training table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import pandas as pd

from ht_backtest.data.split import SplitManifest
from ht_backtest.strategies.base import StrategyContext, StrategyMetadata, TradeCandidate
from ht_backtest.trades.pipeline import generate_trades as _ht_generate_trades
from ht_backtest.trades.state_machine import GateParams
from ht_backtest.trades.tagging import assemble_trade_frame

STRATEGY_ID = "holy_trinity_v10"
STRATEGY_VERSION = "1.0.0"

_DESCRIPTION = (
    "At London or NY killzone open, freeze the nearest unswept swing high and "
    "low as the session range. A trade exists only when (1) one of those two "
    "edges is raided inside the session (one raid per side), (2) price closes "
    "back through the protected internal swing (MSS / displacement), and "
    "(3) the displacement leg left a fair-value gap that is later retested at "
    "the FVG edge. Stop sits past the sweep extreme; planned target is the far "
    "edge of the session range. All other concepts are recorded as tags and "
    "never veto a setup."
)


def _parameter_hash(params: GateParams, pipeline_kwargs: dict[str, Any]) -> str:
    payload = {
        "gate_params": asdict(params),
        "pipeline": pipeline_kwargs,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class HolyTrinityV10Strategy:
    """Adapter: Pine v1.0 defaults via GateParams, identical trade population."""

    params: GateParams = field(default_factory=GateParams)
    sw_len: int = 5
    int_len: int = 2
    eq_tol_mult: float = 0.10
    max_liq: int = 8
    one_raid: bool = True
    ctx_bars: int = 40
    d_piv: int = 2

    def _pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "sw_len": self.sw_len,
            "int_len": self.int_len,
            "eq_tol_mult": self.eq_tol_mult,
            "max_liq": self.max_liq,
            "one_raid": self.one_raid,
            "ctx_bars": self.ctx_bars,
            "d_piv": self.d_piv,
        }

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id=STRATEGY_ID,
            version=STRATEGY_VERSION,
            parameter_hash=_parameter_hash(self.params, self._pipeline_kwargs()),
            description=_DESCRIPTION,
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        trades, _session_range = _ht_generate_trades(
            bars,
            params=self.params,
            primitives=getattr(ctx, "primitives", None),
            **self._pipeline_kwargs(),
        )
        sid = self.metadata().id
        return [TradeCandidate.from_legacy_trade(t, strategy_id=sid, symbol=ctx.symbol) for t in trades]

    def tags(
        self,
        trade: TradeCandidate,
        bars: pd.DataFrame,
        ctx: StrategyContext,
    ) -> Mapping[str, Any]:
        # Static per-trade fields already ride in extras from the FSM.
        # Cross-trade median tags are applied in assemble_symbol_frame.
        return {}

    def assemble_symbol_frame(
        self,
        candidates: list[TradeCandidate],
        symbol: str,
        timeframe: str,
        split: SplitManifest,
    ) -> pd.DataFrame:
        """Preserve HT median-tag + session-range column assembly exactly."""
        legacy = [c.to_legacy_dict() for c in candidates]
        return assemble_trade_frame(legacy, symbol, timeframe, split)


def default_holy_trinity_v10() -> HolyTrinityV10Strategy:
    return HolyTrinityV10Strategy()
