"""Compile intake candidate YAML into a runnable Strategy instance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ht_backtest.strategies.base import StrategyContext, StrategyMetadata, TradeCandidate
from ht_backtest.strategies.hypothesis_helpers import (
    RECLAIM_WIN,
    collect_raid_events,
    ensure_primitives,
    hash_params,
    make_reclaim_trade,
    session_range_frame,
)
from ht_backtest.strategies.hypotheses import (
    FailedRaidNextSessionFadeStrategy,
    HighVolGrabReclaimStrategy,
    KzFirst30mRaidReclaimStrategy,
    KzFirstRaidReclaimStrategy,
    LondonNySameDirectionStrategy,
    LowVolGrabReclaimStrategy,
    STRATEGY_VERSION,
)

# dry_count.method → (factory, canonical registry id for memory/golden alignment)
_METHOD_MAP: dict[str, tuple[type, str]] = {
    "kz_first_raid_reclaim": (KzFirstRaidReclaimStrategy, "kz_first_raid_reclaim"),
    "kz_first_30m_raid_reclaim": (KzFirst30mRaidReclaimStrategy, "kz_first_30m_raid_reclaim"),
    "raid_reclaim_vol_high": (HighVolGrabReclaimStrategy, "high_vol_grab_reclaim"),
    "raid_reclaim_vol_low": (LowVolGrabReclaimStrategy, "low_vol_grab_reclaim"),
    "london_ny_same_direction": (LondonNySameDirectionStrategy, "london_ny_same_direction"),
    "failed_raid_next_session_fade": (FailedRaidNextSessionFadeStrategy, "failed_raid_next_session_fade"),
}


@dataclass
class RaidReclaimAllStrategy:
    """Allowlisted compiler target: any session-range raid + reclaim."""

    reclaim_win: int = RECLAIM_WIN
    version: str = STRATEGY_VERSION
    theoretical_category: str = "pattern"
    signal_type: str = "raid_reclaim_all"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="raid_reclaim_all",
            version=self.version,
            parameter_hash=hash_params({"id": "raid_reclaim_all", "reclaim_win": self.reclaim_win}),
            description=(
                "Why it might beat a coin: baseline raid-and-reclaim without first-of-session "
                "or volume filters — used as a compiler target for audit candidates. "
                f"Mechanics: any session-range edge raid with reclaim within {self.reclaim_win} bars."
            ),
        )

    def generate_trades(self, bars, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        for e in collect_raid_events(bars, prim, sr):
            if e.reclaim_bar is None:
                continue
            t = make_reclaim_trade(bars, ctx, sid, e)
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars, ctx: StrategyContext) -> Mapping[str, Any]:
        return {}


_METHOD_MAP["raid_reclaim_all"] = (RaidReclaimAllStrategy, "raid_reclaim_all")


def register_compiler_targets() -> None:
    """Ensure allowlisted compile targets exist in the process registry (for workers>1)."""
    from ht_backtest.strategies.registry import list_strategies, register_strategy

    if "raid_reclaim_all" not in list_strategies():
        register_strategy("raid_reclaim_all", RaidReclaimAllStrategy)


@dataclass
class CompiledCandidateStrategy:
    """Adapter: inner allowlisted strategy + candidate metadata overlay."""

    inner: Any
    candidate: dict[str, Any]
    registry_id: str

    def __post_init__(self) -> None:
        self.theoretical_category = str(
            self.candidate.get("theoretical_category")
            or getattr(self.inner, "theoretical_category", "pattern")
        )
        self.signal_type = str(
            self.candidate.get("signal_type") or getattr(self.inner, "signal_type", self.registry_id)
        )
        self.requires_symbols = tuple(getattr(self.inner, "requires_symbols", ()) or ())

    def metadata(self) -> StrategyMetadata:
        inner_meta = self.inner.metadata()
        desc = str(self.candidate.get("plain_english_description") or inner_meta.description)
        return StrategyMetadata(
            id=self.registry_id,
            version=inner_meta.version,
            parameter_hash=inner_meta.parameter_hash,
            description=desc,
        )

    def generate_trades(self, bars, ctx: StrategyContext) -> list[TradeCandidate]:
        trades = self.inner.generate_trades(bars, ctx)
        sid = self.metadata().id
        # Ensure strategy_id on candidates matches compiled id
        out: list[TradeCandidate] = []
        for t in trades:
            out.append(
                TradeCandidate(
                    direction=t.direction,
                    entry_bar=t.entry_bar,
                    entry_price=t.entry_price,
                    stop_price=t.stop_price,
                    risk=t.risk,
                    strategy_id=sid,
                    symbol=t.symbol or ctx.symbol,
                    planned_target=t.planned_target,
                    extras=dict(t.extras),
                )
            )
        return out

    def tags(self, trade: TradeCandidate, bars, ctx: StrategyContext) -> Mapping[str, Any]:
        return self.inner.tags(trade, bars, ctx)


class CompileError(ValueError):
    pass


def load_candidate_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CompileError(f"candidate YAML must be a mapping: {path}")
    return raw


def compile_candidate(candidate: dict[str, Any]) -> CompiledCandidateStrategy:
    """Turn candidate YAML dict into a Strategy. Raises CompileError on failure."""
    dc = candidate.get("dry_count") or {}
    method = dc.get("method")
    if not method:
        raise CompileError("dry_count.method missing — cannot compile")
    if method not in _METHOD_MAP:
        raise CompileError(f"unsupported dry_count.method for compile: {method!r}")

    cls, registry_id = _METHOD_MAP[method]
    params = dict(dc.get("params") or {})
    # Prefer reclaim_bars from dry_count params when the class accepts reclaim_win
    kwargs: dict[str, Any] = {}
    if "reclaim_bars" in params and "reclaim_win" in getattr(cls, "__dataclass_fields__", {}):
        kwargs["reclaim_win"] = int(params["reclaim_bars"])
    try:
        inner = cls(**kwargs) if kwargs else cls()
    except TypeError as exc:
        raise CompileError(f"failed to construct {cls.__name__}: {exc}") from exc

    # Optional: force registry id from candidate when it already matches a known id
    cid = str(candidate.get("candidate_id") or "")
    if cid in {registry_id, method}:
        registry_id = cid if cid == registry_id else registry_id

    return CompiledCandidateStrategy(inner=inner, candidate=candidate, registry_id=registry_id)


def compile_candidate_file(path: str | Path) -> tuple[dict[str, Any], CompiledCandidateStrategy]:
    candidate = load_candidate_yaml(path)
    return candidate, compile_candidate(candidate)
