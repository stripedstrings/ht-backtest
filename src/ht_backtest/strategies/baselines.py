"""Simple baseline strategies for batch plumbing (not claimed edges).

These exist so Phase B can queue ~10 strategies against the same locked
split and produce a comparison table. Each emits TradeCandidates only;
reach-vs-RW and train/holdout stay in the shared engine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from ht_backtest.gates.primitives import compute_atr, session_tags
from ht_backtest.strategies.base import StrategyContext, StrategyMetadata, TradeCandidate


def _hash_params(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _atr_stop(entry: float, atr: float, direction: str, mult: float) -> tuple[float, float]:
    risk = mult * atr
    if risk <= 0 or np.isnan(risk):
        return float("nan"), float("nan")
    if direction == "long":
        return entry - risk, risk
    return entry + risk, risk


def _session_open_trades(
    bars: pd.DataFrame,
    ctx: StrategyContext,
    strategy_id: str,
    session: str,
    direction: str,
    atr_mult: float,
    cooldown_bars: int,
) -> list[TradeCandidate]:
    atr = compute_atr(bars, 14).to_numpy()
    sessions = session_tags(bars["timestamp"])
    start_col = "london_start" if session == "london" else "ny_start"
    starts = sessions[start_col].to_numpy()
    close = bars["close"].to_numpy()
    ts = bars["timestamp"].to_numpy()
    out: list[TradeCandidate] = []
    next_ok = 0
    for i in range(len(bars)):
        if i < next_ok or not starts[i] or np.isnan(atr[i]):
            continue
        entry = float(close[i])
        stop, risk = _atr_stop(entry, float(atr[i]), direction, atr_mult)
        if not (risk > 0):
            continue
        out.append(
            TradeCandidate(
                direction=direction,
                entry_bar=i,
                entry_price=entry,
                stop_price=stop,
                risk=risk,
                strategy_id=strategy_id,
                symbol=ctx.symbol,
                planned_target=entry + (risk if direction == "long" else -risk),
                extras={"entry_time": int(ts[i]), "session": session.upper()},
            )
        )
        next_ok = i + cooldown_bars
    return out


@dataclass
class SessionOpenATRStrategy:
    session: str  # "london" | "ny"
    direction: str  # "long" | "short"
    atr_mult: float = 1.0
    cooldown_bars: int = 96  # ~1 day on 15m
    version: str = "1.0.0"

    @property
    def _id(self) -> str:
        return f"{self.session}_open_{self.direction}_atr{self.atr_mult:g}"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id=self._id,
            version=self.version,
            parameter_hash=_hash_params(
                {
                    "session": self.session,
                    "direction": self.direction,
                    "atr_mult": self.atr_mult,
                    "cooldown_bars": self.cooldown_bars,
                }
            ),
            description=(
                f"At each {self.session.upper()} killzone open, enter {self.direction} "
                f"at the bar close with a stop {self.atr_mult:g}×ATR(14) away and a "
                f"1R planned target. Cooldown {self.cooldown_bars} bars. Baseline only."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        return _session_open_trades(
            bars,
            ctx,
            self.metadata().id,
            self.session,
            self.direction,
            self.atr_mult,
            self.cooldown_bars,
        )

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return {}


@dataclass
class DonchianBreakoutStrategy:
    direction: str
    lookback: int = 20
    atr_mult: float = 1.5
    cooldown_bars: int = 32
    version: str = "1.0.0"

    @property
    def _id(self) -> str:
        return f"donchian_{self.lookback}_{self.direction}"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id=self._id,
            version=self.version,
            parameter_hash=_hash_params(
                {
                    "direction": self.direction,
                    "lookback": self.lookback,
                    "atr_mult": self.atr_mult,
                    "cooldown_bars": self.cooldown_bars,
                }
            ),
            description=(
                f"Enter {self.direction} when close breaks the prior {self.lookback}-bar "
                f"{'high' if self.direction == 'long' else 'low'}; stop {self.atr_mult:g}×ATR(14). "
                "Baseline donchian breakout."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        atr = compute_atr(bars, 14).to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        close = bars["close"].to_numpy()
        ts = bars["timestamp"].to_numpy()
        n = len(bars)
        out: list[TradeCandidate] = []
        next_ok = self.lookback
        for i in range(self.lookback, n):
            if i < next_ok or np.isnan(atr[i]):
                continue
            window_hi = float(np.max(high[i - self.lookback : i]))
            window_lo = float(np.min(low[i - self.lookback : i]))
            fired = close[i] > window_hi if self.direction == "long" else close[i] < window_lo
            if not fired:
                continue
            entry = float(close[i])
            stop, risk = _atr_stop(entry, float(atr[i]), self.direction, self.atr_mult)
            if not (risk > 0):
                continue
            sid = self.metadata().id
            out.append(
                TradeCandidate(
                    direction=self.direction,
                    entry_bar=i,
                    entry_price=entry,
                    stop_price=stop,
                    risk=risk,
                    strategy_id=sid,
                    symbol=ctx.symbol,
                    planned_target=entry + (risk if self.direction == "long" else -risk),
                    extras={"entry_time": int(ts[i]), "donchian_lookback": self.lookback},
                )
            )
            next_ok = i + self.cooldown_bars
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return {}


@dataclass
class SMAMomentumStrategy:
    direction: str
    sma_len: int = 50
    atr_mult: float = 1.0
    cooldown_bars: int = 48
    version: str = "1.0.0"

    @property
    def _id(self) -> str:
        return f"sma{self.sma_len}_mom_{self.direction}"

    def metadata(self) -> StrategyMetadata:
        side = "above" if self.direction == "long" else "below"
        return StrategyMetadata(
            id=self._id,
            version=self.version,
            parameter_hash=_hash_params(
                {
                    "direction": self.direction,
                    "sma_len": self.sma_len,
                    "atr_mult": self.atr_mult,
                    "cooldown_bars": self.cooldown_bars,
                }
            ),
            description=(
                f"Enter {self.direction} when close crosses {side} SMA({self.sma_len}); "
                f"stop {self.atr_mult:g}×ATR(14). Baseline momentum."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        atr = compute_atr(bars, 14)
        sma = bars["close"].rolling(self.sma_len).mean()
        close = bars["close"]
        prev_close = close.shift(1)
        prev_sma = sma.shift(1)
        if self.direction == "long":
            cross = (prev_close <= prev_sma) & (close > sma)
        else:
            cross = (prev_close >= prev_sma) & (close < sma)
        atr_v = atr.to_numpy()
        close_v = close.to_numpy()
        ts = bars["timestamp"].to_numpy()
        flags = cross.fillna(False).to_numpy()
        out: list[TradeCandidate] = []
        next_ok = self.sma_len
        for i in range(len(bars)):
            if i < next_ok or not flags[i] or np.isnan(atr_v[i]):
                continue
            entry = float(close_v[i])
            stop, risk = _atr_stop(entry, float(atr_v[i]), self.direction, self.atr_mult)
            if not (risk > 0):
                continue
            sid = self.metadata().id
            out.append(
                TradeCandidate(
                    direction=self.direction,
                    entry_bar=i,
                    entry_price=entry,
                    stop_price=stop,
                    risk=risk,
                    strategy_id=sid,
                    symbol=ctx.symbol,
                    planned_target=entry + (risk if self.direction == "long" else -risk),
                    extras={"entry_time": int(ts[i]), "sma_len": self.sma_len},
                )
            )
            next_ok = i + self.cooldown_bars
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return {}


@dataclass
class MeanReversionExtremaStrategy:
    """Fade a prior-bar range extreme: long after a bar that made an N-bar low."""

    direction: str
    lookback: int = 10
    atr_mult: float = 1.0
    cooldown_bars: int = 24
    version: str = "1.0.0"

    @property
    def _id(self) -> str:
        return f"mr_extrema_{self.lookback}_{self.direction}"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id=self._id,
            version=self.version,
            parameter_hash=_hash_params(
                {
                    "direction": self.direction,
                    "lookback": self.lookback,
                    "atr_mult": self.atr_mult,
                    "cooldown_bars": self.cooldown_bars,
                }
            ),
            description=(
                f"Mean-reversion baseline: enter {self.direction} when the prior bar "
                f"printed the {self.lookback}-bar {'low' if self.direction == 'long' else 'high'}; "
                f"stop {self.atr_mult:g}×ATR(14)."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        atr = compute_atr(bars, 14).to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        close = bars["close"].to_numpy()
        ts = bars["timestamp"].to_numpy()
        out: list[TradeCandidate] = []
        next_ok = self.lookback + 1
        for i in range(self.lookback + 1, len(bars)):
            if i < next_ok or np.isnan(atr[i]):
                continue
            prev = i - 1
            if self.direction == "long":
                fired = low[prev] <= float(np.min(low[prev - self.lookback + 1 : prev + 1]))
            else:
                fired = high[prev] >= float(np.max(high[prev - self.lookback + 1 : prev + 1]))
            if not fired:
                continue
            entry = float(close[i])
            stop, risk = _atr_stop(entry, float(atr[i]), self.direction, self.atr_mult)
            if not (risk > 0):
                continue
            sid = self.metadata().id
            out.append(
                TradeCandidate(
                    direction=self.direction,
                    entry_bar=i,
                    entry_price=entry,
                    stop_price=stop,
                    risk=risk,
                    strategy_id=sid,
                    symbol=ctx.symbol,
                    planned_target=entry + (risk if self.direction == "long" else -risk),
                    extras={"entry_time": int(ts[i]), "mr_lookback": self.lookback},
                )
            )
            next_ok = i + self.cooldown_bars
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return {}


def _default_baselines_path() -> Path:
    # src/ht_backtest/strategies/baselines.py -> repo root specs/batch/baselines.yaml
    return Path(__file__).resolve().parents[3] / "specs" / "batch" / "baselines.yaml"


def _factory_from_spec(spec: dict[str, Any]):
    cls = spec["class"]
    if cls == "SessionOpenATRStrategy":
        return lambda s=spec: SessionOpenATRStrategy(
            session=s["session"],
            direction=s["direction"],
            atr_mult=float(s.get("atr_mult", 1.0)),
            cooldown_bars=int(s.get("cooldown_bars", 96)),
            version=str(s.get("version", "1.0.0")),
        )
    if cls == "DonchianBreakoutStrategy":
        return lambda s=spec: DonchianBreakoutStrategy(
            direction=s["direction"],
            lookback=int(s.get("lookback", 20)),
            atr_mult=float(s.get("atr_mult", 1.5)),
            cooldown_bars=int(s.get("cooldown_bars", 32)),
            version=str(s.get("version", "1.0.0")),
        )
    if cls == "SMAMomentumStrategy":
        return lambda s=spec: SMAMomentumStrategy(
            direction=s["direction"],
            sma_len=int(s.get("sma_len", 50)),
            atr_mult=float(s.get("atr_mult", 1.0)),
            cooldown_bars=int(s.get("cooldown_bars", 48)),
            version=str(s.get("version", "1.0.0")),
        )
    if cls == "MeanReversionExtremaStrategy":
        return lambda s=spec: MeanReversionExtremaStrategy(
            direction=s["direction"],
            lookback=int(s.get("lookback", 10)),
            atr_mult=float(s.get("atr_mult", 1.0)),
            cooldown_bars=int(s.get("cooldown_bars", 24)),
            version=str(s.get("version", "1.0.0")),
        )
    raise ValueError(f"unknown baseline class {cls!r}")


def register_baselines(register_fn, path: str | Path | None = None) -> None:
    """Register baseline factories from specs/batch/baselines.yaml."""
    import yaml

    yaml_path = Path(path) if path else _default_baselines_path()
    if not yaml_path.exists():
        raise FileNotFoundError(f"baseline spec not found: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    for spec in raw.get("baselines", []):
        # stamp file-level version onto each factory metadata via class default
        if "version" not in spec and raw.get("version"):
            spec = {**spec, "version": raw["version"]}
        register_fn(spec["id"], _factory_from_spec(spec))
