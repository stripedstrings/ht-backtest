"""Structural train/holdout split manifest.

Holdout = (bars belonging to a held-out symbol) UNION (bars in the most
recent date_holdout_fraction of the universe's overall calendar span, for
every symbol, train or holdout). Train = train-symbol bars strictly before
the date cutoff. The cutoff is a single global date, not per-symbol, so no
symbol's future regime leaks into another symbol's training window.

This manifest is written once, before any tag discovery, and every report
command must pass --split <name> and load this file rather than touch raw
data directly.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.universe import load_universe


@dataclass
class SplitManifest:
    seed: int
    timeframe: str
    symbol_holdout_fraction: float
    date_holdout_fraction: float
    universe: list[str]
    holdout_symbols: list[str]
    train_symbols: list[str]
    overall_start_ms: int
    overall_end_ms: int
    date_holdout_start_ms: int

    def classify(self, symbol: str, timestamp_ms: int) -> str:
        if symbol in self.holdout_symbols:
            return "holdout"
        if timestamp_ms >= self.date_holdout_start_ms:
            return "holdout"
        if symbol in self.train_symbols:
            return "train"
        raise ValueError(f"symbol {symbol} is not in this split's universe")

    def split_frame(self, symbol: str, df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
        out = df.copy()
        is_holdout_symbol = symbol in self.holdout_symbols
        out["split"] = "holdout" if is_holdout_symbol else "train"
        if not is_holdout_symbol:
            out.loc[out[ts_col] >= self.date_holdout_start_ms, "split"] = "holdout"
        return out

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "timeframe": self.timeframe,
            "symbol_holdout_fraction": self.symbol_holdout_fraction,
            "date_holdout_fraction": self.date_holdout_fraction,
            "universe": self.universe,
            "holdout_symbols": self.holdout_symbols,
            "train_symbols": self.train_symbols,
            "overall_start": pd.Timestamp(self.overall_start_ms, unit="ms", tz="UTC").isoformat(),
            "overall_end": pd.Timestamp(self.overall_end_ms, unit="ms", tz="UTC").isoformat(),
            "date_holdout_start": pd.Timestamp(self.date_holdout_start_ms, unit="ms", tz="UTC").isoformat(),
            "overall_start_ms": self.overall_start_ms,
            "overall_end_ms": self.overall_end_ms,
            "date_holdout_start_ms": self.date_holdout_start_ms,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        with open(path) as f:
            d = json.load(f)
        return cls(
            seed=d["seed"],
            timeframe=d["timeframe"],
            symbol_holdout_fraction=d["symbol_holdout_fraction"],
            date_holdout_fraction=d["date_holdout_fraction"],
            universe=d["universe"],
            holdout_symbols=d["holdout_symbols"],
            train_symbols=d["train_symbols"],
            overall_start_ms=d["overall_start_ms"],
            overall_end_ms=d["overall_end_ms"],
            date_holdout_start_ms=d["date_holdout_start_ms"],
        )


def build_split_manifest(
    universe_path: str | Path,
    timeframe: str,
    seed: int,
    symbol_holdout_fraction: float = 0.30,
    date_holdout_fraction: float = 0.20,
    exchange_id: str = "binanceusdm",
    cache_dir: str | Path = "data/raw",
) -> SplitManifest:
    entries = load_universe(universe_path)
    symbols = [e.symbol for e in entries]
    dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)

    starts, ends = [], []
    for entry in entries:
        since_ms = int(pd.Timestamp(entry.since, tz="UTC").timestamp() * 1000)
        until_ms = (
            int(pd.Timestamp(entry.until, tz="UTC").timestamp() * 1000)
            if entry.until
            else int(pd.Timestamp.utcnow().timestamp() * 1000)
        )
        df = dl.cached_range(entry.symbol, timeframe, since_ms, until_ms)
        if df.empty:
            raise ValueError(f"no cached data for {entry.symbol} {timeframe} -- run the universe download first")
        starts.append(int(df["timestamp"].min()))
        ends.append(int(df["timestamp"].max()))

    overall_start_ms = min(starts)
    overall_end_ms = max(ends)
    span = overall_end_ms - overall_start_ms
    date_holdout_start_ms = overall_end_ms - int(date_holdout_fraction * span)

    n_holdout = round(symbol_holdout_fraction * len(symbols))
    holdout_symbols = sorted(random.Random(seed).sample(sorted(symbols), k=n_holdout))
    train_symbols = sorted(s for s in symbols if s not in holdout_symbols)

    return SplitManifest(
        seed=seed,
        timeframe=timeframe,
        symbol_holdout_fraction=symbol_holdout_fraction,
        date_holdout_fraction=date_holdout_fraction,
        universe=sorted(symbols),
        holdout_symbols=holdout_symbols,
        train_symbols=train_symbols,
        overall_start_ms=overall_start_ms,
        overall_end_ms=overall_end_ms,
        date_holdout_start_ms=date_holdout_start_ms,
    )
