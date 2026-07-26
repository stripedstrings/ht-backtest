"""Batch download + validate for the multi-symbol universe defined in
specs/universe.json. Each entry pulls maximum available history unless an
explicit since/until override is given (e.g. a truncated dead market)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.validator import validate_ohlcv

DEFAULT_UNTIL = None  # None means "now" at call time


@dataclass
class UniverseEntry:
    symbol: str
    since: str
    until: str | None
    requested_as: str | None
    note: str | None


def load_universe(path: str | Path) -> list[UniverseEntry]:
    with open(path) as f:
        cfg = json.load(f)
    default_since = cfg["default_since"]
    return [
        UniverseEntry(
            symbol=s["symbol"],
            since=s.get("since", default_since),
            until=s.get("until"),
            requested_as=s.get("requested_as"),
            note=s.get("note"),
        )
        for s in cfg["symbols"]
    ]


def _to_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def pull_and_validate_universe(
    universe_path: str | Path,
    timeframe: str,
    exchange_id: str = "binanceusdm",
    cache_dir: str | Path = "data/raw",
    reports_dir: str | Path = "data/reports",
    log_fn=print,
) -> pd.DataFrame:
    entries = load_universe(universe_path)
    dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, entry in enumerate(entries, 1):
        since_ms = _to_ms(entry.since)
        until_ms = _to_ms(entry.until) if entry.until else int(time.time() * 1000)
        t0 = time.time()
        log_fn(f"[{i}/{len(entries)}] downloading {entry.symbol} {timeframe} since {entry.since}"
               f"{' until ' + entry.until if entry.until else ''} ...")
        result = dl.download(entry.symbol, timeframe, since_ms, until_ms)
        df = dl.cached_range(entry.symbol, timeframe, since_ms, until_ms)
        elapsed = time.time() - t0

        if df.empty:
            log_fn(f"    -> NO DATA ({elapsed:.1f}s)")
            rows.append(
                {
                    "symbol": entry.symbol,
                    "requested_as": entry.requested_as,
                    "earliest": None,
                    "latest": None,
                    "bars": 0,
                    "duplicates": None,
                    "gaps": None,
                    "missing_bars": None,
                    "impossible_ohlc": None,
                    "non_positive_price": None,
                    "dead_runs": None,
                    "dead_bars": None,
                    "extreme_jumps": None,
                    "outlier_wicks": None,
                    "note": entry.note,
                }
            )
            continue

        report = validate_ohlcv(df, entry.symbol, timeframe)
        safe_name = entry.symbol.replace("/", "_").replace(":", "-")
        with open(reports_dir / f"{safe_name}_{timeframe}_validation.json", "w") as f:
            json.dump(
                {
                    "symbol": report.symbol,
                    "total_bars": report.total_bars,
                    "duplicates": len(report.duplicates),
                    "gaps": report.gaps.to_dict(orient="records"),
                    "missing_bars": report.missing_bar_count(),
                    "impossible_ohlc": report.impossible_ohlc[["timestamp"]].to_dict(orient="records"),
                    "non_positive_price": report.non_positive_price[["timestamp"]].to_dict(orient="records"),
                    "dead_runs": report.dead_runs.to_dict(orient="records"),
                    "extreme_jumps": report.extreme_jumps[["timestamp", "log_return", "zscore"]].to_dict(orient="records"),
                    "outlier_wicks": report.outlier_wicks[["timestamp", "wick_up", "wick_dn", "atr"]].to_dict(orient="records"),
                },
                f,
                indent=2,
                default=str,
            )

        log_fn(
            f"    -> {report.total_bars} bars [{pd.Timestamp(df['timestamp'].min(), unit='ms', tz='UTC').date()} .. "
            f"{pd.Timestamp(df['timestamp'].max(), unit='ms', tz='UTC').date()}], "
            f"{result.fetched_new_bars} new this run, flagged {report.flagged_bar_count} "
            f"({elapsed:.1f}s)"
        )
        rows.append(
            {
                "symbol": entry.symbol,
                "requested_as": entry.requested_as,
                "earliest": pd.Timestamp(df["timestamp"].min(), unit="ms", tz="UTC").date().isoformat(),
                "latest": pd.Timestamp(df["timestamp"].max(), unit="ms", tz="UTC").date().isoformat(),
                "bars": report.total_bars,
                "duplicates": len(report.duplicates),
                "gaps": len(report.gaps),
                "missing_bars": report.missing_bar_count(),
                "impossible_ohlc": len(report.impossible_ohlc),
                "non_positive_price": len(report.non_positive_price),
                "dead_runs": len(report.dead_runs),
                "dead_bars": report.dead_bar_count(),
                "extreme_jumps": len(report.extreme_jumps),
                "outlier_wicks": len(report.outlier_wicks),
                "note": entry.note,
            }
        )

    return pd.DataFrame(rows)
