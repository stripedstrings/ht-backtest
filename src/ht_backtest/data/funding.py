"""Binance USDM funding rates: download, store, causal merge onto 15m bars.

Funding settles every 8h at 00:00 / 08:00 / 16:00 UTC. Each 15m bar carries the
most recently settled rate at that bar's open time (merge_asof backward).
A 09:00 UTC bar therefore holds the 08:00 settlement, never the 16:00 one.

Storage units match Binance/ccxt: decimal fraction of notional per period
(e.g. 0.0001 == 0.01%). Anomaly band: |rate| > 0.001 (== 0.1%) is a data error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from ht_backtest.data.downloader import _safe_symbol

FUNDING_COLUMNS = ["timestamp", "funding_rate"]

# Binance decimal units: 0.0001 == 0.01%, 0.001 == 0.1%
ANOMALY_ABS = 0.001  # |rate| outside ±0.1% → data error
EXTREME_ABS = 0.0001  # |rate| > 0.01% → funding_extreme condition


def funding_path(symbol: str, funding_dir: str | Path = "data/funding") -> Path:
    return Path(funding_dir) / f"{_safe_symbol(symbol)}.parquet"


def rate_to_percent(rate: float) -> float:
    """Decimal notional fraction → percent (0.0001 → 0.01)."""
    return float(rate) * 100.0


@dataclass
class FundingDownloadResult:
    symbol: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    fetched_new: int
    anomaly_count: int


@dataclass
class FundingAnomalyReport:
    symbol: str
    n_rows: int
    n_anomalies: int
    anomalies: pd.DataFrame  # timestamp, funding_rate, funding_rate_pct


class FundingDownloader:
    def __init__(
        self,
        exchange_id: str = "binanceusdm",
        funding_dir: str | Path = "data/funding",
        limit: int = 1000,
    ):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.funding_dir = Path(funding_dir)
        self.funding_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    def load_cached(self, symbol: str) -> pd.DataFrame:
        path = funding_path(symbol, self.funding_dir)
        if not path.exists():
            return pd.DataFrame(columns=FUNDING_COLUMNS)
        df = pd.read_parquet(path)
        return (
            df[FUNDING_COLUMNS]
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _save(self, symbol: str, df: pd.DataFrame) -> None:
        path = funding_path(symbol, self.funding_dir)
        out = (
            df[FUNDING_COLUMNS]
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        out.to_parquet(path, index=False)

    def _fetch_page(self, symbol: str, since_ms: int) -> list[dict]:
        for attempt in range(5):
            try:
                return self.exchange.fetch_funding_rate_history(
                    symbol, since=since_ms, limit=self.limit
                )
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                wait = 2**attempt
                print(f"  funding fetch error ({e}), retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Failed funding history for {symbol} since {since_ms}")

    def download(
        self,
        symbol: str,
        since_ms: int,
        until_ms: int | None = None,
    ) -> FundingDownloadResult:
        """Fetch funding prints in [since_ms, until_ms], merge into parquet cache."""
        until_ms = until_ms if until_ms is not None else int(time.time() * 1000)
        existing = self.load_cached(symbol)
        existing_ts = set(existing["timestamp"].astype(np.int64).tolist()) if not existing.empty else set()

        # Resume forward when early history already covered; otherwise fill from since.
        if (
            not existing.empty
            and int(existing["timestamp"].iloc[0]) <= since_ms
            and int(existing["timestamp"].iloc[-1]) < until_ms
        ):
            cursor = int(existing["timestamp"].iloc[-1]) + 1
        else:
            cursor = since_ms

        new_rows: list[dict] = []
        while cursor <= until_ms:
            batch = self._fetch_page(symbol, cursor)
            if not batch:
                break
            last_ts = int(batch[-1]["timestamp"])
            for row in batch:
                ts = int(row["timestamp"])
                if ts < since_ms or ts > until_ms:
                    continue
                if ts in existing_ts:
                    continue
                new_rows.append(
                    {
                        "timestamp": (ts // 1000) * 1000,  # snap ms noise to whole seconds
                        "funding_rate": float(row["fundingRate"]),
                    }
                )
                existing_ts.add((ts // 1000) * 1000)
            next_cursor = last_ts + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if last_ts >= until_ms:
                break
            if len(batch) < self.limit:
                break

        if new_rows:
            merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            merged = existing

        self._save(symbol, merged)
        final = self.load_cached(symbol)
        anomalies = flag_funding_anomalies(final, symbol)
        start = end = None
        if not final.empty:
            start = pd.Timestamp(int(final["timestamp"].iloc[0]), unit="ms", tz="UTC")
            end = pd.Timestamp(int(final["timestamp"].iloc[-1]), unit="ms", tz="UTC")
        return FundingDownloadResult(
            symbol=symbol,
            rows=len(final),
            start=start,
            end=end,
            fetched_new=len(new_rows),
            anomaly_count=anomalies.n_anomalies,
        )


def download_universe_funding(
    symbols: list[str],
    since_ms: int,
    until_ms: int | None = None,
    funding_dir: str | Path = "data/funding",
    exchange_id: str = "binanceusdm",
    log_fn=print,
) -> list[FundingDownloadResult]:
    dl = FundingDownloader(exchange_id=exchange_id, funding_dir=funding_dir)
    results: list[FundingDownloadResult] = []
    for i, symbol in enumerate(symbols, 1):
        log_fn(f"[{i}/{len(symbols)}] funding {symbol}...")
        r = dl.download(symbol, since_ms=since_ms, until_ms=until_ms)
        flag = f" ANOMALIES={r.anomaly_count}" if r.anomaly_count else ""
        log_fn(f"  → rows={r.rows} new={r.fetched_new} {r.start} → {r.end}{flag}")
        results.append(r)
    return results


def flag_funding_anomalies(df: pd.DataFrame, symbol: str = "") -> FundingAnomalyReport:
    if df.empty:
        return FundingAnomalyReport(symbol, 0, 0, pd.DataFrame())
    rates = df["funding_rate"].astype(float)
    mask = rates.abs() > ANOMALY_ABS
    bad = df.loc[mask, ["timestamp", "funding_rate"]].copy()
    if not bad.empty:
        bad["funding_rate_pct"] = bad["funding_rate"].map(rate_to_percent)
    return FundingAnomalyReport(
        symbol=symbol,
        n_rows=len(df),
        n_anomalies=int(mask.sum()),
        anomalies=bad.reset_index(drop=True),
    )


def merge_funding_onto_bars(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    validate_no_row_change: bool = True,
) -> pd.DataFrame:
    """Attach forward-filled ``funding_rate`` by bar open time (no lookahead).

    Uses merge_asof direction='backward': settlement at T is visible on bars with
    open_time >= T, and not on earlier bars.
    """
    if "timestamp" not in bars.columns:
        raise ValueError("bars must have timestamp (ms) column")
    n_before = len(bars)
    ts_before = bars["timestamp"].to_numpy().copy()

    out = bars.copy()
    if "funding_rate" in out.columns:
        out = out.drop(columns=["funding_rate"])

    if funding is None or len(funding) == 0:
        out["funding_rate"] = np.nan
        return out

    work = out.reset_index(drop=True)
    work["_ord"] = np.arange(len(work), dtype=np.int64)
    left = work.sort_values("timestamp", kind="mergesort")
    right = (
        funding[FUNDING_COLUMNS]
        .dropna(subset=["timestamp", "funding_rate"])
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp", kind="mergesort")
    )
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged = merged.sort_values("_ord", kind="mergesort").drop(columns=["_ord"]).reset_index(drop=True)

    if validate_no_row_change:
        if len(merged) != n_before:
            raise RuntimeError(f"funding merge changed row count: {n_before} → {len(merged)}")
        if not np.array_equal(merged["timestamp"].to_numpy(), ts_before):
            raise RuntimeError("funding merge altered bar timestamps/order")

    return merged


def attach_funding_rate(
    bars: pd.DataFrame,
    symbol: str,
    funding_dir: str | Path = "data/funding",
) -> pd.DataFrame:
    """Load cached funding for ``symbol`` and merge onto bars (row-count safe)."""
    path = funding_path(symbol, funding_dir)
    if not path.exists():
        out = bars.copy()
        if "funding_rate" not in out.columns:
            out["funding_rate"] = np.nan
        return out
    funding = pd.read_parquet(path)
    return merge_funding_onto_bars(bars, funding)


def funding_at_bar_open(
    bars_with_funding: pd.DataFrame, open_time_ms: int
) -> float | None:
    """Return merged funding_rate on the bar whose open equals open_time_ms."""
    hit = bars_with_funding.loc[
        bars_with_funding["timestamp"] == open_time_ms, "funding_rate"
    ]
    if hit.empty:
        return None
    val = hit.iloc[0]
    if pd.isna(val):
        return None
    return float(val)
