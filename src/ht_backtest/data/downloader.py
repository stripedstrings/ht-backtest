"""ccxt OHLCV downloader with a local Parquet cache.

Cache layout: {cache_dir}/{exchange}/{symbol}/{timeframe}/{YYYY-MM}.parquet
One file per calendar month, columns: timestamp (ms, int64, UTC), open, high,
low, close, volume. Re-downloading only fetches bars actually missing from
the requested range (gap-aware, not just "newer than the last cached bar"),
then merges and rewrites the affected months' files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "-")


@dataclass
class DownloadResult:
    symbol: str
    timeframe: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    fetched_new_bars: int


class OHLCVDownloader:
    def __init__(self, exchange_id: str = "binanceusdm", cache_dir: str | Path = "data/raw", limit: int = 1000):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.cache_dir = Path(cache_dir)
        self.limit = limit

    def _month_path(self, symbol: str, timeframe: str, period: pd.Period) -> Path:
        d = self.cache_dir / self.exchange_id / _safe_symbol(symbol) / timeframe
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{period}.parquet"

    def _load_month(self, symbol: str, timeframe: str, period: pd.Period) -> pd.DataFrame:
        path = self._month_path(symbol, timeframe, period)
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    def _save_month(self, symbol: str, timeframe: str, period: pd.Period, df: pd.DataFrame) -> None:
        path = self._month_path(symbol, timeframe, period)
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(path, index=False)

    def cached_range(self, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
        """Read whatever is already cached in [since_ms, until_ms)."""
        start = pd.Timestamp(since_ms, unit="ms", tz="UTC").tz_localize(None).to_period("M")
        end = pd.Timestamp(until_ms, unit="ms", tz="UTC").tz_localize(None).to_period("M")
        periods = pd.period_range(start, end, freq="M")
        frames = [self._load_month(symbol, timeframe, p) for p in periods]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] < until_ms)]
        return df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    def _fetch_page(self, symbol: str, timeframe: str, since_ms: int) -> list[list]:
        for attempt in range(5):
            try:
                return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=self.limit)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                wait = 2**attempt
                print(f"  fetch error ({e}), retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {symbol} {timeframe} since {since_ms} after retries")

    def download(self, symbol: str, timeframe: str, since_ms: int, until_ms: int | None = None) -> DownloadResult:
        """Fetch new bars in [since_ms, until_ms) not already cached, merge into
        monthly cache files, and return a summary. until_ms defaults to now.

        Only fetches bars actually MISSING from [since_ms, until_ms) -- i.e. it
        finds every gap in the existing cache within that range (including a
        gap before the first cached bar, e.g. a prior partial-window pull) and
        fetches each one, rather than assuming the cache is one contiguous
        block starting at since_ms."""
        tf_ms = TIMEFRAME_MS[timeframe]
        until_ms = until_ms if until_ms is not None else int(time.time() * 1000)

        cached = self.cached_range(symbol, timeframe, since_ms, until_ms)
        segments: list[tuple[int, int]] = []
        if cached.empty:
            segments.append((since_ms, until_ms))
        else:
            ts = cached["timestamp"].to_numpy()
            if ts[0] > since_ms:
                segments.append((since_ms, int(ts[0])))
            gap_idx = (ts[1:] - ts[:-1]) > tf_ms
            for i in np.nonzero(gap_idx)[0]:
                segments.append((int(ts[i]) + tf_ms, int(ts[i + 1])))
            last_cursor = int(ts[-1]) + tf_ms
            if last_cursor < until_ms:
                segments.append((last_cursor, until_ms))

        new_rows: list[list] = []
        pages = 0
        fetched_new = 0
        for seg_start, seg_end in segments:
            cursor = seg_start
            while cursor < seg_end:
                batch = self._fetch_page(symbol, timeframe, cursor)
                if not batch:
                    break
                batch = [b for b in batch if b[0] < seg_end]
                if not batch:
                    break
                new_rows.extend(batch)
                fetched_new += len(batch)
                pages += 1
                last_ts = batch[-1][0]
                next_cursor = last_ts + tf_ms
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                # Flush every ~50k bars so 1m multi-year pulls don't hold everything in RAM
                # and so interrupted runs leave usable cache.
                if len(new_rows) >= 50_000:
                    new_df = pd.DataFrame(new_rows, columns=OHLCV_COLUMNS)
                    periods = (
                        pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
                        .dt.tz_localize(None)
                        .dt.to_period("M")
                    )
                    for period, month_df in new_df.groupby(periods):
                        existing = self._load_month(symbol, timeframe, period)
                        merged = pd.concat([existing, month_df], ignore_index=True)
                        self._save_month(symbol, timeframe, period, merged)
                    print(
                        f"    ... flushed {len(new_rows)} bars "
                        f"(through {pd.Timestamp(last_ts, unit='ms', tz='UTC')}, pages={pages})",
                        flush=True,
                    )
                    new_rows = []
                # Do NOT stop merely because len(batch) < limit: Binance often
                # returns 1000 even when limit=1500; that is not end-of-history.

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=OHLCV_COLUMNS)
            periods = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.to_period("M")
            for period, month_df in new_df.groupby(periods):
                existing = self._load_month(symbol, timeframe, period)
                merged = pd.concat([existing, month_df], ignore_index=True)
                self._save_month(symbol, timeframe, period, merged)

        final = self.cached_range(symbol, timeframe, since_ms, until_ms)
        return DownloadResult(
            symbol=symbol,
            timeframe=timeframe,
            rows=len(final),
            start=pd.Timestamp(final["timestamp"].min(), unit="ms", tz="UTC") if not final.empty else None,
            end=pd.Timestamp(final["timestamp"].max(), unit="ms", tz="UTC") if not final.empty else None,
            fetched_new_bars=fetched_new,
        )
