"""Binance USDM open interest: download, store, causal merge onto 15m bars.

Causal contract (same as funding): each bar carries the last OI snapshot with
timestamp <= that bar's **open** (pandas merge_asof direction='backward').
A snapshot at 08:15 is invisible to the 08:00 bar.

Binance ``openInterestHist`` typically serves only a rolling ~30-day window
(startTime/endTime span ≤ 30d). Older bars merge to NaN → conditions None.
We still walk backward in 30-day chunks in case the venue returns more.

Storage: ``data/oi/{safe_symbol}.parquet`` columns ``timestamp``, ``open_interest``
(contracts / sumOpenInterest).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

from ht_backtest.data.downloader import _safe_symbol

OI_COLUMNS = ["timestamp", "open_interest"]
OI_TIMEFRAME = "15m"
OI_PERIOD_MS = 15 * 60 * 1000
# Venue rejects startTime older than ~30d; keep a 29d span for margin.
OI_MAX_SPAN_MS = 29 * 24 * 60 * 60 * 1000
OI_EXTREME_LOOKBACK = 96 * 30  # 30d of 15m bars
OI_EXTREME_MIN_PERIODS = 96  # 1d
OI_P90 = 0.90
OI_P10 = 0.10
OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"


def oi_path(symbol: str, oi_dir: str | Path = "data/oi") -> Path:
    return Path(oi_dir) / f"{_safe_symbol(symbol)}.parquet"


@dataclass
class OiDownloadResult:
    symbol: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    fetched_new: int


def _binance_oi_symbol(symbol: str) -> str:
    """BTC/USDT:USDT → BTCUSDT (Binance data-API contract id)."""
    return symbol.replace("/", "").split(":")[0]


def _http_session() -> requests.Session:
    """Public GET session. Falls back to unverified TLS if the system CA store fails."""
    s = requests.Session()
    s.headers["User-Agent"] = "ht-backtest-oi/1"
    try:
        r = s.get("https://fapi.binance.com/fapi/v1/ping", timeout=15)
        r.raise_for_status()
        return s
    except requests.exceptions.SSLError:
        print("WARNING: TLS verify failed for fapi.binance.com; OI download using unverified HTTPS (public GET)")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        s.verify = False
        return s


class OiDownloader:
    def __init__(
        self,
        exchange_id: str = "binanceusdm",
        oi_dir: str | Path = "data/oi",
        limit: int = 500,
        timeframe: str = OI_TIMEFRAME,
    ):
        if timeframe != "15m":
            raise ValueError("OI downloader v1 only stores 15m snapshots")
        self.exchange_id = exchange_id
        self.oi_dir = Path(oi_dir)
        self.oi_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.timeframe = timeframe
        self._session = _http_session()

    def load_cached(self, symbol: str) -> pd.DataFrame:
        path = oi_path(symbol, self.oi_dir)
        if not path.exists():
            return pd.DataFrame(columns=OI_COLUMNS)
        df = pd.read_parquet(path)
        return (
            df[OI_COLUMNS]
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _save(self, symbol: str, df: pd.DataFrame) -> None:
        path = oi_path(symbol, self.oi_dir)
        out = (
            df[OI_COLUMNS]
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        out.to_parquet(path, index=False)

    def _fetch_page(self, symbol: str, since_ms: int, until_ms: int) -> list[dict]:
        now_ms = int(time.time() * 1000)
        min_start = now_ms - OI_MAX_SPAN_MS
        since_ms = max(int(since_ms), min_start)
        until_ms = min(int(until_ms), now_ms)
        if since_ms >= until_ms:
            return []
        params = {
            "symbol": _binance_oi_symbol(symbol),
            "period": self.timeframe,
            "limit": int(self.limit),
            "startTime": since_ms,
            "endTime": until_ms,
        }
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                resp = self._session.get(OI_HIST_URL, params=params, timeout=30)
                if resp.status_code == 400:
                    # Retry without startTime (latest window) — venue rejects stale startTime.
                    params.pop("startTime", None)
                    resp = self._session.get(OI_HIST_URL, params=params, timeout=30)
                    if resp.status_code == 400:
                        print(f"  oi HTTP 400 ({resp.text[:180]}); stopping this window")
                        return []
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    raise RuntimeError(f"unexpected OI payload: {data!r}"[:200])
                out = []
                for row in data:
                    ts = int(row["timestamp"])
                    amount = row.get("sumOpenInterest")
                    if amount is None:
                        continue
                    out.append({"timestamp": ts, "openInterestAmount": float(amount)})
                out.sort(key=lambda r: r["timestamp"])
                return out
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                wait = 2**attempt
                print(f"  oi fetch error ({e}), retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"Failed OI history for {symbol} since {since_ms}: {last_err}")

    def download(
        self,
        symbol: str,
        since_ms: int,
        until_ms: int | None = None,
    ) -> OiDownloadResult:
        """Fetch OI snapshots in [since_ms, until_ms], merge into parquet cache."""
        until_ms = until_ms if until_ms is not None else int(time.time() * 1000)
        existing = self.load_cached(symbol)
        existing_ts = (
            set(existing["timestamp"].astype(np.int64).tolist()) if not existing.empty else set()
        )

        new_rows: list[dict] = []
        empty_windows = 0
        window_end = until_ms
        while window_end >= since_ms:
            window_start = max(since_ms, window_end - OI_MAX_SPAN_MS)
            page_end = window_end
            window_new = 0
            pages = 0
            while page_end > window_start and pages < 12:
                batch = self._fetch_page(symbol, window_start, page_end)
                time.sleep(0.15)
                pages += 1
                if not batch:
                    break
                oldest = int(batch[0]["timestamp"])
                for row in batch:
                    ts = (int(row["timestamp"]) // 1000) * 1000
                    if ts < since_ms or ts > until_ms:
                        continue
                    if ts in existing_ts:
                        continue
                    amount = row.get("openInterestAmount")
                    if amount is None:
                        continue
                    new_rows.append({"timestamp": ts, "open_interest": float(amount)})
                    existing_ts.add(ts)
                    window_new += 1
                # Walk backward: next page ends just before this page's oldest print.
                next_end = oldest - 1
                if next_end >= page_end:
                    break
                page_end = next_end
                if oldest <= window_start:
                    break
                if len(batch) < self.limit:
                    break
            if window_new == 0:
                empty_windows += 1
                if empty_windows >= 2:
                    break
            else:
                empty_windows = 0
            if window_start <= since_ms:
                break
            window_end = window_start - 1

        if new_rows:
            merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            merged = existing

        self._save(symbol, merged)
        final = self.load_cached(symbol)
        start = end = None
        if not final.empty:
            start = pd.Timestamp(int(final["timestamp"].iloc[0]), unit="ms", tz="UTC")
            end = pd.Timestamp(int(final["timestamp"].iloc[-1]), unit="ms", tz="UTC")
        return OiDownloadResult(
            symbol=symbol,
            rows=len(final),
            start=start,
            end=end,
            fetched_new=len(new_rows),
        )


def download_universe_oi(
    symbols: list[str],
    since_ms: int,
    until_ms: int | None = None,
    oi_dir: str | Path = "data/oi",
    exchange_id: str = "binanceusdm",
    log_fn=print,
) -> list[OiDownloadResult]:
    dl = OiDownloader(exchange_id=exchange_id, oi_dir=oi_dir)
    results: list[OiDownloadResult] = []
    for i, symbol in enumerate(symbols, 1):
        log_fn(f"[{i}/{len(symbols)}] oi {symbol}...")
        r = dl.download(symbol, since_ms=since_ms, until_ms=until_ms)
        log_fn(f"  -> rows={r.rows} new={r.fetched_new} {r.start} -> {r.end}")
        results.append(r)
    return results


def merge_oi_onto_bars(
    bars: pd.DataFrame,
    oi: pd.DataFrame,
    *,
    validate_no_row_change: bool = True,
) -> pd.DataFrame:
    """Attach last OI snapshot at or before bar open (no lookahead).

    merge_asof direction='backward': print at T is visible on bars with
    open_time >= T, and not on earlier bars.
    """
    if "timestamp" not in bars.columns:
        raise ValueError("bars must have timestamp (ms) column")
    n_before = len(bars)
    ts_before = bars["timestamp"].to_numpy().copy()

    out = bars.copy()
    if "open_interest" in out.columns:
        out = out.drop(columns=["open_interest"])

    if oi is None or len(oi) == 0:
        out["open_interest"] = np.nan
        return out

    work = out.reset_index(drop=True)
    work["_ord"] = np.arange(len(work), dtype=np.int64)
    left = work.sort_values("timestamp", kind="mergesort")
    right = (
        oi[OI_COLUMNS]
        .dropna(subset=["timestamp", "open_interest"])
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp", kind="mergesort")
    )
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged = merged.sort_values("_ord", kind="mergesort").drop(columns=["_ord"]).reset_index(drop=True)

    if validate_no_row_change:
        if len(merged) != n_before:
            raise RuntimeError(f"oi merge changed row count: {n_before} → {len(merged)}")
        if not np.array_equal(merged["timestamp"].to_numpy(), ts_before):
            raise RuntimeError("oi merge altered bar timestamps/order")

    return merged


def attach_open_interest(
    bars: pd.DataFrame,
    symbol: str,
    oi_dir: str | Path = "data/oi",
) -> pd.DataFrame:
    """Load cached OI for ``symbol`` and merge onto bars (row-count safe)."""
    path = oi_path(symbol, oi_dir)
    if not path.exists():
        out = bars.copy()
        if "open_interest" not in out.columns:
            out["open_interest"] = np.nan
        return out
    oi = pd.read_parquet(path)
    return merge_oi_onto_bars(bars, oi)


def oi_at_bar_open(bars_with_oi: pd.DataFrame, open_time_ms: int) -> float | None:
    hit = bars_with_oi.loc[bars_with_oi["timestamp"] == open_time_ms, "open_interest"]
    if hit.empty:
        return None
    val = hit.iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def oi_extreme_mask(
    oi: pd.Series,
    *,
    lookback: int = OI_EXTREME_LOOKBACK,
    min_periods: int = OI_EXTREME_MIN_PERIODS,
) -> tuple[pd.Series, pd.Series]:
    """Extreme vs trailing p10/p90 of **prior** bars only (no current-bar OI in the window).

    Returns (is_extreme, defined).
    """
    prior = oi.shift(1)
    p90 = prior.rolling(lookback, min_periods=min_periods).quantile(OI_P90)
    p10 = prior.rolling(lookback, min_periods=min_periods).quantile(OI_P10)
    defined = oi.notna() & p90.notna() & p10.notna()
    extreme = (oi > p90) | (oi < p10)
    return extreme, defined
