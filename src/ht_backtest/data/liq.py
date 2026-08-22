"""Binance USDM liquidations: event store, 15m bins, causal merge onto bars.

Causal timestamp rule (highest lookahead risk in Phase 1)
---------------------------------------------------------
Force-order events are bucketed into half-open UTC 15m bins
``[bin_open, bin_close)`` with ``bin_close = bin_open + 15m``.

A 15m OHLCV bar with open time T (the decision bar) sees **only** the bin
whose close time equals T — i.e. events in ``[T-15m, T)``. That is an
exact join on ``bin_close == bar.timestamp``, not merge_asof. Liquidations
are flow: an empty window is 0 (inside the observed bin range) or NaN
(outside coverage), never a ffill of the last event.

Consequences:
- A liquidation with timestamp in ``[T, T+15m)`` belongs to the *current*
  bar's still-forming bin (closes at T+15m). It is **invisible** to bar T.
- That same event becomes visible on the *next* bar (open T+15m), when the
  bin has closed.
- An event at exactly T is inside ``[T, T+15m)`` and is also invisible to
  bar T.

Do not attach raw events at bar close — that would leak in-bar liquidations
into the same bar's conditions.

Venue depth (as of 2026-08):
- ``GET /fapi/v1/allForceOrders`` (former public MARKET_DATA) returns **404**.
- ``GET /fapi/v1/forceOrders`` is USER_DATA — API keys are forbidden in this repo.
- ``data.binance.vision`` USDM ``liquidationSnapshot`` prefix is empty
  (files stopped ~2024-03-31, then were removed). COIN-M files still exist
  and are the wrong venue.
- Remaining public feed is websocket ``!forceOrder@arr``. The Phase 1 data
  daemon should append those events into this same parquet schema. Until a
  store is populated, merge yields NaN → conditions None (same v1-train gap
  as OI, plus no recent REST window either).

Side convention (Binance force order):
- side=SELL → longs liquidated
- side=BUY  → shorts liquidated

Storage: ``data/liq/{safe_symbol}.parquet``
columns ``timestamp``, ``side``, ``qty``.
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

LIQ_EVENT_COLUMNS = ["timestamp", "side", "qty"]
LIQ_PERIOD_MS = 15 * 60 * 1000
LIQ_HIST_URL = "https://fapi.binance.com/fapi/v1/allForceOrders"
LIQ_SPIKE_LOOKBACK = 96
LIQ_SPIKE_MIN_PERIODS = 20
LIQ_SPIKE_Q = 0.90


def liq_path(symbol: str, liq_dir: str | Path = "data/liq") -> Path:
    return Path(liq_dir) / f"{_safe_symbol(symbol)}.parquet"


def _binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "").split(":")[0]


def bin_open_ms(ts_ms: int | np.ndarray) -> int | np.ndarray:
    return (np.asarray(ts_ms, dtype=np.int64) // LIQ_PERIOD_MS) * LIQ_PERIOD_MS


def bin_close_ms(ts_ms: int | np.ndarray) -> int | np.ndarray:
    return bin_open_ms(ts_ms) + LIQ_PERIOD_MS


def aggregate_events_to_bins(events: pd.DataFrame) -> pd.DataFrame:
    """Roll force-order events into completed 15m bins.

    Output columns: bin_close, liq_long_qty, liq_short_qty, liq_qty.
    ``bin_close`` is the first instant the bin is fully known.
    """
    cols = ["bin_close", "liq_long_qty", "liq_short_qty", "liq_qty"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    work = events.dropna(subset=["timestamp", "side", "qty"]).copy()
    if work.empty:
        return pd.DataFrame(columns=cols)
    work["timestamp"] = work["timestamp"].astype(np.int64)
    work["qty"] = work["qty"].astype(float)
    work["side"] = work["side"].astype(str).str.upper()
    work["bin_close"] = bin_close_ms(work["timestamp"].to_numpy())
    work["long_qty"] = np.where(work["side"] == "SELL", work["qty"], 0.0)
    work["short_qty"] = np.where(work["side"] == "BUY", work["qty"], 0.0)
    grouped = work.groupby("bin_close", as_index=False).agg(
        liq_long_qty=("long_qty", "sum"),
        liq_short_qty=("short_qty", "sum"),
    )
    grouped["liq_qty"] = grouped["liq_long_qty"] + grouped["liq_short_qty"]
    return grouped.sort_values("bin_close").reset_index(drop=True)


def densify_bins(bins: pd.DataFrame) -> pd.DataFrame:
    """Fill empty 15m slots between the first and last observed bin with zeros.

    Liquidations are flow, not state: an empty window is 0, not a ffill of the
    last event. Bars outside [min bin_close, max bin_close] stay unmatched (NaN)
    — that is the API-depth coverage gap.
    """
    cols = ["bin_close", "liq_long_qty", "liq_short_qty", "liq_qty"]
    if bins is None or bins.empty:
        return pd.DataFrame(columns=cols)
    start = int(bins["bin_close"].min())
    end = int(bins["bin_close"].max())
    idx = np.arange(start, end + 1, LIQ_PERIOD_MS, dtype=np.int64)
    full = pd.DataFrame({"bin_close": idx})
    out = full.merge(bins[cols], on="bin_close", how="left")
    for c in ("liq_long_qty", "liq_short_qty", "liq_qty"):
        out[c] = out[c].fillna(0.0)
    return out


def merge_liq_onto_bars(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    validate_no_row_change: bool = True,
) -> pd.DataFrame:
    """Attach the 15m bin that *closed at this bar's open*.

    Exact join on ``bin_close == bar.timestamp`` — not merge_asof. A later
    empty bar must not inherit a previous window's liquidations. In-bar
    events live in the bin that closes at T+15m and therefore miss bar T.
    """
    if "timestamp" not in bars.columns:
        raise ValueError("bars must have timestamp (ms) column")
    n_before = len(bars)
    ts_before = bars["timestamp"].to_numpy().copy()

    out = bars.copy()
    for c in ("liq_long_qty", "liq_short_qty", "liq_qty"):
        if c in out.columns:
            out = out.drop(columns=[c])

    bins = densify_bins(aggregate_events_to_bins(events))
    if bins.empty:
        out["liq_long_qty"] = np.nan
        out["liq_short_qty"] = np.nan
        out["liq_qty"] = np.nan
        return out

    work = out.reset_index(drop=True)
    right = bins.rename(columns={"bin_close": "timestamp"})
    merged = work.merge(right, on="timestamp", how="left")

    if validate_no_row_change:
        if len(merged) != n_before:
            raise RuntimeError(f"liq merge changed row count: {n_before} → {len(merged)}")
        if not np.array_equal(merged["timestamp"].to_numpy(), ts_before):
            raise RuntimeError("liq merge altered bar timestamps/order")

    return merged


def liq_at_bar_open(bars_with_liq: pd.DataFrame, open_time_ms: int, column: str = "liq_qty") -> float | None:
    hit = bars_with_liq.loc[bars_with_liq["timestamp"] == open_time_ms, column]
    if hit.empty:
        return None
    val = hit.iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def attach_liquidations(
    bars: pd.DataFrame,
    symbol: str,
    liq_dir: str | Path = "data/liq",
) -> pd.DataFrame:
    path = liq_path(symbol, liq_dir)
    if not path.exists():
        out = bars.copy()
        for c in ("liq_long_qty", "liq_short_qty", "liq_qty"):
            if c not in out.columns:
                out[c] = np.nan
        return out
    events = pd.read_parquet(path)
    return merge_liq_onto_bars(bars, events)


def liq_spike_mask(
    liq_qty: pd.Series,
    *,
    lookback: int = LIQ_SPIKE_LOOKBACK,
    min_periods: int = LIQ_SPIKE_MIN_PERIODS,
) -> tuple[pd.Series, pd.Series]:
    """Spike vs trailing p90 of **prior** completed bins only."""
    prior = liq_qty.shift(1)
    p90 = prior.rolling(lookback, min_periods=min_periods).quantile(LIQ_SPIKE_Q)
    defined = liq_qty.notna() & p90.notna()
    return liq_qty > p90, defined


@dataclass
class LiqDownloadResult:
    symbol: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    fetched_new: int


def _http_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "ht-backtest-liq/1"
    try:
        r = s.get("https://fapi.binance.com/fapi/v1/ping", timeout=15)
        r.raise_for_status()
        return s
    except requests.exceptions.SSLError:
        print("WARNING: TLS verify failed for fapi.binance.com; liq download using unverified HTTPS (public GET)")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        s.verify = False
        return s


class LiqDownloader:
    """Best-effort public REST ingest. The historical MARKET_DATA route is dead.

    Class flag ``_rest_unavailable`` is set on the first 404 so a universe
    download does not hammer Binance with 30 identical HTML error pages.
    """

    _rest_unavailable: bool = False

    def __init__(self, liq_dir: str | Path = "data/liq", limit: int = 1000):
        self.liq_dir = Path(liq_dir)
        self.liq_dir.mkdir(parents=True, exist_ok=True)
        self.limit = min(int(limit), 1000)
        self._session = _http_session()

    def load_cached(self, symbol: str) -> pd.DataFrame:
        path = liq_path(symbol, self.liq_dir)
        if not path.exists():
            return pd.DataFrame(columns=LIQ_EVENT_COLUMNS)
        df = pd.read_parquet(path)
        return (
            df[LIQ_EVENT_COLUMNS]
            .drop_duplicates(subset=["timestamp", "side", "qty"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _save(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        path = liq_path(symbol, self.liq_dir)
        out = (
            df[LIQ_EVENT_COLUMNS]
            .drop_duplicates(subset=["timestamp", "side", "qty"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        out.to_parquet(path, index=False)

    def _fetch_page(self, symbol: str, *, end_time: int | None = None) -> list[dict]:
        """Former public ``allForceOrders``. Expect empty: endpoint 404s.

        Do not query USER_DATA ``/fapi/v1/forceOrders`` from this repo.
        """
        if LiqDownloader._rest_unavailable:
            return []
        params: dict[str, int | str] = {"symbol": _binance_symbol(symbol), "limit": self.limit}
        if end_time is not None:
            params["endTime"] = int(end_time)
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                resp = self._session.get(LIQ_HIST_URL, params=params, timeout=30)
                if resp.status_code == 404:
                    LiqDownloader._rest_unavailable = True
                    print(
                        "VENUE GAP: GET /fapi/v1/allForceOrders is gone (404). "
                        "USDM data.binance.vision/liquidationSnapshot is empty. "
                        "/fapi/v1/forceOrders is USER_DATA (keys forbidden here). "
                        "Remaining path: websocket !forceOrder@arr (daemon). "
                        "Empty store → liq conditions None."
                    )
                    return []
                if resp.status_code in (400, 401, 403, 418, 451):
                    print(f"  liq HTTP {resp.status_code} ({resp.text[:120]}); empty")
                    return []
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    return []
                return data
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"Failed liq history for {symbol}: {last_err}")

    def download(self, symbol: str) -> LiqDownloadResult:
        existing = self.load_cached(symbol)
        keyset = (
            set(zip(existing["timestamp"].astype(np.int64), existing["side"], existing["qty"].astype(float)))
            if not existing.empty
            else set()
        )
        new_rows: list[dict] = []
        end_time: int | None = None
        seen_oldest: int | None = None
        # Venue depth is tiny; a handful of backward pages is enough.
        for _ in range(8):
            batch = self._fetch_page(symbol, end_time=end_time)
            if not batch:
                break
            page_ts: list[int] = []
            for row in batch:
                ts = int(row.get("time") or row.get("updateTime") or 0)
                if ts <= 0:
                    continue
                page_ts.append(ts)
                side = str(row.get("side", "")).upper()
                if side not in ("BUY", "SELL"):
                    continue
                qty = float(row.get("executedQty") or row.get("origQty") or 0.0)
                if qty <= 0:
                    continue
                key = (ts, side, qty)
                if key in keyset:
                    continue
                new_rows.append({"timestamp": ts, "side": side, "qty": qty})
                keyset.add(key)
            if not page_ts:
                break
            oldest = min(page_ts)
            if seen_oldest is not None and oldest >= seen_oldest:
                break
            seen_oldest = oldest
            end_time = oldest - 1
            if len(batch) < self.limit:
                break
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
        return LiqDownloadResult(
            symbol=symbol,
            rows=len(final),
            start=start,
            end=end,
            fetched_new=len(new_rows),
        )


def download_universe_liq(
    symbols: list[str],
    liq_dir: str | Path = "data/liq",
    log_fn=print,
) -> list[LiqDownloadResult]:
    dl = LiqDownloader(liq_dir=liq_dir)
    results: list[LiqDownloadResult] = []
    for i, symbol in enumerate(symbols, 1):
        log_fn(f"[{i}/{len(symbols)}] liq {symbol}...")
        r = dl.download(symbol)
        log_fn(f"  -> rows={r.rows} new={r.fetched_new} {r.start} -> {r.end}")
        results.append(r)
        time.sleep(0.2)
    return results
