"""4h OHLCV helpers: causal attach of closed-bar features onto 15m frames.

A 4h candle is only visible after it closes. At 15m bar open T we take the
most recent 4h bar with close_time <= T (never the still-forming candle).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader, TIMEFRAME_MS
from ht_backtest.gates.primitives import confirmed_fractal_high, confirmed_fractal_low

TF_4H_MS = TIMEFRAME_MS["4h"]
EMA_SPAN = 20
SWING_N = 2  # fractal half-width; confirmation delay = SWING_N bars


def download_universe_4h(
    symbols: list[str],
    since_ms: int,
    until_ms: int | None = None,
    *,
    cache_dir: str | Path = "data/raw",
    exchange_id: str = "binanceusdm",
    log_fn=print,
) -> None:
    dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)
    for i, symbol in enumerate(symbols, 1):
        r = dl.download(symbol, "4h", since_ms=since_ms, until_ms=until_ms)
        log_fn(f"[{i}/{len(symbols)}] 4h {symbol}: rows={r.rows} new={r.fetched_new_bars} {r.start} → {r.end}")


def load_4h(symbol: str, cache_dir: str | Path = "data/raw", exchange_id: str = "binanceusdm") -> pd.DataFrame:
    dl = OHLCVDownloader(exchange_id=exchange_id, cache_dir=cache_dir)
    return dl.cached_range(symbol, "4h", 0, 4_102_444_800_000)


def ema(series: pd.Series, span: int = EMA_SPAN) -> pd.Series:
    """EMA matching common exchange convention (adjust=False, span)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def closed_4h_feature_table(h4: pd.DataFrame) -> pd.DataFrame:
    """One row per 4h bar, keyed by close_time_ms (when the bar becomes known)."""
    if h4.empty:
        return pd.DataFrame(
            columns=[
                "close_time_ms",
                "open_time_ms",
                "close",
                "ema20",
                "hh_hl",
                "lh_ll",
            ]
        )
    df = h4.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    open_ms = df["timestamp"].astype(np.int64)
    close_ms = open_ms + TF_4H_MS
    close = df["close"].astype(float)
    ema20 = ema(close, EMA_SPAN)

    # Confirmed swings (no lookahead): fractal at k confirmed at k+SWING_N
    conf_hi = confirmed_fractal_high(df["high"], SWING_N).to_numpy()
    conf_lo = confirmed_fractal_low(df["low"], SWING_N).to_numpy()
    hi_px = df["high"].shift(SWING_N).to_numpy()
    lo_px = df["low"].shift(SWING_N).to_numpy()

    swing_hi_times: list[int] = []
    swing_hi_prices: list[float] = []
    swing_lo_times: list[int] = []
    swing_lo_prices: list[float] = []
    hh_hl = np.zeros(len(df), dtype=bool)
    lh_ll = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if conf_hi[i] and not np.isnan(hi_px[i]):
            swing_hi_times.append(int(close_ms.iloc[i]))
            swing_hi_prices.append(float(hi_px[i]))
        if conf_lo[i] and not np.isnan(lo_px[i]):
            swing_lo_times.append(int(close_ms.iloc[i]))
            swing_lo_prices.append(float(lo_px[i]))
        if len(swing_hi_prices) >= 2 and len(swing_lo_prices) >= 2:
            h1, h2 = swing_hi_prices[-1], swing_hi_prices[-2]
            l1, l2 = swing_lo_prices[-1], swing_lo_prices[-2]
            hh_hl[i] = h1 > h2 and l1 > l2
            lh_ll[i] = h1 < h2 and l1 < l2

    return pd.DataFrame(
        {
            "close_time_ms": close_ms.to_numpy(),
            "open_time_ms": open_ms.to_numpy(),
            "close": close.to_numpy(),
            "ema20": ema20.to_numpy(),
            "hh_hl": hh_hl,
            "lh_ll": lh_ll,
        }
    )


def attach_4h_features(
    bars_15m: pd.DataFrame,
    h4: pd.DataFrame,
    *,
    validate_no_row_change: bool = True,
) -> pd.DataFrame:
    """Merge closed-4h EMA and structure onto 15m bars by open time (no lookahead)."""
    n_before = len(bars_15m)
    ts_before = bars_15m["timestamp"].to_numpy().copy()
    out = bars_15m.copy()
    for col in ("htf_4h_ema20", "htf_4h_close", "htf_4h_hh_hl", "htf_4h_lh_ll"):
        if col in out.columns:
            out = out.drop(columns=[col])

    feats = closed_4h_feature_table(h4)
    if feats.empty:
        out["htf_4h_ema20"] = np.nan
        out["htf_4h_close"] = np.nan
        out["htf_4h_hh_hl"] = False
        out["htf_4h_lh_ll"] = False
        return out

    work = out.reset_index(drop=True)
    work["_ord"] = np.arange(len(work), dtype=np.int64)
    left = work.sort_values("timestamp", kind="mergesort")
    right = feats.rename(columns={"close_time_ms": "timestamp"}).sort_values(
        "timestamp", kind="mergesort"
    )
    # Only columns we need from right
    right = right[
        ["timestamp", "close", "ema20", "hh_hl", "lh_ll"]
    ].rename(
        columns={
            "close": "htf_4h_close",
            "ema20": "htf_4h_ema20",
            "hh_hl": "htf_4h_hh_hl",
            "lh_ll": "htf_4h_lh_ll",
        }
    )
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged = merged.sort_values("_ord", kind="mergesort").drop(columns=["_ord"]).reset_index(drop=True)

    if validate_no_row_change:
        if len(merged) != n_before:
            raise RuntimeError(f"4h merge changed row count: {n_before} → {len(merged)}")
        if not np.array_equal(merged["timestamp"].to_numpy(), ts_before):
            raise RuntimeError("4h merge altered bar timestamps/order")

    return merged


def attach_4h_for_symbol(
    bars_15m: pd.DataFrame,
    symbol: str,
    *,
    cache_dir: str | Path = "data/raw",
    exchange_id: str = "binanceusdm",
) -> pd.DataFrame:
    h4 = load_4h(symbol, cache_dir=cache_dir, exchange_id=exchange_id)
    return attach_4h_features(bars_15m, h4)
