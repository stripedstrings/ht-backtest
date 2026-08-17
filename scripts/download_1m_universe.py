"""Download 1m OHLCV for the v1 universe; validate BTC; report storage/timing.

Estimated from 15m density before start; prints actual wall clock at end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.data.downloader import OHLCVDownloader  # noqa: E402
from ht_backtest.data.split import SplitManifest  # noqa: E402
from ht_backtest.data.universe import load_universe, pull_and_validate_universe  # noqa: E402
from ht_backtest.data.validator import validate_ohlcv  # noqa: E402

CACHE = ROOT / "data" / "raw"
TF = "1m"
# Binance USDM klines allow up to 1500; larger pages cut wall clock.
PAGE_LIMIT = 1500


def _dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def estimate() -> dict:
    split = SplitManifest.load(ROOT / "specs" / "splits" / "v1.json")
    dl = OHLCVDownloader(cache_dir=CACHE, limit=PAGE_LIMIT)
    sizes = []
    for s in split.universe:
        df = dl.cached_range(s, "15m", 0, 4_102_444_800_000)
        sizes.append(len(df))
    total_15 = sum(sizes)
    est_1m = total_15 * 15
    pages = est_1m / PAGE_LIMIT
    # enableRateLimit ~50-120ms binance; use 0.2s conservative with retries
    est_min = pages * 0.20 / 60.0
    return {
        "total_15m_bars": total_15,
        "est_1m_bars": est_1m,
        "est_pages": pages,
        "est_minutes": est_min,
        "est_gb_rough": est_1m * 48 / (1024**3),  # ~48 bytes/row parquet rough
    }


def validate_btc(dl: OHLCVDownloader) -> None:
    sym = "BTC/USDT:USDT"
    # full cached span
    df = dl.cached_range(sym, TF, 0, 4_102_444_800_000)
    report = validate_ohlcv(df, sym, TF)
    start = pd.Timestamp(int(df["timestamp"].iloc[0]), unit="ms", tz="UTC") if len(df) else None
    end = pd.Timestamp(int(df["timestamp"].iloc[-1]), unit="ms", tz="UTC") if len(df) else None
    print("=" * 72)
    print("BTC 1m VALIDATION")
    print(f"  bar count : {len(df):,}")
    print(f"  date range : {start} -> {end}")
    print(f"  gap runs   : {len(report.gaps)}  (missing bars={report.missing_bar_count()})")
    print(f"  duplicates : {len(report.duplicates)}")
    print(report.summary())
    print("=" * 72)


def main() -> int:
    est = estimate()
    print("=== 1m DOWNLOAD ESTIMATE (from 15m density) ===")
    print(f"  total 15m bars (30 symbols) : {est['total_15m_bars']:,}")
    print(f"  est 1m bars                 : {est['est_1m_bars']:,}")
    print(f"  est API pages (@{PAGE_LIMIT})     : {est['est_pages']:,.0f}")
    print(f"  est wall clock              : {est['est_minutes']:.0f} min (~{est['est_minutes']/60:.1f} h)")
    print(f"  rough parquet size          : ~{est['est_gb_rough']:.1f} GB")
    print("Starting download now...")
    print()

    t0 = time.perf_counter()
    dl = OHLCVDownloader(exchange_id="binanceusdm", cache_dir=CACHE, limit=PAGE_LIMIT)

    # BTC first for early validation
    entries = load_universe(ROOT / "specs" / "universe.json")
    btc = next(e for e in entries if e.symbol.startswith("BTC/"))
    since_ms = int(pd.Timestamp(btc.since, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    print(f"[BTC first] downloading {btc.symbol} 1m ...")
    r = dl.download(btc.symbol, TF, since_ms, until_ms)
    print(f"  BTC done: rows={r.rows} new={r.fetched_new_bars} {r.start} -> {r.end}")
    validate_btc(dl)

    print("\nDownloading remaining universe symbols...")
    pull_and_validate_universe(
        universe_path=ROOT / "specs" / "universe.json",
        timeframe=TF,
        exchange_id="binanceusdm",
        cache_dir=CACHE,
        reports_dir=ROOT / "data" / "reports",
    )

    elapsed = time.perf_counter() - t0
    # storage under 1m trees
    size_1m = 0.0
    for p in (CACHE / "binanceusdm").rglob("1m"):
        if p.is_dir():
            size_1m += _dir_size_gb(p)
    # more reliable: sum all **/1m/**/*.parquet
    total_bytes = 0
    for p in (CACHE / "binanceusdm").glob("*/*/1m/*.parquet"):
        total_bytes += p.stat().st_size
    # also nested path exchange/symbol/1m/
    for p in (CACHE / "binanceusdm").rglob("*.parquet"):
        if "/1m/" in str(p).replace("\\", "/") or p.parent.name == "1m":
            # avoid double count — use set
            pass
    paths = list((CACHE / "binanceusdm").rglob("*.parquet"))
    paths = [p for p in paths if p.parent.name == "1m"]
    total_bytes = sum(p.stat().st_size for p in paths)

    print()
    print("=== 1m DOWNLOAD COMPLETE ===")
    print(f"  actual wall clock : {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"  estimate was      : {est['est_minutes']:.0f} min")
    print(f"  storage (1m parquet): {total_bytes/(1024**3):.2f} GB  ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
