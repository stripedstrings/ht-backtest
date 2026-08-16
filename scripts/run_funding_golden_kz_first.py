"""Golden: kz_first_raid_reclaim train n must stay 20,382 after funding merge.

The merge must not drop or duplicate OHLCV bars. Full train universe.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ht_backtest.data.downloader import OHLCVDownloader  # noqa: E402
from ht_backtest.data.funding import attach_funding_rate  # noqa: E402
from ht_backtest.data.split import SplitManifest  # noqa: E402
from ht_backtest.reports.universe_report import generate_pooled_trades  # noqa: E402
from ht_backtest.strategies.registry import get_strategy  # noqa: E402

EXPECTED_TRAIN_N = 20_382
BTC = "BTC/USDT:USDT"


def main() -> int:
    split = SplitManifest.load(ROOT / "specs" / "splits" / "v1.json")
    dl = OHLCVDownloader(cache_dir=str(ROOT / "data" / "raw"))
    bars = dl.cached_range(BTC, "15m", 0, 4_102_444_800_000)
    n0 = len(bars)
    merged = attach_funding_rate(bars, BTC, funding_dir=ROOT / "data" / "funding")
    if len(merged) != n0:
        print(f"FAIL BTC bar count drift: {n0} → {len(merged)}")
        return 1
    print(f"OK BTC bars unchanged: {n0}")
    if merged["funding_rate"].notna().sum() == 0:
        print("FAIL: no funding_rate values on BTC after merge (download first?)")
        return 1
    print(f"OK BTC funding coverage: {merged['funding_rate'].notna().mean():.1%} of bars")

    strategy = get_strategy("kz_first_raid_reclaim")
    trades, timings = generate_pooled_trades(
        strategy,
        split,
        timeframe="15m",
        exchange_id="binanceusdm",
        cache_dir=str(ROOT / "data" / "raw"),
        mfe_win=100,
        workers=4,
        strategy_name="kz_first_raid_reclaim",
        split_path=str(ROOT / "specs" / "splits" / "v1.json"),
        funding_dir=str(ROOT / "data" / "funding"),
        attach_funding=True,
    )
    train = trades[trades["split"] == "train"] if not trades.empty else trades
    n = len(train)
    print(f"train n={n} (expected {EXPECTED_TRAIN_N})")
    print(f"wall≈{timings.total_s:.1f}s")
    if n != EXPECTED_TRAIN_N:
        print("FAIL: trade count drifted after funding merge — fix before step 1")
        return 1
    print("PASS: funding merge did not change kz_first_raid_reclaim train n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
