import pandas as pd
import pytest

from ht_backtest.data.downloader import OHLCVDownloader, TIMEFRAME_MS

TF = "15m"
TF_MS = TIMEFRAME_MS[TF]


class _FakeExchange:
    """Serves bars from a fixed synthetic timeline; fetch_ohlcv pages through
    it like a real exchange would, so we exercise the downloader's own
    pagination/resume logic without hitting the network."""

    def __init__(self, first_ts: int, last_ts: int):
        self.first_ts = first_ts
        self.last_ts = last_ts
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append(since)
        start = max(since, self.first_ts)
        if start > self.last_ts:
            return []
        out = []
        ts = start
        while ts <= self.last_ts and len(out) < limit:
            out.append([ts, 1.0, 1.0, 1.0, 1.0, 1.0])
            ts += TF_MS
        return out


@pytest.fixture
def dl(tmp_path, monkeypatch):
    d = OHLCVDownloader(exchange_id="binanceusdm", cache_dir=tmp_path)
    return d


def test_download_fills_gap_before_a_previously_cached_island(dl):
    # Simulate exactly the bug scenario: an earlier run only cached a recent
    # "island" window, then a later run asks for much deeper history. The
    # downloader must fetch the earlier missing segment, not silently skip it
    # because it only looked at the island's own max timestamp.
    day0 = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    island_start = int(pd.Timestamp("2026-05-01", tz="UTC").timestamp() * 1000)
    island_end = int(pd.Timestamp("2026-05-02", tz="UTC").timestamp() * 1000)
    now = int(pd.Timestamp("2026-05-02", tz="UTC").timestamp() * 1000) + TF_MS

    fake = _FakeExchange(first_ts=day0, last_ts=now)
    dl.exchange = fake

    # first, only pull the island (mirrors the earlier sanity-window run)
    dl.download("BTC/USDT:USDT", TF, island_start, island_end)
    island_only = dl.cached_range("BTC/USDT:USDT", TF, day0, now)
    assert island_only["timestamp"].min() == island_start

    # now request full history back to day0 -- the historical gap must be filled
    dl.download("BTC/USDT:USDT", TF, day0, now)
    full = dl.cached_range("BTC/USDT:USDT", TF, day0, now)
    assert full["timestamp"].min() == day0
    expected_bars = (now - day0) // TF_MS
    assert len(full) == expected_bars


def test_download_is_idempotent_and_makes_no_calls_when_fully_cached(dl):
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2026-01-02", tz="UTC").timestamp() * 1000)
    fake = _FakeExchange(first_ts=start, last_ts=end)
    dl.exchange = fake

    dl.download("BTC/USDT:USDT", TF, start, end)
    calls_after_first = len(fake.calls)
    assert calls_after_first > 0

    dl.download("BTC/USDT:USDT", TF, start, end)
    assert len(fake.calls) == calls_after_first  # no new fetches needed


def test_download_fills_internal_gap_between_two_cached_blocks(dl):
    day0 = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    block_a_end = day0 + 10 * TF_MS
    block_b_start = day0 + 20 * TF_MS
    block_b_end = day0 + 30 * TF_MS

    fake = _FakeExchange(first_ts=day0, last_ts=block_b_end)
    dl.exchange = fake
    dl.download("BTC/USDT:USDT", TF, day0, block_a_end)
    dl.download("BTC/USDT:USDT", TF, block_b_start, block_b_end)

    partial = dl.cached_range("BTC/USDT:USDT", TF, day0, block_b_end)
    assert len(partial) < (block_b_end - day0) // TF_MS

    dl.download("BTC/USDT:USDT", TF, day0, block_b_end)
    full = dl.cached_range("BTC/USDT:USDT", TF, day0, block_b_end)
    assert len(full) == (block_b_end - day0) // TF_MS
