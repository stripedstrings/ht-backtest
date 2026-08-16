"""Fetch Binance BTC funding around 2024-03-01 for manual cross-check."""
from __future__ import annotations

import ccxt
import pandas as pd

ex = ccxt.binanceusdm({"enableRateLimit": True})
since = int(pd.Timestamp("2024-02-29T00:00:00Z").timestamp() * 1000)
rows = ex.fetch_funding_rate_history("BTC/USDT:USDT", since=since, limit=20)
print("Binance live API (BTC/USDT:USDT):")
for r in rows:
    ts = pd.Timestamp(r["timestamp"], unit="ms", tz="UTC")
    rate = float(r["fundingRate"])
    print(f"  {ts.isoformat()}  decimal={rate:.8f}  percent={rate * 100:.4f}%")
