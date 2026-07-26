# Holy Trinity v1.0 — Training Result

**The v1.0 session-range-gated ICT/Wyckoff model, tested across 4,483 trades on 21 crypto perpetual pairs, shows no directional signal at any target horizon from 0.5R to 3R. All three pre-registered tag hypotheses failed to clear the 5pp bar. The holdout set was not opened.**

## Pooled Training Reach vs. Random Walk

Binance USDT-M perpetuals, 15m, training split only (21 symbols, dates before 2025-03-11), n = 4,483 trades, ~68 trades/month.

| target | n | reach% | 1/(1+T)% | diff (pp) |
|---:|---:|---:|---:|---:|
| 0.5R | 4,483 | 65.1% | 66.7% | −1.6 |
| 1.0R | 4,483 | 50.2% | 50.0% | +0.2 |
| 1.5R | 4,483 | 39.9% | 40.0% | −0.1 |
| 2.0R | 4,483 | 32.8% | 33.3% | −0.5 |
| 2.5R | 4,483 | 27.9% | 28.6% | −0.7 |
| 3.0R | 4,483 | 23.6% | 25.0% | −1.4 |

Every horizon is within 1.6 percentage points of the random-walk baseline. Pooled and unconditioned, this is a coin.

## Pre-Registered Tag Hypotheses (Training Only)

Promotion bar: ON-side reach rate beats 1/(1+T) by **>5pp** at **n_on ≥ 200**, on training data only.

| tag | target | n_on | ON% | n_off | OFF% | RW% | ON edge (pp) | Δreach ON−OFF (pp) | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| stacked_imbalance | 0.5R | 1,758 | 64.5% | 2,725 | 65.5% | 66.7% | −2.2 | −1.0 | no |
| stacked_imbalance | 1.0R | 1,758 | 50.6% | 2,725 | 49.9% | 50.0% | +0.6 | +0.6 | no |
| stacked_imbalance | 1.5R | 1,758 | 41.2% | 2,725 | 39.0% | 40.0% | +1.2 | +2.3 | no |
| stacked_imbalance | 2.0R | 1,758 | 35.0% | 2,725 | 31.4% | 33.3% | +1.6 | +3.5 | no |
| stacked_imbalance | 2.5R | 1,758 | 30.7% | 2,725 | 26.1% | 28.6% | +2.1 | +4.5 | no |
| stacked_imbalance | 3.0R | 1,758 | 26.0% | 2,725 | 22.0% | 25.0% | +1.0 | +4.0 | no |
| big_displacement | 0.5R | 2,180 | 64.3% | 2,135 | 65.6% | 66.7% | −2.4 | −1.3 | no |
| big_displacement | 1.0R | 2,180 | 50.0% | 2,135 | 50.3% | 50.0% | +0.0 | −0.2 | no |
| big_displacement | 1.5R | 2,180 | 40.5% | 2,135 | 39.2% | 40.0% | +0.5 | +1.3 | no |
| big_displacement | 2.0R | 2,180 | 33.4% | 2,135 | 32.3% | 33.3% | +0.1 | +1.1 | no |
| big_displacement | 2.5R | 2,180 | 28.0% | 2,135 | 27.9% | 28.6% | −0.6 | +0.1 | no |
| big_displacement | 3.0R | 2,180 | 23.0% | 2,135 | 24.1% | 25.0% | −2.0 | −1.1 | no |
| wide_range | 0.5R | 2,069 | 65.3% | 2,196 | 64.3% | 66.7% | −1.3 | +1.0 | no |
| wide_range | 1.0R | 2,069 | 50.3% | 2,196 | 49.7% | 50.0% | +0.3 | +0.5 | no |
| wide_range | 1.5R | 2,069 | 39.9% | 2,196 | 39.6% | 40.0% | −0.1 | +0.4 | no |
| wide_range | 2.0R | 2,069 | 32.8% | 2,196 | 32.7% | 33.3% | −0.5 | +0.2 | no |
| wide_range | 2.5R | 2,069 | 27.6% | 2,196 | 28.1% | 28.6% | −1.0 | −0.5 | no |
| wide_range | 3.0R | 2,069 | 22.8% | 2,196 | 24.1% | 25.0% | −2.2 | −1.4 | no |

Stacked imbalance comes closest — its ON-vs-OFF gap widens toward the longer horizons (+4.5pp at 2.5R) with a comfortable sample size (n_on = 1,758) — but its ON-vs-random-walk edge tops out at +2.1pp, well short of the +5pp bar. This is not a sample-size problem; the effect simply isn't there at the pre-declared threshold.

## Method (summary)

- **Universe:** 30 Binance USDT-M perpetuals, 15m, maximum available history (earliest: BTC 2019-09-08; latest data: 2026-07-26). One symbol (MKR) truncated at 2025-09-08, when it went inactive.
- **Split:** 9/30 symbols (30%) held out entirely (random, seed 20260726); the most recent 20% of the overall calendar span (cutoff 2025-03-11) held out for every symbol. Training set = 21 symbols, dates before cutoff only.
- **Gate:** session-range liquidity raid → market-structure shift through the protected internal swing → fair-value-gap retest, in that order — the only three gates. Entry at the FVG edge, stop beyond the sweep extreme, target at the far session-range edge (Pine v1.0 defaults).
- **Reach tracker:** each entry followed independently for 100 bars, asking whether 0.5R–3R was reached before a 1R adverse move — regardless of the trade's own live target. Same-bar stop/target ambiguity resolved conservatively (stop assumed first; such bars flagged, not dropped).
- **Holdout:** not opened. No tag met the pre-declared promotion bar on training data, so per protocol there is nothing to confirm out of sample.

Full trade-level data: `data/reports/pooled_trades_all.parquet`. Table sources: `data/reports/training_reach_table.csv`, `data/reports/training_tag_hypotheses.csv`.
