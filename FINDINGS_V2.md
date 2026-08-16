# FINDINGS_V2 — Independent Strategy Screen & Condition Grid

**Date:** 2026-08-16  
**Code:** [github.com/stripedstrings/ht-backtest](https://github.com/stripedstrings/ht-backtest)

## Executive summary

We tested the Holy Trinity session-range model and ten related “smart money” style strategies on years of Binance crypto perpetual data, then tried every sensible pair of market filters on top of the simplest raid-and-reclaim setup. Almost everything landed at or below a fair-coin benchmark. The one idea we pre-registered and checked on sealed holdout data also failed its bar. After correcting for looking at hundreds of combinations at once, **zero** filter pairs survived. In plain terms: on this dataset, these rules look like a coin flip—not a trading edge.

## Methodology overview

### Data

- **Venue / product:** Binance USDT-M perpetual futures  
- **Timeframe:** 15-minute bars (plus 4h bars and funding rates for the filter grid)  
- **Universe:** 30 liquid crypto pairs (`specs/splits/v1.json`)  
- **History:** roughly 2019-09-08 → 2026-07-26 (symbol list dates vary)

### Split (sealed holdout)

- **21 train / 9 holdout symbols** (30% symbol holdout, seed 20260726)  
- **Date holdout:** last 20% of the calendar (from 2025-03-11) on every symbol  
- Discovery and grid search used **training data only**. Holdout was opened only for one pre-registered SMT test.

### Metrics

- **Headline:** reach rate vs random-walk baseline `1/(1+T)` at profit multiples T = 0.5R … 3R  
- At **1R**, a fair coin that hits +1R or −1R with equal chance reaches the target **50%** of the time  
- **Promotion bar** for a standalone strategy (train): edge **>5 percentage points** above that baseline at some horizon, with **n ≥ 200**

### Multiple-testing correction (condition grid)

- Base entry: first killzone raid + reclaim (`kz_first_raid_reclaim`)  
- Library: 22 conditions (Asia session excluded from combo generation); depth-2 AND pairs only  
- Primary test per combo: one-sided binomial at 1R vs 50%  
- **Benjamini–Hochberg** and **Benjamini–Yekutieli** FDR at q = 0.05  
- Survivor rule: BH-significant **and** train edge **>+3pp** at 1R  
- Synthetic injection test (70% reach on 500 trades) passed before the real grid; baseline returned empty after removal

## Results table — training reach vs RW at 1R

Eleven strategies: Holy Trinity v1.0 plus the ten independent hypotheses from the survivor batch. Holdout column filled only where a pre-registered test was run.

| Strategy | n (train) | Reach 1R | RW 1R | Diff (pp) | Holdout | Promoted |
|----------|----------:|---------:|------:|----------:|---------|:--------:|
| holy_trinity_v10 | 4,483 | 50.2% | 50.0% | +0.2 | not opened | no |
| kz_first_raid_reclaim | 20,382 | 47.7% | 50.0% | −2.3 | not opened | no |
| kz_first_30m_raid_reclaim | 5,170 | 48.3% | 50.0% | −1.7 | not opened | no |
| high_vol_grab_reclaim | 20,321 | 47.8% | 50.0% | −2.2 | not opened | no |
| low_vol_grab_reclaim | 1,614 | 47.5% | 50.0% | −2.5 | not opened | no |
| tight_asia_spring | 3,261 | 48.8% | 50.0% | −1.2 | not opened | no |
| asia_mid_bias_raid | 13,581 | 47.7% | 50.0% | −2.3 | not opened | no |
| london_ny_same_direction | 2,215 | 49.2% | 50.0% | −0.8 | not opened | no |
| failed_raid_next_session_fade | 896 | 47.9% | 50.0% | −2.1 | not opened | no |
| smt_fade_sweeper_btc_eth | 4,462 | 46.7% | 50.0% | −3.3 | not opened | no |
| smt_trade_holder_btc_sol | 3,763 | 49.8% | 50.0% | −0.2 | **pre-reg 6h exit:** mean fwd R **+0.087** (n=1,147); bar was +0.10R → **FAIL** | no |

Notes:

- Several strategies show small **positive** edges at *other* horizons on train (e.g. low-vol ≈ +1.8pp at 3R; SMT holder ≈ +1.5pp at 2R). None cleared the **>5pp / n≥200** promotion bar.  
- HT tags from FINDINGS.md (stacked imbalance, big displacement, wide range) also failed the 5pp bar; holdout stayed sealed for those.  
- Sources: `FINDINGS.md`, `data/runs/batch_v1_20260816T130731Z/`, `data/memory/hypothesis_log.csv`, `specs/pre_registered/smt_6h_exit.md`.

## Grid search summary

| Item | Value |
|------|------:|
| Base entry | `kz_first_raid_reclaim` (train n = 20,382) |
| Conditions in grid | 22 (`asia_session` kept in library, excluded from combos) |
| Combos generated (depth 2) | 231 |
| Mutex-skipped | 12 |
| Killed (n &lt; 200) | 13 |
| **Tested** | **206** |
| Diff at 1R — mean | −1.95 pp |
| Diff at 1R — median | −1.98 pp |
| Diff at 1R — min / max | −8.20 / +5.27 pp |
| Diff at 1R — std | 1.76 pp |
| BH survivors (+3pp rule) | **0** |
| BY survivors (+3pp rule) | **0** |

The single best raw pair was `4h_hh_hl ∧ london_open_30m` (+5.27pp, n=275, uncorrected p≈0.046). It did **not** survive FDR across 206 tests. Most pairs sat below the coin baseline. Funding filters did not produce a corrected edge. Depth-3 was not run.

Artifacts: `data/grid/condition_grid.sqlite` (`run_id=depth2_real`).

## Conclusion

1. **Holy Trinity v1.0** is statistically a coin on train (FINDINGS.md).  
2. **Ten related independent strategies** also fail the pre-declared train promotion bar at every horizon that matters for promotion.  
3. The **only** pre-registered holdout check (SMT holder, 6h time exit) **failed** its +0.10R mean bar.  
4. A **systematic depth-2 condition grid** on the simplest reclaim base, after BH/BY correction, produced **zero** survivors.  
5. Taken together: on this universe and protocol, we do **not** have evidence of a tradable edge from these ICT/Wyckoff-inspired rules or from pairwise filters on top of them.

## What this means for retail traders

If you have been told that “killzone raids,” “judas swings,” “SMT divergence,” or “Asia range springs” reliably beat the market on crypto perpetuals, **this study did not find that**.

We measured something simple and honest: after a setup fires, how often does price reach a clean profit target before an equal-sized stop—compared with a fair coin. Across thousands of trades and many popular storylines, the answer was almost always “about the same as a coin,” and sometimes worse. When we tried stacking filters (volume, funding, trend, session timing) and then accounted for the fact that testing hundreds of ideas creates false winners, **nothing remained**.

That does **not** prove nobody can make money, and it is **not** financial advice. Markets change; other assets, timeframes, or costs might look different. It **does** mean you should treat confident ICT/SMC marketing with skepticism until you see a test like this—large sample, train/holdout discipline, and a correction for fishing through many variations—come out clearly positive.

Past performance (including these flat results) does not predict the future. If you trade, size small, assume costs and slippage, and do not confuse a good story with a measured edge.

---

*Related: `FINDINGS.md` (Holy Trinity pooled train result), `specs/architecture/condition_grid_search.md`, `data/memory/hypothesis_log.csv`.*
