# Autonomous research and execution system

**Status:** architecture (no implementation in this document).  
**Date:** 2026-08-22  
**Constraint:** [`promotion_gate.md`](promotion_gate.md) is mandatory. This system cannot lower the reach-vs-RW bar, skip pre-registered holdout, or treat paper as statistical confirmation.

This document specifies everything **downstream of the existing research engine**: data daemon, hypothesis generator, strategy synthesizer, execution (separate process), and monitoring. It does not replace [`condition_grid_search.md`](condition_grid_search.md) or the working backtest.

**Trust boundary:** `ht-backtest` is research only (public market data). Paper and live trading run in a **separate process** (recommended: separate repo). API keys never live in this repository.

```
Untrusted:  URL text, LLM prose, Mode B Python (forbidden unsupervised)
Trusted:    YAML compiler (after human expansion review), dry-count, train worker,
            grid FDR, promotion gate, golden HT test
Execution:  separate process; keys local to that process; loads passports only
```

---

## What already exists (research engine)

| Piece | Location | Notes |
|-------|----------|--------|
| Reach-vs-RW | `src/ht_backtest/reports/reach.py` | Headline `1/(1+T)` |
| Train comparison | `src/ht_backtest/reports/comparison.py` | Defaults 5pp / n≥200 — **must become the gate**, not a default |
| Strategy protocol | `src/ht_backtest/strategies/base.py` | Strategies emit trades; never score or split |
| HT v1.0 + 10 hypotheses | `holy_trinity_v10.py`, `hypotheses.py` | All failed train promotion; SMT holdout failed |
| Locked split v1 | `specs/splits/v1.json` | 21/9 symbols, date holdout from 2025-03-11; TF-agnostic |
| OHLCV / funding / 4h | `data/downloader.py`, `funding.py`, `htf_4h.py` | Causal as-of pattern |
| Condition library (23) | `src/ht_backtest/conditions/` | Grid excludes `asia_session` |
| Depth-2 FDR grid | `src/ht_backtest/grid/engine.py` | 206 tested; **0** BH/BY survivors (`FINDINGS_V2.md`) |
| Discovery MVP | `src/ht_backtest/discovery/` | Intake → Claude/fixture YAML → dry-count → worker **train only** |
| YAML compiler | `discovery/compile.py` | Allowlist of existing classes only (`_METHOD_MAP`) |
| Hypothesis memory | `data/memory/hypothesis_log.csv` | Query exists; **does not auto-reject**; no `condition_signature` yet |
| Pre-reg template | `specs/pre_registered/smt_6h_exit.md` | Only holdout opened; failed |
| Execution / paper / live / daemon | — | **None** |

[`auto_discovery_system.md`](auto_discovery_system.md) still describes Mode B, combination engine, and EA generator as design. **This document supersedes Components 3–4 for execution and combination.** Intake Mode A + worker (that doc’s Phases 1–2) are already in code.

---

## System overview

```
┌─────────────┐     public data      ┌──────────────────┐
│ Data daemon │ ───────────────────► │ Caches + library │
└─────────────┘                      └────────┬─────────┘
                                              │
┌─────────────┐   Mode A YAML        ┌────────▼─────────┐
│ Hypothesis  │ ───────────────────► │ Compiler         │
│ generator   │   (budget + memory)  │ (FDR inherit)    │
└──────▲──────┘                      └────────┬─────────┘
       │                                      │ queued
       │ memory                               ▼
       │                             ┌──────────────────┐
       └─────────────────────────────│ Train worker     │  train only
                                     └────────┬─────────┘
                                              │
                                     ┌────────▼─────────┐
                                     │ Promotion gate   │
                                     └────────┬─────────┘
                    human pre-reg commit      │
                    human compiler review     │
                                              ▼ holdout_cleared passport
                                     ┌──────────────────┐
                                     │ Synthesizer      │  artifact only
                                     └────────┬─────────┘
                                              │
                         ═ trust boundary ════╪════════════════
                                              ▼
                                     ┌──────────────────┐
                                     │ ht-execution     │  keys here only
                                     │ paper → live     │
                                     │ 30d audit/kill   │
                                     └────────┬─────────┘
                                              │ fill parquets / live_decay
                                              ▼
                                     hypothesis memory
```

---

## Phase 1 — Data enrichment

**Goal:** new causal series and condition atoms feeding the **existing** grid. No new research architecture.

### Exists

OHLCV (incl. 1m), funding, 4h HTF, `enrich_condition_features`, 23 atoms, amortized FDR grid, frozen split v1. Downloads are one-shot CLI. **No** open interest, liquidations, OFI, or vol-regime **conditions** (ER/ATR exist as primitives only).

### Build

1. **Data daemon** (`ht-backtest daemon --loop`): incremental public-data refresh of 15m/1m OHLCV, funding, 4h, plus new stores. Append/dedup-by-timestamp only. Coverage logs. No trading keys.
2. **Open interest:** `data/oi/{symbol}.parquet`; `merge_asof` backward at bar open (funding clone). Atoms: `oi_rising`, `oi_falling`, `oi_extreme` (trailing percentile, prior windows only).
3. **Liquidations:** public force-order stream, aggregated to **completed** 15m bins. Events inside the current bar are unavailable at that bar’s close decision. Atoms: `recent_long_liq`, `recent_short_liq`. Same rule in paper/live.
4. **Tick OFI:** aggTrades (or 1m if tick storage is impractical) → signed imbalance over a **completed** window, attached to the **next** 15m bar. Store downsampled series in `data/ofi/`, not raw ticks. Atoms: `ofi_buy_pressure`, `ofi_sell_pressure`.
5. **Vol regime:** wire existing `efficiency_ratio` / ATR percentiles: `er_trending`, `er_choppy`, `atr_high`, `atr_low`. No new exchange API.
6. Extend `enrich_condition_features`, `MUTEX_PAIRS`, `library_version()`, coverage + download/validate CLIs. **Re-run full depth-2 FDR** on the new library; do not cherry-pick atoms. Append one memory row for the grid `library_version` hash.
7. Golden: HT reach table and OHLCV row counts unchanged after enrichment (funding-merge invariant).

### Cursor build time

**4–6 agent-days.** Long pole: OFI causal window + storage. Vol-regime &lt;1 day. OI ~1 day. Liquidations ~1–1.5 days. Daemon ~0.5–1 day.

### Most dangerous failure mode

**Lookahead in OFI/liquidation timestamps** — using prints that occur during bar T to filter an entry decided at T’s close (or open). That class of bug historically turned a fake edge into a null when fixed.

**Prevention:** copy funding’s contract (`merge_asof` backward at a documented timestamp); checkpoint unit tests; `None` when missing; grid None-safe AND. **Human checkpoint 1:** causal test review before a grid run on the new `library_version` counts as evidence.

**Autonomy:** daemon download is fully automatable. Grid scoring is automatable. Promotion is not.

---

## Phase 2 — Autonomous research loop

**Goal:** no human between idea generation and **train** scoring. Human remains at pre-reg, and at **compiler expansion review** before the loop is enabled after any compiler change.

### Exists

Intake → Claude YAML → risk scan → dry-count n≥200 → filesystem queue → worker trains on train only → memory → HTML report. Compiler maps only existing strategy classes. Memory does not auto-reject. No rolling split. Combination engine and Mode B are unbuilt.

Without compiler expansion, a generator can only rediscover the exhausted allowlist — the loop would be theater.

### Build

1. **Hypothesis generator** (Claude, Mode A YAML only): packs `format_prior_results_summary`, unused categories, latest grid `library_version`, condition coverage. Writes intake YAML. Hard weekly **candidate** budget. Memory is a **gate**: auto-kill exhausted `signal_type`; refuse `(base_entry_id, condition_signature)` that matches `live_decay` ([`promotion_gate.md`](promotion_gate.md)). Never Mode B. Never HT+tag as a “new” strategy.

2. **Compiler expansion:** YAML = allowlisted **base geometry** (session-range / SMT helpers in `hypothesis_helpers.py`) **plus** AND of library conditions. Canonical `strategy_id` from base + `condition_signature`.
   - **FDR inheritance (compiler-enforced):** if `frozenset(condition_ids)` on that base matches an existing `grid_results` row, **inherit** the result; **consume no new FDR slot** and no weekly new-set slot. New condition sets consume one slot from `WEEKLY_NEW_CONDITION_SET_SLOTS` and must enter the grid family before their p-value counts. See promotion gate.
   - Independent new **bases** (no extra filters) use the 5pp/200 train bar and the generator candidate budget, not the FDR slot budget.

3. **Automated queue:** daemon freshness → compile (inheritance/slots) → `worker --loop`. Schedule depth-2 grid when `library_version()` changes. Train only.

4. **Rolling split refresh:**
   - v1 remains the discovery split for all hashes already in memory.
   - Daemon extends caches; bars after v1 `overall_end` are `unscored_future` (not train, not peekable holdout).
   - New split versions (`v2`, …) only for **new** hashes. Binding `(strategy_id, parameter_hash) → split_id` forever. Retrain-on-unsealed-holdout is forbidden.
   - Phase 4 live audit is **not** a research holdout reopen.

5. **Pre-reg assist:** Claude drafts `specs/pre_registered/*.md`. **Does not commit.** Holdout scorer requires the file at a git SHA.

6. **Watchlist:** train edges in (2pp, 5pp) log only. Combinations still need 5pp/200. No bar cut. Filter-style watchlist items go through the grid/inheritance path, not a parallel unpaid family.

### Cursor build time

**5–8 agent-days.** Hard parts: compiler expansion, FDR inheritance, split binding.

### Most dangerous failure mode

**Holdout leakage** via rolling split or a generator “just looking,” **or** a compiler that emits look-ahead / skips FDR inheritance so YAML combos get a second unpaid test.

**Prevention:** promotion gate; worker/generator have no holdout path; inheritance in the compiler; Mode A only. **Human checkpoint 2:** git commit of pre-reg (never auto). **Human checkpoint 5:** compiler expansion review before enabling the autonomous loop after any compiler change (code review, not data review). A compiler bug is undetectable downstream because train tables will use the same buggy path.

### Autonomy

| Component | Full autonomy? |
|-----------|----------------|
| Generator Mode A + memory kill + train worker | Yes, with budgets |
| Mode B LLM Python | **No** |
| New filter combos | Compiler inherit-or-slot; no unsupervised second family |
| Unsupervised LLM combination search | **No** — FDR grid or weekly cap |
| Pre-reg commit | **Never** |
| Holdout score after commit | Yes (fixed metric) |
| Enable loop after compiler change | **Never** until checkpoint 5 |

---

## Phase 3 — Strategy synthesizer and paper trading

**Goal:** only `holdout_cleared` passports become execution logic. Paper on **Binance testnet** in the execution process. 30-day **operational** validation. Correlation check before live enable.

### Exists

EA generator: design only. ccxt here is market data only. No paper, testnet, or fill engine.

### Build

1. **Synthesizer (research repo):** from `holdout_cleared`, emit passport + execution spec (entry/stop/target/time-exit identical to backtest; 15m close; same-bar stop-first). `artifacts/passports/`. **Stops at artifact.** No keys.

2. **ht-execution (separate process):** loads passport; rejects state ≠ `holdout_cleared` for paper. Maps spec → ccxt **testnet**. Keys: `BINANCE_TESTNET_*` in execution env only.

3. **Fill-parity contract:** fees, funding, intrabar ambiguity documented. Paper fills that use information the backtest did not are a protocol bug. Stop-first on same-bar.

4. **Paper pass bar (frozen, operational, sign only)** — [`promotion_gate.md`](promotion_gate.md) C5–C8:
   - Mean forward R at **6 hours** (24 × 15m bars) **≥ 0**
   - **n ≥ 50** paper trades with a complete 6h path
   - **≥ 30 calendar days**
   - If n &lt; 50 at day 30: **extend the window**, do not lower n
   - **Paper passed ≠ edge confirmed.** Holdout remains the statistical gate. Paper answers “does the stack trade without disaster and is 6h mean R non-negative?” not “is there an edge?”

5. **Portfolio correlation:** before `paper_cleared`, pairwise |ρ| of overlapping returns vs already-paper/live strategies. If `|ρ| > 0.7`, deploy at most the better holdout metric. Human may override **down** (deploy none), never **up** without a written execution-config exception.

### Cursor build time

**6–10 agent-days** (new process, testnet, parity tests).

### Most dangerous failure mode

Deploying from `train_promoted`, **or** paper fills that look ahead of the research model, **or** an operator treating paper pass as confirmation.

**Prevention:** execution refuses non-passport / wrong state; synthesizer requires `holdout_cleared`; fill-parity tests; paper bar is sign-only and documented as operational. **Human checkpoint 3:** explicit paper enable for a named passport in the execution process.

**Autonomy:** spec codegen after holdout clear may be automatic. Paper enable is human. Keys never in `ht-backtest`.

---

## Phase 4 — Live execution and monitoring

**Goal:** live Binance USDM via ccxt in the execution process; hard risk limits; rolling 30-day reach-vs-RW; auto-suspend on decay; feedback into memory. No live keys in the research repo.

### Exists

Nothing.

### Build

1. **Live adapter:** `BINANCE_USDM_*` in execution env. Same fill model as paper. Requires `paper_cleared` **and** human `live_enabled`.

2. **Hard risk limits** (execution, **before** `create_order`):
   - Max position notional **2% of equity** per strategy
   - Portfolio cap: sum of live notionals ≤ **6% of equity** (three concurrent 2% names; freeze at impl if a different cap is chosen — still a protocol constant, not a flag)
   - **Daily loss 5% of equity** (UTC) → flatten + halt new entries for the day
   - **Kill switch:** file or `HT_KILL=1` every loop; flatten and cancel. Research cannot disable this.

3. **Rolling 30-day audit:** `reach_vs_random_walk` on live fills (same definition as research). Research consumes **exported fill parquets** (no keys) to append memory.

4. **Auto-suspend on decay:** if rolling live 1R reach ≤ `1/(1+1)` with n ≥ `LIVE_DECAY_MIN_N` (50), set `live_suspended`, flatten, write `holdout_result=live_decay`. If n &lt; 50, state stays `observe` — do not declare decay **or** a new edge. Generator refuses matching `(base_entry_id, condition_signature)`.

5. **Feedback:** decayed signatures are exhausted. Decay is not an excuse to peek unused v1 holdout or refit parameters on live data. Auto-resume is forbidden.

### Cursor build time

**4–6 agent-days.**

### Most dangerous failure mode

**Position or kill-switch bug** (oversized order, ignored kill, auto-resume), **or** using live PnL to retune rules.

**Prevention:** limits in the order path with tests; kill switch independent of strategy logic; auto-suspend may flatten; **auto-resume never**. **Human checkpoint 4:** first live enable per passport after paper 30d; human-only resume after suspend or kill.

### Autonomy

| Component | Full autonomy? |
|-----------|----------------|
| First live enable / keys | **Never** |
| Daily loss halt / kill / decay suspend | **Yes** (safer on) |
| Resume after suspend | **Never** — human |
| Raising 2%/5% limits | **Never** automatic |
| Memory append from fill exports | Yes |

---

## Human checkpoints (minimum set)

Five gates. Everything else may run unattended. Two of them are **undetectable downstream** if skipped (a later table cannot reveal the cheat).

| # | Checkpoint | When | Review type | Undetectable if skipped? |
|---|------------|------|-------------|--------------------------|
| 1 | Causal review of each new data/condition family | Before that `library_version` grid run is evidence | Data / tests | No — golden and checkpoint tests can still fail |
| 2 | Git commit of `specs/pre_registered/*.md` | Before any holdout score | Protocol | **Yes** — peeking is invisible in the holdout number |
| 3 | Enable paper for a named `holdout_cleared` passport | Execution process | Ops | No — missing enable means no orders |
| 4 | Enable live after paper 30d; human resume after kill/suspend | Execution process | Capital | No — missing enable means no live orders |
| 5 | **Compiler expansion review** | After **any** compiler change, **before** the autonomous loop is enabled | **Code review, not data review** | **Yes** — look-ahead or FDR-bypass compiles into every subsequent train table |

Checkpoint 5 is the only other undetectable-downstream gate besides pre-reg. A compiler that skips inheritance, emits future bars, or maps YAML onto the wrong geometry will produce internally consistent (and false) reach tables. Enablement of `worker --loop` / generator after a compiler diff requires a recorded human review of `discovery/compile.py` and its tests.

Mode B Python and unsupervised combination search remain **out of scope for autonomy** even after Phase 2.

---

## Phase summary

| Phase | Cursor time | Dangerous failure | Prevented by |
|-------|-------------|-------------------|--------------|
| 1 Data | 4–6 days | OFI/liq lookahead | Funding-style as-of + checkpoint 1 |
| 2 Research loop | 5–8 days | Holdout leak / unpaid FDR / compiler look-ahead | Gate + compiler inheritance + checkpoints 2 and 5 |
| 3 Paper | 6–10 days | Train-only deploy / fill mismatch / paper-as-proof | Passport states + sign-only paper bar + checkpoint 3 |
| 4 Live | 4–6 days | Size/kill bug / live refit | Order-path limits + no auto-resume + checkpoint 4 + `live_decay` signature refuse |

**Total (implementation, later):** ~20–30 agent-days. This document is spec only.

---

## Explicit non-goals (this spec)

- API keys, `.env`, or ccxt signed trading in `ht-backtest`
- Automating pre-reg commits, compiler-review sign-off, paper enable, or live enable
- Mode B unsupervised Python
- Lowering C1/C2 for combinations or autonomous discoveries
- Treating `paper_cleared` as statistical confirmation of edge
