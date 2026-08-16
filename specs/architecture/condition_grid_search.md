# Architecture: Condition Library & Grid Search Engine

**Status:** design only — no implementation until review.  
**Date:** 2026-08-16  
**Parent:** extends `specs/architecture/auto_discovery_system.md` Component 3 (combination), but replaces opportunistic LLM pairing with a **systematic, FDR-controlled** grid over atomic conditions.  
**Non-negotiables:** train-only scoring; holdout sealed; dry-count `n ≥ 200`; no HT+filter pseudo-strategies as the *base* (base is already an independent session-range reclaim entry); survivors ≠ “tested.”

---

## Goal

Define a fixed library of **20–30 atomic, bar-local conditions** and a grid engine that tests every **unordered pair and triple** (AND) of those conditions as filters on a **single fixed base entry**: session-range first-raid reclaim (`kz_first_raid_reclaim` geometry).

For each combination:

1. Dry-count on train → skip if `n < 200`.
2. Else compute train reach-vs-RW table (same horizons as today).
3. Persist row keyed by **combination signature**.
4. After the full family of tests is complete, apply **Benjamini–Hochberg FDR** at `q = 0.05` and write **survivors** to a separate store so “tested” vs “significant after correction” cannot be confused.

Human-facing queries must answer e.g. “all volume-tagged combos that cleared BH” or “top 20 by 1R edge with `n > 500`.”

---

## Relationship to what already exists

| Existing piece | Role in this phase |
|----------------|--------------------|
| `run_session_range_engine` + `KzFirstRaidReclaimStrategy` | **Base entry** (fixed; not searched) |
| Primitives: sessions, ATR, ER, daily bias, volume, pools | Condition implementations (most categories) |
| Dry-count + worker train path | Semantics to preserve; **execution shape changes** (amortize, don’t re-FSM 2k times) |
| `hypothesis_log.csv` | Append a **summary row per grid run** + optional per-survivor rows; not the primary combo DB |
| Audit report generator | Later: one report per BH survivor |
| LLM combination engine (auto-discovery §3) | Orthogonal: curated/sparse; this grid is exhaustive/local |

**Explicit non-goal for this phase:** searching over base-entry variants (30m first raid, vol-gated reclaim, SMT, etc.). Those stay separate strategies. The grid only multiplies **conditions × fixed base**.

---

## Component A — Condition library

### Contract

```text
Condition.eval(bar_index, symbol_ctx) -> True | False | None
```

| Return | Meaning |
|--------|---------|
| `True` | Condition holds at this bar |
| `False` | Condition fails at this bar |
| `None` | Undefined / insufficient history / missing aux — **trade excluded** from that combo (same as fail for AND filters, but logged separately for coverage diagnostics) |

**Hard rules (stateless per bar, no lookahead, no repainting):**

1. Output at bar `t` may use only data with timestamp **≤ bar `t` close** (for funding: last **published** rate with `fundingTime ≤ open_time(t)` — never the rate that settles *during* or *after* the bar).
2. No center-window confirmation that flips a past bar’s value later (fractals only in **confirmed** form, already in primitives).
3. No cross-bar mutable “state machine” inside the condition itself. Session/prior-outcome facts must be **features already materialized** on the bar (or looked up from a precomputed causal series).
4. Conditions are **pure functions of precomputed feature columns** + bar index. Grid code never calls `shift(-k)`.

### Categories and proposed atoms (target **L = 26**)

Counts are the design target for estimates; exact IDs freeze at implementation review.

#### 1. Session timing (6)

| ID | True when (causal) |
|----|--------------------|
| `sess_london_kz` | Bar in London KZ |
| `sess_ny_kz` | Bar in NY KZ |
| `sess_first_60m_kz` | Within first 60m of current KZ instance |
| `sess_second_half_kz` | In second half of current KZ window |
| `sess_monday` | London-calendar Monday |
| `sess_not_friday` | Not Friday (London calendar) |

#### 2. Volume (5)

| ID | True when |
|----|-----------|
| `vol_gt_p50_50` | Volume > rolling median(50), prior windows only |
| `vol_gt_p80_50` | Volume > rolling 80th pct(50) |
| `vol_lt_p50_50` | Volume ≤ rolling median(50) |
| `vol_up_vs_prev` | `volume[t] > volume[t-1]` |
| `vol_gt_session_med` | Volume > expanding/rolling median of **same session tag** (causal) |

#### 3. Funding rate — **new data source** (4)

| ID | True when |
|----|-----------|
| `fund_positive` | Last known funding rate > 0 |
| `fund_negative` | Last known funding rate < 0 |
| `fund_hi_p90` | Funding ≥ trailing 90th pct over 30d of funding prints |
| `fund_lo_p10` | Funding ≤ trailing 10th pct over 30d |

**Ingest (new):** Binance USDT-M premiumIndex / fundingRate history per symbol, 8h cadence, stored under e.g. `data/cache/funding/{symbol}.parquet` with columns `fundingTime`, `fundingRate`. Align to 15m bars by as-of merge backward. Symbols without history → all funding conditions `None` until coverage exists; grid may still run non-funding combos.

**Live-parity note:** backtest must use the same as-of rule the eventual EA would see at bar close decision time.

#### 4. Range context (5)

| ID | True when |
|----|-----------|
| `range_wide_atr` | `range_width_atr > 1.0` (or ATR-normalized session range proxy already in stack) |
| `range_tight_atr` | `range_width_atr < 0.5` |
| `er_trending` | Efficiency ratio(40) > 0.4 |
| `er_choppy` | Efficiency ratio(40) < 0.2 |
| `in_premium` / `in_discount` | Daily bias premium/discount (pick one ID each — counts as 2 toward L; table shows the pair) |

*(Library count: treat `in_premium` and `in_discount` as two atoms → range category contributes 6 if both kept; freeze at impl so **L ∈ [24, 28]**.)*

**Frozen count for this doc’s math:** **L = 26** with range atoms = `range_wide_atr`, `range_tight_atr`, `er_trending`, `er_choppy`, `in_premium`, `in_discount` (6) and session 6 + volume 5 + funding 4 + htf 3 + prior 2 = 26.

#### 5. HTF trend (3)

| ID | True when |
|----|-----------|
| `htf_daily_bull` | Daily bias bullish (existing pivots) |
| `htf_daily_bear` | Daily bias bearish |
| `htf_above_h1` | Close above confirmed daily H1 (causal) |

No separate 1h/4h series in v1 — reuse UTC daily bias machinery already in primitives (deliberate: no new timeframe join bugs).

#### 6. Prior session outcome (2)

| ID | True when |
|----|-----------|
| `prior_kz_raid_up` | Previous KZ instance on this symbol had an upside raid |
| `prior_kz_raid_down` | Previous KZ instance had a downside raid |

Materialize as bar features at KZ open from the session-range engine’s prior instance — still causal if tagged on bars *after* that instance completed.

### Metadata each condition must declare

```yaml
id: vol_gt_p50_50
category: volume          # session|volume|funding|range|htf|prior_session
version: 1
lookahead_class: causal_ok
requires: [ohlcv]         # or [funding]
warmup_bars: 50
mutually_exclusive_with: [vol_lt_p50_50]   # optional; grid may skip absurd ANDs
```

**Optional prune (recommended):** skip combinations that AND mutually exclusive pairs (e.g. `vol_gt_p50` ∧ `vol_lt_p50`, `fund_positive` ∧ `fund_negative`, `htf_daily_bull` ∧ `htf_daily_bear`). This reduces m slightly and removes structural zeros — document the exclusion list in the run manifest so FDR family size is exact.

### Evaluation surface

Prefer a **condition matrix** per symbol:

- Shape: `(n_bars, L)` of `{1, 0, -1}` or nullable bool.
- Built once after primitives (+ funding as-of).
- Entry filter for combo `C`: at entry bar `t`, `all(matrix[t, c] is True for c in C)` (None → reject).

---

## Component B — Base entry (fixed)

**Base:** first killzone raid + reclaim within `reclaim_win=3`, same as `kz_first_raid_reclaim` / golden worker path.

Grid does **not** re-search reclaim windows, one_raid flags, or session definitions.

Each base trade carries:

- `entry_ts`, `symbol`, `side`, stop/target geometry  
- Precomputed forward reach flags / MFE path used today for the reach table  
- Pointer to entry bar index for condition lookup  

Empirical scale (completed golden): **n ≈ 20,382** train trades on 21 symbols — headroom for filters before `n < 200`.

---

## Component C — Grid search engine

### Search space

- Library size `L` (default 26).  
- Depth 2: all unordered pairs `{i,j}`.  
- Depth 3: all unordered triples `{i,j,k}`.  
- Operator: **AND** only (v1). No OR, no XOR, no “condition on stop distance.”  
- Signature (canonical key):

```text
base=kz_first_raid_reclaim|d=2|c=vol_gt_p50_50+sess_london_kz
```

Condition IDs sorted lexicographically; `d` = depth; base id pinned.

### Amortized execution plan (mandatory)

Naive “full train per combo” is rejected (see estimates).

```
┌─────────────────────────────┐
│ 1. Load train OHLCV (21)    │
│ 2. Primitives + funding asof│
│ 3. Condition matrix (L cols)│
│ 4. Base entries + forward   │  ← once (~27s wall @4 workers today)
│    reach features           │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ For each combo signature:   │
│  - mask entries by AND      │
│  - dry-count n              │
│  - if n < 200: log skip     │
│  - else aggregate reach     │  ← vectorized; ms–tens of ms
│  - store row + primary p    │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ BH FDR on primary p-values  │
│ Write survivors table       │
│ Append hypothesis_log summary│
└─────────────────────────────┘
```

Parallelism: shard **combinations** across 8 cores after the shared base artifact is in memory (or memory-mapped). Do **not** shard by re-running gate/FSM per combo.

### Dry-count

Same rule as worker: `n_train ≥ 200` or skip reach aggregation. Skips still get a DB row (`status=skipped_low_n`) so the family and coverage are auditable; **skipped rows are excluded from the BH family** (they were not tested for edge).

### Primary endpoint for FDR (one test per combo)

To avoid multiplying horizons inside the correction:

- **Primary:** one-sided test at **1R** vs random walk `p₀ = 1/(1+1) = 0.5`.  
  `H₀: reach_1R ≤ 0.5` vs `H₁: reach_1R > 0.5`.  
- Test: exact binomial or normal approx with continuity; store `p_value_1r`.  
- **Secondary (logged, not in FDR family):** full reach table `diff_pp` at 0.5R…3R, best horizon, etc.

Promotion *policy* elsewhere still uses >5pp & n≥200; BH survivors are a **statistical shortlist**, not auto-promotion.

### Multiple-testing: Benjamini–Hochberg

1. Collect `p_i` for all combos with `status=tested` (n≥200), family size `m`.  
2. Sort `p_(1) ≤ … ≤ p_(m)`.  
3. Find largest `k` with `p_(k) ≤ (k/m) · q`, `q = 0.05`.  
4. Reject `H₀` for all `i ≤ k`.  
5. Store `bh_rank`, `bh_threshold`, `bh_significant` on each tested row; **copy significant rows** into `grid_survivors`.

**Dependence:** overlapping combos induce positive dependence. Classic BH remains valid for **PRDS** positive regression dependence under one-sided tests in many settings; document as “BH under PRDS assumption.” Optional sensitivity: Benjamini–Yekutieli (`q / H_m`) as a conservative companion column `by_significant` — not required for v1 UI, but cheap to compute.

**Do not** feed secondary horizons into the same FDR family.

---

## Component D — Searchable database

**Store:** SQLite file `data/grid/condition_grid.sqlite` (gitignored under `data/`; run manifests in `data/grid/runs/<run_id>/`).

### Schema (logical)

**`grid_runs`**

| Column | Notes |
|--------|-------|
| `run_id` | UUID / timestamp |
| `base_strategy_id` | `kz_first_raid_reclaim` |
| `library_version` | hash of condition ID list + versions |
| `L`, `depth_max` | 26, 3 |
| `m_planned`, `m_tested`, `m_skipped` | family accounting |
| `q_fdr` | 0.05 |
| `split_path`, `git_commit` | reproducibility |
| `wall_seconds`, `n_base_train` | |

**`grid_results`** (full database — every planned combo)

| Column | Notes |
|--------|-------|
| `run_id` | |
| `signature` | **PRIMARY KEY** with `run_id` |
| `depth` | 2 or 3 |
| `c1`, `c2`, `c3` | nullable c3 |
| `categories` | sorted unique category tags (denormalized for LIKE/JSON queries) |
| `n` | |
| `status` | `tested` \| `skipped_low_n` \| `skipped_mutex` |
| `reach_1r_pct`, `diff_pp_1r`, … | full table as JSON blob **and** key scalars indexed |
| `p_value_1r` | null if skipped |
| `bh_significant` | 0/1 after correction pass |
| `bh_rank` | |

**`grid_survivors`**

Strict subset: `INSERT … SELECT` where `bh_significant = 1`. Same columns + `promoted_eligible` (whether also `diff_pp_1r > 5` or best>5 — policy flag, default false until human).  
**UI rule:** any “cleared the bar” query defaults to **`grid_survivors`**, never to raw `diff_pp > 0` on `grid_results`.

**`condition_tags`** (optional normalized)

`(run_id, signature, condition_id, category)` for clean “involving volume” SQL without string parsing.

### Example queries

```sql
-- Combinations involving volume that cleared BH
SELECT s.*
FROM grid_survivors s
JOIN condition_tags t USING (run_id, signature)
WHERE t.category = 'volume' AND s.run_id = :run;

-- Top 20 by 1R edge with n > 500 (exploratory — NOT survivors)
SELECT signature, n, diff_pp_1r, p_value_1r, bh_significant
FROM grid_results
WHERE run_id = :run AND status = 'tested' AND n > 500
ORDER BY diff_pp_1r DESC
LIMIT 20;
```

CLI sketch (impl later): `ht-backtest grid-query --run … --category volume --survivors-only`.

### Distinction always visible

| Layer | Meaning |
|-------|---------|
| `grid_results` | Everything considered (incl. skips) |
| `status=tested` | Paid for a p-value |
| `bh_significant` | Cleared FDR |
| `grid_survivors` | Materialized significant set |
| `hypothesis_log` | Human/agent memory of the *run* + each survivor spine |

Reports and chat agents must say “tested” vs “BH survivor” explicitly.

---

## Estimates

### Combination counts

Unordered AND combinations, no mutex pruning:

| L | Depth 2 `C(L,2)` | Depth 3 `C(L,3)` | Total |
|---|----------------:|-----------------:|------:|
| 20 | 190 | 1,140 | **1,330** |
| 24 | 276 | 2,024 | **2,300** |
| **26** | **325** | **2,600** | **2,925** |
| 28 | 378 | 3,276 | **3,654** |
| 30 | 450 | 4,060 | **4,510** |

**Reference design (L=26):** **325** pairs + **2,600** triples = **2,925** planned signatures.

Mutex pruning (vol high/low, fund pos/neg, daily bull/bear, premium/discount): roughly **5–15%** fewer tested rows depending on exclusivity graph — order-of-magnitude unchanged.

Expected dry-count kills: unknown a priori; funding-missing and tight ANDs will inflate `skipped_low_n`. For timing, assume **60–90%** reach `n ≥ 200` given base n≈20k (triples still often large unless conditions are rare).

### Wall clock on 8 cores

**A. Naive (forbidden):** full FSM+train per combo ≈ 15–27s each.  
`2925 × ~20s / 8 ≈ 2–2.5 hours` is optimistic if jobs contend on IO; realistically **many hours** and wasteful.

**B. Amortized (required):**

| Phase | Estimate @8 cores |
|-------|-------------------|
| Load + primitives + funding as-of (21 train symbols) | 1–3 min (funding download once, then cache) |
| Base entries + forward reach (like golden, more cores) | **~15–25 s** wall (today ~27s @4 workers) |
| Condition matrices | 10–40 s |
| 2,925 filter + aggregate + insert | **30–90 s** (embarrassingly parallel) |
| BH + survivor materialize | <1 s |
| **Total** | **≈ 3–8 minutes** end-to-end after caches warm |

Cold start with funding history backfill for 30 symbols: add **5–20 min** one-time ingest (network), not per grid rerun.

### False positives (statistical)

Assume primary one-sided tests, **complete null** (no real edge), independence approximation for ballpark; real dependence → slightly different, not order-of-magnitude.

Let `m ≈ 0.8 × 2925 ≈ 2,340` tested combos (after low-n skips).

| Regime | Expected false positives (order of magnitude) |
|--------|-----------------------------------------------|
| **Uncorrected** α = 0.05 (call “significant” if `p < 0.05`) | `E[V] ≈ α·m ≈ 0.05 × 2340 ≈` **~117** |
| Same at L=20 / m≈1000 | **~50** |
| Same at L=30 / m≈3600 | **~180** |
| **BH FDR q = 0.05**, complete null, independent p-values | FWER ≤ q ⇒ `P(V≥1) ≤ 0.05`, so **`E[V]` is ≪ 1** (typically **0 survivors**; rarely a small handful) |
| BH when some alternatives are real | Among survivors, expected false discovery **rate** ≤ 5%; absolute FP count ≈ `0.05 × (#survivors)` |

**Interpretation for operators:** without BH, a “green” 1R screen on this grid will **manufacture ~O(100)** coin-flip “winners.” After BH, under pure noise you should usually see an **empty** survivor table; a non-empty table is interesting but still **train-only** and still needs pre-registered holdout before any promotion story.

Secondary peeking (`max diff_pp` over 6 horizons without correction) would roughly **multiply** uncorrected FPs — another reason the FDR family is **1R-only**.

---

## Failure modes

| Failure | Mitigation |
|---------|------------|
| Lookahead in funding/session features | As-of join tests; golden unit tests per condition |
| Re-running full backtest per combo | Amortized architecture gate in code review |
| Treating `diff_pp > 0` as success | Survivors table + BH flag only |
| Horizon fishing inside FDR | Single primary endpoint |
| Mutex / overlapping conditions inflate dependence | Document PRDS; optional BY column |
| Funding gaps → silent empty filters | `None` + coverage report per condition |
| Writing survivors into promotion path automatically | Survivors ≠ promoted; holdout pre-reg still mandatory |
| Agent re-tests exhausted categories via grid | Grid run logs to memory as one `signal_type=condition_grid_v1`; query before re-running same library hash |

---

## Implementation phases (after approval only)

1. **Funding ingest + as-of align** + coverage report.  
2. **Condition library** module + causal unit tests (no grid yet).  
3. **Base artifact builder** (entries + reach features + condition matrix).  
4. **Grid runner** → SQLite `grid_results`.  
5. **BH pass** → `grid_survivors` + CLI query + memory summary.  
6. **Audit report** hook for each survivor (optional).

No EA, no holdout, no parameter search on the base entry in this phase.

---

## Open choices for review

1. **Base entry:** confirm `kz_first_raid_reclaim` (first raid only) vs `raid_reclaim_all` (higher n, different spine).  
2. **L and exclusivity graph:** freeze the 26 IDs vs expand funding/prior.  
3. **Primary endpoint:** 1R vs RW only, or also require `diff_pp_1r > 2` (coin zone) *in addition to* BH for survivor materialization?  
4. **BY companion correction:** ship in v1 or defer?  
5. **Depth 3 in first run vs depth 2 only:** depth 2 alone is 325 tests (uncorrected E[FP]≈16 at α=0.05) — cheaper science if you want a pilot.

---

## Summary

| Item | Value |
|------|-------|
| Library | ~26 causal bar conditions in 6 categories; funding is the new dependency |
| Grid | All pairs + triples AND-filtered on fixed session-range reclaim base |
| Planned combos (L=26) | **325 + 2,600 = 2,925** |
| Wall clock (amortized, 8 cores, warm cache) | **~3–8 minutes** |
| Uncorrected FPs (α=0.05, m≈2340) | **~100–120** |
| BH q=0.05 under null | **~0 expected survivors** |
| Persistence | SQLite full DB + separate survivors; signature primary key; queryable by category/edge/n |
