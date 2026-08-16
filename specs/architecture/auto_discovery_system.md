# Architecture: Automated Strategy Discovery & Testing

**Status:** design only — no implementation yet.  
**Date:** 2026-08-16  
**Constraint:** sits on top of the existing reach-vs-RW engine; does not weaken train/holdout separation, golden HT integrity, or hypothesis memory.

---

## Goal

A system that, without human intervention for routine cases:

1. Ingests strategy ideas from external sources (URLs, files, text).
2. Translates them into `Strategy` subclasses (or YAML component configs) via LLM.
3. Dry-counts, training-batches, and logs to hypothesis memory.
4. Proposes combinations of sub-threshold positive signals.
5. Generates executable trading code **only** after a strategy clears the holdout pre-registration protocol.

Human gates remain where false positives are expensive: holdout unlocks, live capital, and any translation that touches look-ahead / repainting.

---

## Non-negotiables (inherited from the platform)

| Rule | Implication for automation |
|------|----------------------------|
| Reach vs `1/(1+T)` on **train** for discovery | Test queue never scores holdout for promotion decisions |
| Holdout sealed until **pre-registered** | Combination engine and EA generator cannot “peek” holdout |
| HT golden (`n=4483`, 0.00pp) after engine/context changes | Ingestion must not mutate pooler/gates without a golden gate |
| Hypothesis memory before new strategies | Ingestion + combination must query memory for category/signal exhaustion |
| Independent strategies, not HT+filters | Translator emits full `Strategy` (or composable signal modules), not tag filters on HT |
| Dry-count `n ≥ 200` or kill | Test queue enforces this before training spend |

---

## System overview

```
┌─────────────┐     candidate.json      ┌──────────────────┐
│  Ingestion  │ ───────────────────────►│  Candidate queue │
│  pipeline   │   (+ confidence, risks) │  (disk / SQLite) │
└─────────────┘                         └────────┬─────────┘
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │  Test queue      │
                                        │  worker          │
                                        └────┬───┬─────────┘
                           dry-count kill ◄──┘   │
                                                 │ train batch + memory append
                                                 ▼
                                        ┌──────────────────┐
                     ◄──────────────────│ Hypothesis memory│
                     │                  │ + combination    │
                     │ sub-threshold    │   engine         │
                     │ flags (>2,<5pp)  └────────┬─────────┘
                     │                           │ new candidates
                     └───────────────────────────┘
                                                 │
                    only if pre-reg holdout CLEAR ▼
                                        ┌──────────────────┐
                                        │  EA generator    │
                                        └──────────────────┘
```

Shared stores:

- `data/candidates/` — ingested candidate files (pending / killed / trained).
- `data/memory/hypothesis_log.csv` — durable results (exists).
- `specs/pre_registered/` — holdout protocol docs (exists).
- `data/runs/batch_*` — train artifacts (exists).
- `specs/generated/` or `src/ht_backtest/strategies/generated/` — LLM-written strategies (new; quarantined import path).

---

## Component 1 — Ingestion pipeline

### Responsibility

Accept any strategy source → produce a **candidate file** ready for the queue:

- Structured metadata (id, category, signal_type, source URI, plain-English spine).
- Either: generated `Strategy` Python module path, **or** YAML component config that a thin adapter loads.
- Translation confidence score ∈ [0, 1].
- Risk flags: `repainting_risk`, `look_ahead_risk`, `uses_future_bars`, `needs_aux_bars`, `untranslatable`.
- Memory collision report: same category/signal already exhausted?

### What already exists

| Asset | Use |
|-------|-----|
| `Strategy` protocol + `TradeCandidate` (`strategies/base.py`) | Target shape for translation |
| `requires_symbols` / `ctx.aux_bars` | Cross-asset candidates |
| `theoretical_category` / `signal_type` on hypotheses | Required fields on candidates |
| `hypothesis_log` + `prior_for_category` / `query_log` | Pre-check exhaustion |
| `GateParams` / primitives / session-range engine | Building blocks the LLM should prefer over inventing new geometry |
| Dry-count pattern (`scripts/dry_count_hypotheses.py`) | Template for “loosest defensible” count defs |
| OHLCV cache + split `v1` | No new data sources required for v1 automation |

### What needs to be built

1. **Source adapters:** URL fetch (HTML/Markdown strip), local file (`.pine`, `.txt`, `.md`, `.py`), paste/API payload.
2. **Claude translation prompt + schema:** JSON schema for candidate; prefer composing from an allowlisted primitive catalog (ATR, sessions, grabs, Asia, volume percentiles, SMT helper) rather than free-form Python.
3. **Two output modes:**
   - **A (safer):** YAML “signal graph” → deterministic compiler → `Strategy` subclass (recommended default).
   - **B (flexible):** LLM writes Python into `strategies/generated/` → static check + sandbox import.
4. **Static risk scanner:** reject `shift(-k)`, center-window indicators, `request.security` without lookahead guards, future merges on timestamp, ffill from future aux, etc.
5. **Confidence heuristic:** schema completeness + risk flags + overlap with known primitives − novelty of unlisted ops.
6. **Candidate writer:** `data/candidates/<id>_<hash>.json` + optional `.yaml` / `.py`.

### Main failure modes

| Failure | Consequence |
|---------|-------------|
| Silent look-ahead in translation | Fake train edge → wasted holdout or worse, false promotion |
| Repainting Pine ported literally | Backtest ≠ live |
| LLM invents new “gates” that duplicate HT with filters | Confounds axes; memory pollution |
| Over-confident score on garbage text | Queue floods with junk |
| Category mis-label | Exhaustion checks lie; combinations mis-pair |

### Human vs automated

| Fully automate | Human must review |
|----------------|-------------------|
| Fetch, strip, schema validate, memory collision check | Any candidate with `look_ahead_risk` or `repainting_risk` = true |
| Low-confidence reject (&lt; threshold, e.g. 0.55) auto-kill | Mode B (raw Python) until YAML compiler is mature |
| Queue enqueue for clean Mode A candidates | Novel data sources beyond OHLCV (order book, tweets, etc.) |
| | First N translations after prompt changes (spot-check) |

### Automation risk flag — **HIGH**

Full automation of **Python emission (Mode B)** is genuinely risky: one look-ahead bug historically moved results from large positive to null. Prefer Mode A (YAML → compiler) for unsupervised runs; Mode B only behind human review or extremely strict AST allowlists.

---

## Component 2 — Test queue worker

### Responsibility

Pull next candidate → dry-count → (if n≥200) training batch → append hypothesis memory → flag combination engine if any horizon has **+2pp &lt; diff_pp &lt; +5pp** (and n≥200). Kill / archive otherwise. Never open holdout.

### What already exists

| Asset | Use |
|-------|-----|
| Dry-count script pattern + train-only split classify | Port to library function per candidate |
| `generate_pooled_trades` + process pool | Training run |
| `run_batch` / comparison table / promotion bar (5pp, n≥200) | Scoring |
| Hypothesis memory append on batch | Logging |
| Primitive cache + stage timings | Cost control (~&lt;60s/hypothesis target historically) |
| Registry `get_strategy` / `register_strategy` | Load generated strategies |

### What needs to be built

1. **Queue store:** filesystem queue (`pending/`, `running/`, `done/`, `killed/`) or SQLite — start with filesystem.
2. **Worker loop:** single-process first; later multi-worker with file locks.
3. **Per-candidate dry-count contract:** candidate must declare countable event definition (or LLM supplies it at ingest); worker refuses “no dry-count spec.”
4. **Auto YAML batch fragment** → call existing batch runner for one strategy (or thin wrapper).
5. **Sub-threshold flag:** write `data/memory/combination_watchlist.csv` (or column on log) when best train edge ∈ (2, 5) pp.
6. **Golden gate:** if candidate registration touches engine/pooler, run HT golden; else skip.
7. **Idempotency:** skip if `strategy_id`+`parameter_hash` already in memory.

### Main failure modes

| Failure | Consequence |
|---------|-------------|
| Dry-count definition looser than live strategy | Train n looks fine; live logic rarer → underpowered |
| Dry-count tighter than live | False kill of good ideas |
| Worker registers broken strategy → batch crash | Stuck queue |
| Memory append without train comparison | Blind discovery |
| Treating holdout parquet “just for curiosity” | Protocol breach |

### Human vs automated

| Fully automate | Human must review |
|----------------|-------------------|
| Dry-count kill (n&lt;200), train batch, memory log | Strategy that crashes golden or mutates core gates |
| Sub-threshold watchlist flagging | Interpreting *why* an edge exists (skew, single-symbol, etc.) before holdout pre-reg |
| Kill on risk flags from ingest | Any proposed holdout pre-registration text (see Component 4 gate) |
| Retry transient IO errors | Raising promotion bar / changing 2pp watchlist threshold |

### Automation risk flag — **MEDIUM**

Safe **if** holdout stays sealed and dry-count specs are mandatory. Risk rises if the worker is allowed to invent dry-count definitions that don’t match the strategy code.

---

## Component 3 — Combination engine

### Responsibility

Read hypothesis memory (+ watchlist) → find **compatible** positive-but-sub-threshold signals → Claude reasons over memory → emit **new** independent strategy candidates (not “AND filters on HT”) → enqueue.

Compatibility heuristics (deterministic pre-filter before LLM):

- Different `theoretical_category` or clearly different `signal_type`.
- Shared geometry only if composition is explicit (e.g. volume condition **on the same grab event** as timing) — still emitted as one new strategy id.
- Reject pairing two exhausted identical signal types.
- Reject pairing rivals that are mutually exclusive by construction (e.g. high-vol vs low-vol on same event) unless the proposal is an explicit “regime switch.”

### What already exists

| Asset | Use |
|-------|-----|
| `hypothesis_log.csv` with categories, summaries, holdout notes | Input corpus |
| `format_prior_results_summary` / `query_log` | Context packing for Claude |
| Sub-threshold examples already observed (e.g. low_vol ~+1.8pp, SMT holder ~+1.5pp — below 2pp watchlist; adjust threshold or include “best effort” near-misses carefully) | Seeds |
| Strategy independence doctrine | Combinations must be new strategies |

### What needs to be built

1. **Watchlist reader** and feature extraction from `training_result_summary`.
2. **Compatibility graph** (deterministic).
3. **Claude combination prompt:** memory dump + compatibility edges → 1–3 proposals with spines + YAML specs.
4. **Dedup** against memory and pending queue.
5. **Rate limits:** max combinations per week; max depth (no combine-of-combines without human).
6. **Feedback loop:** combined candidates go through **ingestion risk scan + test worker** like any other candidate.

### Main failure modes

| Failure | Consequence |
|---------|-------------|
| Multiple-testing / stacking | Combinatorial explosion of false positives |
| AND-filter on weak signals | Curve-fit that dies on holdout |
| LLM proposes “HT + tag” again | Violates independence doctrine |
| Combining correlated signals (first raid ∩ first 30m) | Pseudo-replication |
| Ignoring skew (low_vol mean≫median) | Combinations inherit fragile tails |

### Human vs automated

| Fully automate | Human must review |
|----------------|-------------------|
| Deterministic pairing shortlist | Approve combination **depth &gt; 1** or &gt;N candidates/week |
| Draft proposals + enqueue Mode A YAML | Any combination that includes a holdout-failed signal as if it were validated |
| Memory-aware rejection of duplicates | Changing the 2pp watchlist / 5pp promotion bars |

### Automation risk flag — **HIGH**

This is the classic multiple-comparisons machine. Fully unsupervised combination search will **manufacture** apparent edges. Mitigations that are not optional:

- Hard weekly budget on combinations.
- Combinations still need dry-count ≥200 and train ≥5pp for promotion path — do not lower the bar for combos.
- Holdout still requires **fresh pre-registration** naming the combination explicitly.
- Prefer human approval of the combination *proposal list* even if enqueue+train is automatic for Mode A.

---

## Component 4 — EA generator

### Responsibility

Triggered **only** when:

1. Train promotion criteria met (diff_pp &gt; 5 at some horizon, n≥200), **and**
2. A holdout test was **pre-registered** (markdown in `specs/pre_registered/`), committed, **then** scored, **and**
3. Pre-registered metric clears its stated bar.

Then emit:

- Python/ccxt live module stub (preferred; matches stack), and/or MQL5 stub.
- Header comments: strategy id, parameter_hash, pre-registration path, timestamp, holdout metric result, train summary.

### What already exists

| Asset | Use |
|-------|-----|
| Pre-registration protocol + `specs/pre_registered/smt_6h_exit.md` | Template for gate |
| Holdout classification on trades / split manifest | Evaluation slice |
| Strategy metadata (description, hash) | Comment block + identity |
| ccxt downloader patterns | Live data path familiarity — **not** an execution engine yet |

### What needs to be built

1. **Promotion state machine:** `train_promoted` → `holdout_preregistered` → `holdout_cleared` → `ea_eligible`.
2. **Pre-reg writer assist:** Claude drafts `specs/pre_registered/*.md`; **human commit** still required before score (keeps “commit before touch” intact).
3. **Holdout runner** for the exact pre-registered metric only (no fishing).
4. **Codegen templates:** Python bracket/time-exit executor skeleton; MQL5 stub with same parameters.
5. **Artifact freeze:** pin parameter_hash + git commit in generated file.

### Main failure modes

| Failure | Consequence |
|---------|-------------|
| Generating EA on train-only “almost” | Live trading noise / loss |
| Pre-reg after peeking holdout | Invalid science; false confidence |
| Codegen ≠ backtest fills (intrabar, fees, funding) | Live drift |
| MQL5 semantics ≠ Python 15m close model | Broker mismatch |
| Auto-sizing / keys in repo | Security incident |

### Human vs automated

| Fully automate | Human must review |
|----------------|-------------------|
| Draft EA from template after `holdout_cleared` | **Commit** of pre-registration doc (protocol integrity) |
| Embed metrics in comments | **Go-live:** keys, size, venues, kill-switch |
| Archive EA under `artifacts/ea/<id>/` | Fee/slippage model vs backtest assumptions |
| | Any strategy with intrabar ambiguity flags |

### Automation risk flag — **CRITICAL** for live capital

Automating **code generation** after a clean holdout clear is reasonable. Automating **deployment / order routing / API keys** is not. EA generator must stop at “artifact written”; human starts live trading.

Also: automating the **pre-registration commit** would defeat the point of pre-registration (machine could pre-reg after soft-peek). Keep “human commits pre-reg before holdout touch” as a hard gate.

---

## Cross-cutting concerns

### Trust boundary for LLM code

```
Untrusted: URL text, LLM Python (Mode B), combination prose
Trusted:   YAML schema compiler, dry-count runner, batch runner, memory, golden test
```

Never `exec` LLM Python in the same process as the research engine without AST allowlist + subprocess + resource limits.

### Cost & throughput

- Target: dry-count ≪ train; train ~tens of seconds per strategy with cache/workers.
- Budget Claude calls: ingest once, combine rarely.
- Dedup by content hash of source + parameter_hash.

### Observability

- Candidate lifecycle events → `data/candidates/events.jsonl`.
- Every kill reason machine-readable (`dry_count`, `risk_flag`, `memory_dup`, `batch_error`).
- Dashboard later; v1 is log files + memory summary.

### What this system must never do

- Open holdout for discovery or combination fitting.
- Lower promotion bar automatically because “combinations need a chance.”
- Mutate HT / gates without golden.
- Generate EA from train-only results.

---

## Build sequence

### Phase 0 — Harden contracts (short)

1. Formalize **candidate JSON schema** + **dry-count spec** fields.
2. Extend memory with optional `watchlist_flag` / combination notes (or side CSV).
3. Document promotion state machine in `specs/`.
4. No LLM yet.

### Phase 1 — Test queue worker (no LLM)

1. Filesystem queue + worker CLI.
2. Manual candidate drop-in (hand-written Strategy or YAML).
3. Dry-count → train → memory → watchlist flag.
4. Prove loop on one known strategy replay (idempotent).

**Exit:** unsupervised train pipeline works for trusted candidates.

### Phase 2 — Ingestion Mode A (YAML compiler + Claude)

1. Primitive catalog + YAML schema.
2. Deterministic compiler → `Strategy`.
3. Claude translates sources → YAML only.
4. Risk scanner + confidence + memory pre-check.
5. Human review gate for risk flags; auto-queue otherwise.

**Exit:** URL/text → candidate → queue without writing free-form Python.

### Phase 3 — Combination engine (constrained)

1. Watchlist → compatibility graph.
2. Claude proposes ≤K combinations/week into YAML.
3. **Human approves proposal batch** (or auto if K=1 and both legs Mode A — product choice; recommend human approve initially).
4. Queued combos use **same** 5pp bar; no special casing.

**Exit:** memory-aware recombination without holdout leakage.

### Phase 4 — Holdout protocol automation assist + EA stub

1. Claude drafts pre-reg markdown; human commits.
2. Worker runs **only** the pre-registered metric on holdout.
3. On clear → EA template generator (Python stub first).
4. MQL5 stub later if needed.

**Exit:** end-to-end path from idea to EA artifact with science gates intact.

### Phase 5 — Optional Mode B (quarantined Python)

1. AST allowlist + subprocess runner.
2. Always human review before queue.
3. Only if Mode A coverage is insufficient.

---

## Component risk summary

| Component | Full automation? | Why |
|-----------|------------------|-----|
| Ingestion (YAML Mode A) | Mostly yes, with risk-flag hold | Schema-constrained |
| Ingestion (Python Mode B) | **No** unsupervised | Look-ahead / silent bugs |
| Test queue worker | Yes (train only) | Engine already trusted |
| Combination engine | **Risky unsupervised** | Multiple testing; needs budget + prefer human proposal approval |
| Holdout pre-reg commit | **Never fully auto** | Defeats pre-registration |
| Holdout scoring (after commit) | Yes | Metric fixed |
| EA codegen | Yes after clear | Artifact only |
| Live trading / keys | **Never** | Capital at risk |

---

## Suggested repo layout (future)

```
src/ht_backtest/
  discovery/
    ingest/          # sources, Claude client, risk scan
    compile/         # YAML → Strategy
    queue/           # worker, states
    combine/         # memory → proposals
    ea/              # templates
data/candidates/
specs/candidates_schema/
specs/pre_registered/
specs/architecture/  # this document
```

---

## Open product decisions (resolve before Phase 2)

1. **Watchlist threshold:** strict `(2,5)` pp vs also flag best-edge ∈ `(1,2)` for research-only (not combo fuel).
2. **Combination auto-enqueue vs human approve.**
3. **YAML-only vs allow Mode B early.**
4. Whether Claude API runs in-process or via a separate `discovery` service with its own rate limits/keys (keys must not live in research git history).
