# Promotion gate

**Status:** architecture (no implementation in this document).  
**Date:** 2026-08-22  
**Protocol version:** `promotion_gate.v1`  
**Applies to:** every strategy, including those discovered autonomously.

This is a **hard constraint**, not a guideline. The reach-vs-random-walk benchmark and the pre-registered holdout protocol cannot be lowered, skipped, or replaced by paper trading, grid shortlists, or operator urgency. Changing any frozen constant is a **new protocol version** (new file or version bump + human review), never a runtime flag.

Companion: [`autonomous_system.md`](autonomous_system.md).

---

## Why this exists

Today the 5pp / n≥200 bar is a **default argument** in `src/ht_backtest/reports/comparison.py` (`min_edge_pp=5.0`, `min_n=200`). `strategy_comparison_table` will score holdout if a caller passes `split="holdout"`. That is a leak surface.

The gate replaces defaults with a **monotonic state machine** no component can bypass: generator, compiler, worker, grid, synthesizer, and the separate execution process.

---

## Frozen constants (`promotion_gate.v1`)

Changing these is a protocol version bump, not a config tweak.

| ID | Constant | Value | Role |
|----|----------|------:|------|
| C1 | `TRAIN_MIN_EDGE_PP` | `5.0` | Train promotion: `diff_pp > 5` at some horizon vs `1/(1+T)` |
| C2 | `TRAIN_MIN_N` | `200` | Train promotion and dry-count floor |
| C3 | `GRID_FDR_Q` | `0.05` | Benjamini–Hochberg and Benjamini–Yekutieli |
| C4 | `GRID_SURVIVOR_EDGE_PP` | `3.0` | BH/BY shortlist **and** `diff_pp_1r > 3`. Shortlist ≠ promotion |
| C5 | `PAPER_HORIZON_BARS` | `24` | 6 hours on 15m bars |
| C6 | `PAPER_MIN_MEAN_FWD_R` | `0.0` | Sign only: mean forward R at 6h **≥ 0** |
| C7 | `PAPER_MIN_N` | `50` | Paper trades with a complete 6h path |
| C8 | `PAPER_MIN_CALENDAR_DAYS` | `30` | Minimum paper window |
| C9 | `LIVE_AUDIT_DAYS` | `30` | Rolling live reach-vs-RW window |
| C10 | `LIVE_DECAY_MIN_N` | `50` | Below this n, live audit stays `observe` (no decay call, no new edge claim) |
| C11 | `CORR_MAX_ABS` | `0.7` | Pairwise return correlation cap before a second live/paper strategy |
| C12 | `MAX_POSITION_EQUITY_PCT` | `0.02` | Per-strategy notional vs equity |
| C13 | `DAILY_LOSS_EQUITY_PCT` | `0.05` | UTC-day loss halt |
| C14 | `WEEKLY_NEW_CONDITION_SET_SLOTS` | (product choice; freeze at impl, recommend `10`) | New YAML/grid condition sets per UTC week |

**Illegal:** a `skip_validation`, `min_edge_pp` override, “combination discount,” or environment variable that changes C1–C13 at runtime.

---

## Paper vs holdout (do not confuse)

| Gate | What it answers | What it does **not** answer |
|------|-----------------|----------------------------|
| **Holdout** (pre-registered metric) | **Statistical confirmation** of the research claim | Whether the fill engine, keys, or ops stack work |
| **Paper** (C5–C8) | **Operational** readiness: fills, crashes, limits, sign of 6h forward R | Whether an edge exists. Holdout already did that |

**Paper passed ≠ edge confirmed.** Paper is a sign-only operational gate: mean forward R at 6 hours ≥ 0 at n ≥ 50 over ≥ 30 calendar days. Magnitude is ignored. The holdout (and only the holdout) is the statistical gate.

If paper n &lt; 50 at day 30, **extend the calendar window**. Do not lower `PAPER_MIN_N`. Do not treat a small-n positive mean as confirmation. Do not reopen holdout to “help” paper.

Forward R at 6h matches the existing SMT pre-reg definition (`specs/pre_registered/smt_6h_exit.md`): signed close-to-close change over 24 bars divided by the trade’s own risk. Trades without a complete 24-bar path are excluded from n.

---

## State machine

States are monotonic except `live_suspended` (terminal for that passport until a **human** resume, which is a new live-enable, not a reverse transition).

```
ingested
  → dry_counted
  → train_scored
  → train_promoted          # C1 ∧ C2 on train only
  → holdout_preregistered   # human git commit of specs/pre_registered/*.md
  → holdout_cleared         # dedicated scorer, exact pre-reg metric only
  → ea_eligible             # synthesizer may emit passport
  → paper_enabled           # human enable in execution process
  → paper_cleared           # C5–C8 and correlation check
  → live_enabled            # human enable in execution process
  → live_suspended          # decay, daily loss, or kill
```

Unscored / killed branches (`dry_count_killed`, `train_failed_bar`, `holdout_failed`, `paper_failed`) are terminal for that `(strategy_id, parameter_hash)`. They do not skip forward.

### Illegal transitions (must be unrepresentable)

- `train_scored` or `train_promoted` → `ea_eligible` / paper / live
- Holdout score without a **git-committed** pre-reg file that names `strategy_id`, `parameter_hash`, metric, and bar
- Any generator, worker, grid, or combination path scoring holdout
- Lowering C1/C2 because a combination “needs a chance”
- Execution loading a strategy whose passport is not `holdout_cleared` (paper) or `paper_cleared` (live)
- Auto-resume from `live_suspended`
- Treating grid BH survivors or paper pass as `holdout_cleared`

### Holdout scorer contract

A single dedicated entry point. It must:

1. Resolve the pre-reg markdown at a specified git commit SHA.
2. Verify the blob names this `strategy_id` + `parameter_hash`.
3. Score **only** the declared metric on rows with `split == "holdout"`.
4. Refuse if the file is working-tree-only (uncommitted).

`strategy_comparison_table(..., split="holdout")` must not remain callable from the worker, generator, grid, or compiler. Those paths compile without a holdout argument.

---

## `condition_signature`

Parameter tweaks must not reopen a dead family.

### Definition

```text
condition_signature = ",".join(sorted(unique condition_ids))
```

- **Condition ids only.** Thresholds, lookbacks, `reclaim_win`, and other YAML/parameter values are **excluded** (parameter-value-agnostic).
- Empty filter set → empty string `""`.
- Ids are library ids (`volume_high`, `london_session`, …), not display names.

### Namespaced refuse key

The generator refuse key is:

```text
(base_entry_id, condition_signature)
```

`base_entry_id` is the allowlisted geometry (`kz_first_raid_reclaim`, `smt_trade_holder_btc_sol`, …). Empty signatures on **different** bases do not collide (an unconditioned SMT decay does not block unconditioned first-raid).

### Memory log schema addition

Extend `data/memory/hypothesis_log.csv` (and `LOG_COLUMNS` in `src/ht_backtest/memory/hypothesis_log.py` when implemented):

| Column | Required | Notes |
|--------|:--------:|-------|
| `condition_signature` | yes | Canonical sorted-id string above |
| `base_entry_id` | yes | Allowlisted geometry id |
| `split_id` | yes | Bound forever to this hash (e.g. `v1`) |
| `promotion_state` | yes | State-machine value |
| `holdout_result` | existing | Includes `live_decay` when live audit suspends |

Existing columns stay. Backfill: known strategies with no filters get `condition_signature=""` and their registry id as `base_entry_id`.

### Generator rule (mandatory)

If **any** memory row has `holdout_result=live_decay` (or `promotion_state=live_suspended` due to decay) with the same `(base_entry_id, condition_signature)`, the generator **refuses** the new `parameter_hash`. No enqueue, no dry-count, no “try a different lookback.”

This closes the parameter-variation loop: decayed `volume_high ∧ london_session` on `kz_first_raid_reclaim` cannot return as the same pair with `reclaim_win=5`.

---

## FDR inheritance (compiler-enforced)

YAML condition ANDs on a **fixed base** are the same statistical family as the grid. They must not buy a second unpaid test.

### Lookup

Canonical combo key:

```text
(base_entry_id, frozenset(condition_ids), library_version)
```

`library_version()` is the existing condition-library hash. Grid SQLite (`data/grid/condition_grid.sqlite`) already keys results by sorted condition ids, depth, and base.

### Rules (enforced in `discovery/compile.py`, not documentation)

1. **Before** dry-count or train, the compiler computes `frozenset(condition_ids)` from the YAML.
2. If a `grid_results` row exists for the same key with `status=tested` (or `skipped_*`):
   - **Inherit** that row’s `n`, reach table, `p_value_1r`, BH/BY flags.
   - **Do not** add a member to the FDR family (`m` unchanged).
   - **Do not** consume a weekly new-condition-set slot (C14).
   - Candidate is marked `inherited_grid=true` with the grid `run_id` + `signature`.
   - Inherited BH-non-significant results **cannot** be re-tested as a “new” YAML strategy to fish a different horizon.
   - Inherited BH-significant shortlist still **is not promotion**; the hash still needs C1/C2 if it is an independent geometry, and still needs pre-reg + holdout. (On the grid base, a filter combo that already failed FDR stays failed.)
3. If **no** grid row exists:
   - This is a **new condition set**.
   - It **consumes one** weekly new-condition-set slot (C14).
   - If the weekly budget is exhausted → compile error / queue reject `weekly_fdr_budget`.
   - The new set must be written into the grid family (run a grid increment or register the signature in `grid_results` with a pending test) **before** its p-value can count. The compiler must not train a new filter combo as if it were an independent unpaid test.
4. Mutex / `asia_session` exclusions follow the existing grid rules. Mutex YAML → `skipped_mutex`, no slot, no train.

### Independent geometries (not filter combos)

A YAML that is a **new allowlisted base** (different `base_entry_id`) with no extra conditions is **not** an FDR grid member. It is an independent strategy: dry-count C2, train C1/C2, weekly **candidate** budget (generator K), not C14. C14 is for new **condition sets** on a base that the grid already owns (today: `kz_first_raid_reclaim`).

If a new base is later added to the grid, its filter combos join **that** base’s FDR family.

### Why the compiler

A documented rule will be skipped under time pressure. Inheritance and slot accounting belong in `compile_candidate_file` (or a function it must call). A candidate that reaches `queued/` without an inheritance decision is a compiler bug.

---

## Passport (execution may load only this)

JSON, no secrets, content-hashed. Execution process verifies the hash and `promotion_state`.

Required fields:

| Field | Notes |
|-------|--------|
| `schema_version` | `promotion_gate.v1` |
| `strategy_id` | |
| `parameter_hash` | |
| `base_entry_id` | |
| `condition_signature` | |
| `split_id` | Frozen binding |
| `research_git_commit` | |
| `train_reach_table` | Full 0.5R–3R |
| `pre_reg_path` | Under `specs/pre_registered/` |
| `pre_reg_commit_sha` | |
| `holdout_metric` | Copied from pre-reg |
| `holdout_result` | Pass/fail + numeric |
| `promotion_state` | Must be `holdout_cleared` to paper-load; `paper_cleared` to live-load |
| `content_hash` | Hash of all other fields |

Missing, forged, or wrong-state passport → refuse to trade. Research repo never stores API keys. Execution lives in a **separate process** (recommended: separate repo).

---

## Split binding

`(strategy_id, parameter_hash)` binds to `split_id` forever. Retraining the same hash on a later split that unseals former holdout is forbidden. Bars after a split’s `overall_end` are `unscored_future` until a **new** hash on a **new** split version.

---

## Human checkpoints that this gate depends on

See [`autonomous_system.md`](autonomous_system.md) § Human checkpoints. Two of five are undetectable if skipped:

1. **Pre-reg commit** — without it, holdout scoring is not science.
2. **Compiler expansion review** — a look-ahead or FDR-bypass bug in YAML→Strategy is not caught by reach tables that use the buggy path.

The gate’s dedicated holdout scorer implements (1) in code. (2) is a process gate: the autonomous loop must not be enabled after a compiler change until that review is recorded.
