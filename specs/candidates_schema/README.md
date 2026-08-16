# Audit intake (MVP)

Intake captures a discretionary idea and produces a **candidate YAML** plus
confidence, then dry-counts on the crypto 15m train universe.

## Quick start (no Claude)

```bash
ht-backtest intake specs/candidates_schema/examples/intake_first_raid.yaml --fixture
```

## With Claude

```bash
pip install -e ".[discovery]"
set ANTHROPIC_API_KEY=...
ht-backtest intake path/to/intake.yaml
```

## Web form

```bash
ht-backtest intake-serve --fixture
# open http://127.0.0.1:8765/
```

## Outputs

- `data/candidates/queued/*.yaml` — dry-count passed (n≥200); ready for the training worker
- `data/candidates/rejected/*` — intake rejects: `n_too_low`, `repainting_indicator_detected`, `insufficient_data`, …
- `data/candidates/completed/{strategy_id}/` — worker finished: reach table, promotion flag, trades
- `data/candidates/failed/` — worker rejects: compilation failure or full dry-count n too low

## Queue worker

```bash
ht-backtest worker --run-queue
# or poll:
ht-backtest worker --loop --interval 60
```

Human reviews `completed/` before any holdout. Golden:

```bash
.venv/Scripts/python.exe scripts/run_worker_golden_kz_first.py
```

## Limits (MVP)

- Dry-count / queue path supports **crypto + 15m** only (existing Binance cache).
- Compiler maps allowlisted `dry_count.method` values onto existing Strategy classes.
- Does not open holdout.
