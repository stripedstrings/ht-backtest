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

- `data/candidates/queued/*.yaml` — dry-count passed (n≥200); ready for a later training batch worker
- `data/candidates/rejected/*.yaml` — reasons: `n_too_low`, `repainting_indicator_detected`, `insufficient_data`, `untranslatable_dry_count`, `look_ahead_risk`
- Matching `*_result.json` with confidence and dry-count n

## Limits (MVP)

- Dry-count / queue path supports **crypto + 15m** only (existing Binance cache).
- Dry-count methods are allowlisted (session raid/reclaim family); Claude must map to those.
- Does not yet compile YAML → live `Strategy` or run the training batch automatically.
