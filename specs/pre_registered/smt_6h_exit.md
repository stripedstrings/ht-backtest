# Pre-registered holdout test: SMT holder 6h time exit

**Status:** pre-registered before any holdout evaluation of this rule.  
**Registered:** 2026-08-16  
**Holdout:** sealed for all other strategies and variants until this test is scored.

## Hypothesis

`smt_trade_holder_btc_sol` with a **fixed 6-hour time exit** has a **positive mean forward return** on holdout symbols.

## Strategy

- **Id:** `smt_trade_holder_btc_sol`
- **Version / hash (training artifact):** from batch `data/runs/batch_v1_20260816T130731Z` (parameter_hash `203c1e2d61942d59`)
- **Universe slice:** holdout only (`split == "holdout"`), using the already-generated trade list for this strategy. No other strategy is evaluated on holdout under this registration.

## Exit rule

- Enter as defined by `smt_trade_holder_btc_sol` (BTC sweeper / SOL holder / enter SOL).
- **Exit:** close at the **24th 15m bar after entry** (fixed 6-hour clock), regardless of price level, stop, or R target.
- No path-dependent stop/target resolution for this metric.

## Metric

- **Primary:** mean forward R at 24 bars after entry.
- Forward R for a trade: signed price change from entry close to the close 24 bars later, divided by that trade’s own risk distance.
  - Long: `(close[entry_bar + 24] − entry_price) / risk`
  - Short: `(entry_price − close[entry_bar + 24]) / risk`
- Trades without a complete 24-bar forward path are excluded from n.

## Secondary reporting (not the pass/fail bar)

- Median forward R at 24 bars
- n (holdout trades with full path)
- Percentage of trades with forward R > 0

## Interesting bar (pre-committed)

- **Interesting if:** mean forward R **> +0.10R**
- Rationale: half the training observation at this horizon (~+0.22R mean on train); a holdout mean above +0.10R is the minimum bar for discussing a candidate further.
- If mean ≤ +0.10R: test fails; holdout remains sealed for everything else and we reassess without expanding discovery on holdout.

## Explicit exclusions

- Do **not** run `low_vol_grab_reclaim` (or any other strategy) on holdout under this registration.
- Do **not** treat other horizons, R-reach tables, or path-dependent exits as part of this pre-registered claim.
