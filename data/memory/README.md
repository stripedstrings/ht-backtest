# Hypothesis memory

`hypothesis_log.csv` is the durable record of every batch-tested strategy:
category, signal type, train summary, holdout (if any), and promotion.

Before implementing a new strategy, query it:

```python
from ht_backtest.memory import prior_for_category, query_log

print(prior_for_category("volume"))
print(query_log(category="cross-asset", signal_type="smt_trade_holder"))
```

Every `ht-backtest batch` run prints a PRIOR RESULTS summary (exhausted vs
unexplored categories) and appends one row per strategy automatically.
