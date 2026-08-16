"""Unit tests for hypothesis memory log (no universe run)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ht_backtest.memory.hypothesis_log import (
    append_records,
    format_prior_results_summary,
    load_log,
    prior_for_category,
    query_log,
    records_from_batch_comparison,
    training_summary_from_comparison,
    update_holdout_result,
)
from ht_backtest.strategies.registry import get_strategy


def test_training_summary_picks_best_diff():
    comp = pd.DataFrame(
        {
            "strategy_id": ["x", "x"],
            "target_R": [1.0, 2.0],
            "n": [100, 100],
            "diff_pp": [0.5, 1.8],
            "promoted": [False, False],
        }
    )
    summary, promoted = training_summary_from_comparison(comp)
    assert "best_diff_pp=+1.80" in summary
    assert "at 2.0R" in summary
    assert promoted is False


def test_append_and_prior_summary(tmp_path_factory):
    # Prefer repo-local temp dir (Windows may block pytest's AppData temp).
    try:
        log_path = Path(tmp_path_factory.mktemp("mem")) / "hypothesis_log.csv"
    except PermissionError:
        log_path = Path("data/memory/_test_hypothesis_log.csv")
        if log_path.exists():
            log_path.unlink()

    s = get_strategy("low_vol_grab_reclaim")
    comp = pd.DataFrame(
        {
            "strategy_id": ["low_vol_grab_reclaim"] * 2,
            "strategy_parameter_hash": [s.metadata().parameter_hash] * 2,
            "description": [s.metadata().description] * 2,
            "target_R": [2.0, 3.0],
            "n": [1614, 1614],
            "diff_pp": [0.5, 1.8],
            "promoted": [False, False],
        }
    )
    records = records_from_batch_comparison(comp, strategies={"low_vol_grab_reclaim": s}, date="2026-08-16")
    append_records(records, log_path)
    df = load_log(log_path)
    assert len(df) == 1
    assert df.iloc[0]["theoretical_category"] == "volume"
    assert "low_vol_grab_reclaim" in df.iloc[0]["key_parameters"]
    assert df.iloc[0]["holdout_result"] == ""

    text = format_prior_results_summary(log_path)
    assert "volume" in text
    assert "EXHAUSTED" in text
    assert "UNEXPLORED" in text

    prior = prior_for_category("volume", path=log_path)
    assert "low_vol_grab_reclaim" in prior
    q = query_log(category="volume", signal_type="low_vol_grab_reclaim", path=log_path)
    assert len(q) == 1

    update_holdout_result(
        "low_vol_grab_reclaim",
        "mean_fwd_R_6h=+0.01 n=10 fail",
        path=log_path,
    )
    df2 = load_log(log_path)
    assert "mean_fwd_R_6h" in str(df2.iloc[0]["holdout_result"])
    if log_path.name.startswith("_test_"):
        log_path.unlink(missing_ok=True)
