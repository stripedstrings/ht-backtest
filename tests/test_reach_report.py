import numpy as np
import pandas as pd

from ht_backtest.reports.reach import (
    format_reach_table,
    promotion_flags,
    reach_vs_random_walk,
    tag_on_off_tables,
)


def test_reach_table_basic_counts_and_baseline():
    df = pd.DataFrame(
        {
            "reach_0.5R": [True, True, False, False],
            "reach_1.0R": [True, False, False, False],
        }
    )
    out = reach_vs_random_walk(df, targets=(0.5, 1.0))
    row_05 = out[out["target_R"] == 0.5].iloc[0]
    assert row_05["n"] == 4
    assert row_05["reach_pct"] == 50.0
    assert abs(row_05["random_walk_pct"] - 100 / 1.5) < 1e-9
    row_10 = out[out["target_R"] == 1.0].iloc[0]
    assert row_10["reach_pct"] == 25.0
    assert abs(row_10["random_walk_pct"] - 50.0) < 1e-9


def test_reach_table_excludes_unknown_from_denominator():
    df = pd.DataFrame({"reach_0.5R": [True, False, np.nan, np.nan]})
    out = reach_vs_random_walk(df, targets=(0.5,))
    row = out.iloc[0]
    assert row["n"] == 2  # the two NaNs are excluded, not counted as misses
    assert row["reach_pct"] == 50.0


def test_tag_on_off_tables_excludes_undefined_median_tag_from_both_sides():
    df = pd.DataFrame(
        {
            "big_displacement": [1, 1, 0, 0, -1, -1],
            "reach_1.0R": [True, True, False, False, True, False],
        }
    )
    on, off, cmp_ = tag_on_off_tables(df, "big_displacement", on_value=1, off_value=0, exclude_values=(-1,), targets=(1.0,))
    row = cmp_.iloc[0]
    assert row["n_on"] == 2
    assert row["n_off"] == 2
    assert row["reach_on_pct"] == 100.0
    assert row["reach_off_pct"] == 0.0
    assert row["delta_reach_pp"] == 100.0


def test_tag_on_off_tables_boolean_tag_has_no_exclusions():
    df = pd.DataFrame(
        {
            "stacked_imbalance": [True, True, False, False],
            "reach_1.0R": [True, False, True, False],
        }
    )
    on, off, cmp_ = tag_on_off_tables(
        df, "stacked_imbalance", on_value=True, off_value=False, targets=(1.0,)
    )
    row = cmp_.iloc[0]
    assert row["n_on"] == 2
    assert row["n_off"] == 2


def test_promotion_flags_requires_both_edge_and_sample_size():
    cmp_ = pd.DataFrame({"edge_on_pp": [6.0, 6.0, 4.0], "n_on": [250, 100, 250]})
    flags = promotion_flags(cmp_, min_edge_pp=5.0, min_n=200)
    assert flags.tolist() == [True, False, False]


def test_format_reach_table_handles_mixed_dtype_n_column():
    # n ends up as a float64 column (shares a dtype-inferred DataFrame with
    # reach_pct/diff_pp); formatting must not choke on that.
    df = pd.DataFrame({"reach_0.5R": [True, False, np.nan]})
    table = reach_vs_random_walk(df, targets=(0.5,))
    text = format_reach_table(table, "title")
    assert "0.5R" in text
    assert "2" in text  # n=2 after excluding the NaN
