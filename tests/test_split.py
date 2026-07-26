import random

import pandas as pd

from ht_backtest.data.split import SplitManifest


def _manifest():
    return SplitManifest(
        seed=42,
        timeframe="15m",
        symbol_holdout_fraction=0.3,
        date_holdout_fraction=0.2,
        universe=["A", "B", "C", "D"],
        holdout_symbols=["A"],
        train_symbols=["B", "C", "D"],
        overall_start_ms=0,
        overall_end_ms=1_000_000,
        date_holdout_start_ms=800_000,
    )


def test_holdout_symbol_fully_held_out_regardless_of_date():
    m = _manifest()
    assert m.classify("A", 0) == "holdout"
    assert m.classify("A", 999_999) == "holdout"


def test_train_symbol_before_cutoff_is_train():
    m = _manifest()
    assert m.classify("B", 0) == "train"
    assert m.classify("B", 799_999) == "train"


def test_train_symbol_after_cutoff_is_holdout():
    m = _manifest()
    assert m.classify("B", 800_000) == "holdout"
    assert m.classify("B", 999_999) == "holdout"


def test_unknown_symbol_raises():
    m = _manifest()
    try:
        m.classify("Z", 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_split_frame_matches_classify():
    m = _manifest()
    df = pd.DataFrame({"timestamp": [0, 799_999, 800_000, 999_999]})
    out = m.split_frame("B", df)
    assert list(out["split"]) == ["train", "train", "holdout", "holdout"]

    out_a = m.split_frame("A", df)
    assert list(out_a["split"]) == ["holdout", "holdout", "holdout", "holdout"]


def test_save_load_roundtrip(tmp_path):
    m = _manifest()
    path = tmp_path / "split.json"
    m.save(path)
    loaded = SplitManifest.load(path)
    assert loaded.holdout_symbols == m.holdout_symbols
    assert loaded.train_symbols == m.train_symbols
    assert loaded.date_holdout_start_ms == m.date_holdout_start_ms
    assert loaded.seed == m.seed


def test_symbol_sampling_is_reproducible_with_fixed_seed():
    symbols = [f"SYM{i}" for i in range(30)]
    a = random.Random(20260726).sample(sorted(symbols), k=9)
    b = random.Random(20260726).sample(sorted(symbols), k=9)
    assert a == b
    assert len(set(a)) == 9
