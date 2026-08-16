"""Atomic market conditions: True / False / None per bar (no lookahead)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Condition(Protocol):
    id: str
    category: str

    def eval(self, bars: pd.DataFrame) -> pd.Series:
        """Return boolean/object series aligned to bars: True | False | None."""
        ...


def as_object_bool(mask: pd.Series, defined: pd.Series) -> pd.Series:
    """Where defined is False → None; else True/False from mask."""
    out = np.full(len(mask), None, dtype=object)
    ok = defined.fillna(False).to_numpy(dtype=bool)
    out[ok] = mask.to_numpy()[ok].astype(bool)
    return pd.Series(out, index=mask.index)
