"""Condition library registry + mutex pairs."""

from __future__ import annotations

from ht_backtest.conditions.funding import FUNDING_CONDITIONS
from ht_backtest.conditions.htf import HTF_CONDITIONS
from ht_backtest.conditions.prior_session import PRIOR_SESSION_CONDITIONS
from ht_backtest.conditions.range_ctx import RANGE_CONDITIONS
from ht_backtest.conditions.session import SESSION_CONDITIONS
from ht_backtest.conditions.volume import VOLUME_CONDITIONS

ALL_CONDITIONS = (
    *SESSION_CONDITIONS,
    *VOLUME_CONDITIONS,
    *RANGE_CONDITIONS,
    *HTF_CONDITIONS,
    *PRIOR_SESSION_CONDITIONS,
    *FUNDING_CONDITIONS,
)

CONDITION_BY_ID = {c.id: c for c in ALL_CONDITIONS}

# Unordered mutex pairs — grid skips any combo containing both sides.
MUTEX_PAIRS: tuple[tuple[str, str], ...] = (
    ("london_session", "ny_session"),
    ("london_session", "asia_session"),
    ("ny_session", "asia_session"),
    ("london_open_30m", "ny_open_30m"),
    ("volume_high", "volume_low"),
    ("asia_range_tight", "asia_range_wide"),
    ("price_above_asia_mid", "price_below_asia_mid"),
    ("above_4h_ema20", "below_4h_ema20"),
    ("4h_hh_hl", "4h_lh_ll"),
    ("funding_positive", "funding_negative"),
    ("london_raided_high", "london_raided_low"),
    ("prior_session_same_direction", "prior_session_opposite_direction"),
)


def is_mutex_combo(condition_ids: list[str] | tuple[str, ...] | set[str]) -> bool:
    s = set(condition_ids)
    for a, b in MUTEX_PAIRS:
        if a in s and b in s:
            return True
    return False


def library_version() -> str:
    parts = [f"{c.id}:v{getattr(c, 'version', 1)}" for c in ALL_CONDITIONS]
    return "|".join(parts)
