"""Pine-equivalent primitives: ATR, pool/protected-swing fractals, session
tagging, daily prev-high/low, and daily bias pivots.

TWO DELIBERATELY DIFFERENT CLOCKS -- confirmed with the user 2026-07-26, do
not unify them:
  - UTC is used for every DAY-boundary concept: PDH/PDL, daily bias pivots,
    daily-timeframe candle aggregation, first-grab-of-day sequence reset.
  - Europe/London is used for every SESSION/killzone concept: the London,
    NY and Asia session windows and the session-range gate itself.
The Pine source only pins session `input.session()` calls to Europe/London
explicitly; day-boundary constructs (`timeframe.change("D")`,
`request.security(..., "D", ...)`) inherit the chart's timezone, which for
crypto data conventionally means UTC (it matches Binance's own daily candle
boundaries). Both clocks are real and intentional -- a bar can be in the
London killzone while still belonging to "yesterday" in UTC terms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LONDON_TZ = "Europe/London"


def efficiency_ratio(df: pd.DataFrame, ctx_bars: int = 40) -> pd.Series:
    """Pine's `erV`: |net move over ctx_bars| / sum(|bar-to-bar move|) over
    the same window. Low = price went nowhere (a range existed to spring
    from); high = a trend (the "grab" was just a breakdown continuing)."""
    close = df["close"]
    er_num = (close - close.shift(ctx_bars)).abs()
    er_den = close.diff().abs().rolling(ctx_bars).sum()
    return (er_num / er_den.replace(0, np.nan))


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder RMA of true range, matching Pine's ta.atr(length)."""
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = hi.iloc[0] - lo.iloc[0]
    alpha = 1.0 / length
    return tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


def _rolling_side_max(values: np.ndarray, n: int, direction: int) -> np.ndarray:
    """direction=+1: max of the n values AFTER each position (exclusive).
    direction=-1: max of the n values BEFORE each position (exclusive).
    Positions without a full window of n values on that side get NaN."""
    s = pd.Series(values)
    if direction == -1:
        return s.shift(1).rolling(n).max().to_numpy()
    reversed_max = s[::-1].shift(1).rolling(n).max()
    return reversed_max[::-1].to_numpy()


def fractal_high(high: pd.Series, n: int) -> pd.Series:
    """True at bar j if high[j] is strictly greater than the n bars on
    each side (Pine f_upFrac's candidate bar, before the n-bar confirmation
    delay is applied)."""
    v = high.to_numpy(dtype=float)
    left_max = _rolling_side_max(v, n, direction=-1)
    right_max = _rolling_side_max(v, n, direction=1)
    ok = (v > left_max) & (v > right_max)
    return pd.Series(ok, index=high.index).fillna(False)


def fractal_low(low: pd.Series, n: int) -> pd.Series:
    """True at bar j if low[j] is strictly less than the n bars on each side."""
    v = low.to_numpy(dtype=float)
    left_min = -_rolling_side_max(-v, n, direction=-1)
    right_min = -_rolling_side_max(-v, n, direction=1)
    ok = (v < left_min) & (v < right_min)
    return pd.Series(ok, index=low.index).fillna(False)


def confirmed_fractal_high(high: pd.Series, n: int) -> pd.Series:
    """Pine's `poolUp`/`intUp`: True at the CURRENT bar once the fractal
    n bars back is confirmed. Safe to consume at its own index -- no lookahead."""
    return fractal_high(high, n).shift(n, fill_value=False)


def confirmed_fractal_low(low: pd.Series, n: int) -> pd.Series:
    """Pine's `poolDn`/`intDn`."""
    return fractal_low(low, n).shift(n, fill_value=False)


def protected_swing_levels(df: pd.DataFrame, int_len: int) -> tuple[pd.Series, pd.Series]:
    """Pine's `lastIntHi`/`lastIntLo`: var float that persists (ffill) at the
    price of the most recently confirmed protected-swing fractal."""
    int_up = confirmed_fractal_high(df["high"], int_len)
    int_dn = confirmed_fractal_low(df["low"], int_len)
    hi_price_at_confirm = df["high"].shift(int_len)
    lo_price_at_confirm = df["low"].shift(int_len)
    last_int_hi = pd.Series(np.where(int_up, hi_price_at_confirm, np.nan), index=df.index).ffill()
    last_int_lo = pd.Series(np.where(int_dn, lo_price_at_confirm, np.nan), index=df.index).ffill()
    return last_int_hi, last_int_lo


def session_tags(timestamp_ms: pd.Series) -> pd.DataFrame:
    """Europe/London session membership: London 0700-1000, NY 1230-1500
    (both London-local), Asia 0000-0600. Half-open [start, end)."""
    local = pd.to_datetime(timestamp_ms, unit="ms", utc=True).dt.tz_convert(LONDON_TZ)
    minute_of_day = local.dt.hour * 60 + local.dt.minute

    def _in_session(start_hhmm: str, end_hhmm: str) -> pd.Series:
        sh, sm = int(start_hhmm[:2]), int(start_hhmm[2:])
        eh, em = int(end_hhmm[:2]), int(end_hhmm[2:])
        start, end = sh * 60 + sm, eh * 60 + em
        return (minute_of_day >= start) & (minute_of_day < end)

    in_lon = _in_session("0700", "1000")
    in_ny = _in_session("1230", "1500")
    in_asia = _in_session("0000", "0600")

    out = pd.DataFrame(
        {
            "in_london": in_lon,
            "in_ny": in_ny,
            "in_asia": in_asia,
            "in_killzone": in_lon | in_ny,
        },
        index=timestamp_ms.index,
    )
    out["london_start"] = out["in_london"] & ~out["in_london"].shift(1, fill_value=False)
    out["ny_start"] = out["in_ny"] & ~out["in_ny"].shift(1, fill_value=False)
    out["asia_start"] = out["in_asia"] & ~out["in_asia"].shift(1, fill_value=False)
    out["asia_end"] = ~out["in_asia"] & out["in_asia"].shift(1, fill_value=False)
    out["killzone_start"] = out["in_killzone"] & ~out["in_killzone"].shift(1, fill_value=False)
    out["hour_london"] = local.dt.hour
    return out


def utc_day_tags(timestamp_ms: pd.Series) -> pd.DataFrame:
    """Day-boundary concepts, deliberately in UTC (not Europe/London)."""
    utc = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
    date = utc.dt.date
    new_day = pd.Series(date, index=timestamp_ms.index) != pd.Series(date, index=timestamp_ms.index).shift(1)
    new_day.iloc[0] = True
    return pd.DataFrame({"utc_date": date, "new_day": new_day}, index=timestamp_ms.index)


def daily_prev_high_low(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Pine's `[pdh, pdl] = request.security(..., "D", [high[1], low[1]])`:
    the FULLY CLOSED previous UTC day's high/low, available from the first
    intraday bar of the following UTC day onward. Zero lookahead: by the time
    "today" (UTC) starts, "yesterday" (UTC) is already closed."""
    utc_date = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    daily = df.groupby(utc_date).agg(high=("high", "max"), low=("low", "min"))
    daily_prev = daily.shift(1)
    pdh = utc_date.map(daily_prev["high"])
    pdl = utc_date.map(daily_prev["low"])
    pdh.index = df.index
    pdl.index = df.index
    return pdh, pdl


def pool_swing_events(
    df: pd.DataFrame, atr: pd.Series, sw_len: int = 5, eq_tol_mult: float = 0.10
) -> pd.DataFrame:
    """Pine's swing-liquidity pool creation (poolUp/poolDn) plus EQH/EQL vs
    BSL/SSL tagging: a new pool is tagged equal-high/low if it lands within
    eq_tol_mult * ATR of the immediately preceding same-side pool."""
    pool_up = confirmed_fractal_high(df["high"], sw_len)
    pool_dn = confirmed_fractal_low(df["low"], sw_len)
    hi_price = df["high"].shift(sw_len)
    lo_price = df["low"].shift(sw_len)

    last_pool_hi = pd.Series(np.nan, index=df.index)
    last_pool_lo = pd.Series(np.nan, index=df.index)
    is_eqh = pd.Series(False, index=df.index)
    is_eql = pd.Series(False, index=df.index)

    prev_hi, prev_lo = np.nan, np.nan
    hi_vals = hi_price.to_numpy()
    lo_vals = lo_price.to_numpy()
    atr_vals = atr.to_numpy()
    up_vals = pool_up.to_numpy()
    dn_vals = pool_dn.to_numpy()
    eqh_out = np.zeros(len(df), dtype=bool)
    eql_out = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if up_vals[i]:
            px = hi_vals[i]
            if not np.isnan(prev_hi) and not np.isnan(atr_vals[i]) and abs(px - prev_hi) <= eq_tol_mult * atr_vals[i]:
                eqh_out[i] = True
            prev_hi = px
        if dn_vals[i]:
            px = lo_vals[i]
            if not np.isnan(prev_lo) and not np.isnan(atr_vals[i]) and abs(px - prev_lo) <= eq_tol_mult * atr_vals[i]:
                eql_out[i] = True
            prev_lo = px

    return pd.DataFrame(
        {
            "pool_up": pool_up,
            "pool_dn": pool_dn,
            "pool_up_price": hi_price,
            "pool_dn_price": lo_price,
            "is_eqh": pd.Series(eqh_out, index=df.index),
            "is_eql": pd.Series(eql_out, index=df.index),
        }
    )


def asia_range_pools(df: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Pine's Asia-session accumulation: running max(high)/min(low) from
    asia_start to asia_end (Europe/London clock), frozen as a pool at
    asia_end. Emits one event row per completed Asia session."""
    in_asia = sessions["in_asia"].to_numpy()
    asia_start = sessions["asia_start"].to_numpy()
    asia_end = sessions["asia_end"].to_numpy()
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()

    asia_hi_out = np.full(len(df), np.nan)
    asia_lo_out = np.full(len(df), np.nan)
    cur_hi, cur_lo = np.nan, np.nan
    for i in range(len(df)):
        if in_asia[i]:
            cur_hi = hi[i] if asia_start[i] or np.isnan(cur_hi) else max(cur_hi, hi[i])
            cur_lo = lo[i] if asia_start[i] or np.isnan(cur_lo) else min(cur_lo, lo[i])
        if asia_end[i] and not np.isnan(cur_hi):
            asia_hi_out[i] = cur_hi
            asia_lo_out[i] = cur_lo
            cur_hi, cur_lo = np.nan, np.nan

    return pd.DataFrame(
        {"asia_pool_high": asia_hi_out, "asia_pool_low": asia_lo_out}, index=df.index
    )


def daily_bias_pivots(df: pd.DataFrame, d_piv: int = 2) -> pd.DataFrame:
    """Pine's daily bias block: pivot high/low (left=right=d_piv) on the UTC
    daily timeframe, tracking the last two confirmed pivots each side to
    derive structure (strBull/strBear) and the dealing-range equilibrium.
    A pivot at daily bar k is confirmed d_piv days later; the value is then
    held (var float semantics) until superseded, then broadcast back onto
    every intraday bar of that UTC day."""
    utc_date = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    daily = df.groupby(utc_date).agg(high=("high", "max"), low=("low", "min"))

    piv_high = fractal_high(daily["high"], d_piv)
    piv_low = fractal_low(daily["low"], d_piv)
    confirmed_high = piv_high.shift(d_piv, fill_value=False)
    confirmed_low = piv_low.shift(d_piv, fill_value=False)
    h_val_at_confirm = daily["high"].shift(d_piv)
    l_val_at_confirm = daily["low"].shift(d_piv)

    # H2/L2 must be the PREVIOUS confirmed pivot's value, not lag-1 of an
    # already-ffilled series (which would just repeat H1 on non-confirming
    # days) -- so shift at confirmation events only, then ffill.
    h_confirmed_vals = pd.Series(np.where(confirmed_high, h_val_at_confirm, np.nan), index=daily.index)
    h_confirmed_vals = h_confirmed_vals.dropna()
    h1_events = h_confirmed_vals.reindex(daily.index).ffill()
    h2_events = h_confirmed_vals.shift(1).reindex(daily.index).ffill()

    l_confirmed_vals = pd.Series(np.where(confirmed_low, l_val_at_confirm, np.nan), index=daily.index)
    l_confirmed_vals = l_confirmed_vals.dropna()
    l1_events = l_confirmed_vals.reindex(daily.index).ffill()
    l2_events = l_confirmed_vals.shift(1).reindex(daily.index).ffill()

    daily_out = pd.DataFrame(
        {
            "H1": h1_events,
            "H2": h2_events,
            "L1": l1_events,
            "L2": l2_events,
        }
    )
    daily_out["str_bull"] = (
        daily_out["H2"].notna()
        & daily_out["L2"].notna()
        & (daily_out["H1"] > daily_out["H2"])
        & (daily_out["L1"] > daily_out["L2"])
    )
    daily_out["str_bear"] = (
        daily_out["H2"].notna()
        & daily_out["L2"].notna()
        & (daily_out["H1"] < daily_out["H2"])
        & (daily_out["L1"] < daily_out["L2"])
    )
    daily_out["eq"] = (
        daily_out[["H1", "H2"]].max(axis=1).combine_first(daily["high"])
        + daily_out[["L1", "L2"]].min(axis=1).combine_first(daily["low"])
    ) / 2

    out = daily_out.reindex(utc_date).reset_index(drop=True)
    out.index = df.index
    # premium/discount and bias direction are evaluated against the CURRENT
    # bar's close (intraday), not the daily close -- so this step happens
    # after broadcasting the daily pivot state onto intraday bars.
    out["premium"] = df["close"].to_numpy() > out["eq"].to_numpy()
    # v0.7's fix: structure decides bias when it speaks (HH+HL / LH+LL);
    # otherwise the dealing range (premium/discount) does, so every trade
    # gets a direction instead of ~8% coverage from structure alone.
    out["bias_bull"] = out["str_bull"] | (~out["str_bear"].fillna(False) & ~out["premium"])
    out["bias_bear"] = ~out["bias_bull"]
    return out
