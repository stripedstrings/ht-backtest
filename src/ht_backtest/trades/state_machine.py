"""The three trade gates, in order: session-range liquidity raid (from
gates.session_range) -> MSS through the protected internal swing -> FVG
retest. Nothing else may veto a setup; entry/stop/target match the Pine
defaults exactly (FVG edge retest, stop beyond the sweep extreme, target at
the far session-range edge).

This is a direct bar-by-bar port of Pine's beSt/buSt state machine (states
0..4: idle -> waiting for close-back-inside -> waiting for MSS -> waiting for
retest -> live trade), not a vectorized approximation -- the state depends on
per-bar history (running highs/lows since the grab, bars-since counters) in
a way that only a sequential port can stay faithful to.

Pine runs exactly ONE bear (short) and ONE bull (long) state machine at a
time, system-wide: a new grab is ignored while a setup of that same
direction is already in progress (states 1-4). Bear and bull run
independently of each other.

Alongside the three gates, this also captures the RAW values every
deliverable-5 tag is derived from (wick/body/coil in ATR units, imbalance
count, order-block presence, session/hour/daily-bias/premium-discount at
entry) at the exact bar Pine captures them -- grab bar, MSS bar, or entry
bar, per tag. The median-based tags themselves (wick/body/coil/age/range
above-or-below the running median) are NOT computed here: they depend on a
cross-trade running median that only makes sense once trades are assembled
in chronological order, so that lives in trades/tagging.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GateParams:
    sweep_win: int = 3
    wick_min: float = 0.0
    bos_use_close: bool = True
    mom_atr: float = 0.0
    max_bars_to_bos: int = 20
    retest_mode: str = "Edge"  # "Edge" | "50%" | "Full fill"
    max_bars_to_retest: int = 30
    stop_mode: str = "Sweep extreme"  # "Sweep extreme" | "FVG far side"
    target_mode: str = "Session range edge"  # "Session range edge" | "Opposing liquidity" | "Fixed R"
    fixed_r: float = 2.5
    max_hold: int = 200
    cost_pts: float = 0.0
    cons_bars: int = 4  # coil lookback, Pine's consBars
    ote_lo: float = 0.62
    ote_hi: float = 0.79


def _find_bear_fvg(low: np.ndarray, high: np.ndarray, i: int, rng_lo: float, rng: float, leg_len: int):
    """Bearish FVG search inside the down-leg ending at bar i: for offset j
    (0=most recent), t=low[i-(j+2)] (older candle) vs b=high[i-j] (newer
    candle); a gap exists where t > b. Keeps the gap with the largest
    retracement depth d = (midpoint - rng_lo) / rng (closest to the swept
    high), and counts every valid gap found (Pine's nF, "stacked imbalance")."""
    best_d, best_t, best_b, best_j = None, None, None, None
    n_found = 0
    if rng <= 0:
        return None, None, None, 0, None
    for j in range(0, leg_len + 1):
        t_idx, b_idx = i - (j + 2), i - j
        if t_idx < 0 or b_idx < 0:
            continue
        t, b = low[t_idx], high[b_idx]
        if t > b:
            n_found += 1
            m = (t + b) / 2
            d = (m - rng_lo) / rng
            if d <= 1.0 and (best_d is None or d > best_d):
                best_d, best_t, best_b, best_j = d, t, b, j
    return best_t, best_b, best_d, n_found, best_j


def _find_bull_fvg(low: np.ndarray, high: np.ndarray, i: int, rng_hi: float, rng: float, leg_len: int):
    """Bullish FVG search inside the up-leg ending at bar i: for offset j,
    b=high[i-(j+2)] (older candle) vs t=low[i-j] (newer candle); a gap exists
    where t > b. Keeps the gap with the largest retracement depth
    d = (rng_hi - midpoint) / rng (closest to the swept low)."""
    best_d, best_t, best_b, best_j = None, None, None, None
    n_found = 0
    if rng <= 0:
        return None, None, None, 0, None
    for j in range(0, leg_len + 1):
        b_idx, t_idx = i - (j + 2), i - j
        if t_idx < 0 or b_idx < 0:
            continue
        b, t = high[b_idx], low[t_idx]
        if t > b:
            n_found += 1
            m = (t + b) / 2
            d = (rng_hi - m) / rng
            if d <= 1.0 and (best_d is None or d > best_d):
                best_d, best_t, best_b, best_j = d, t, b, j
    return best_t, best_b, best_d, n_found, best_j


def _coil_ratio(low: np.ndarray, high: np.ndarray, i: int, j: int, atr_i: float, cons_bars: int) -> float:
    """Pine's f_coilRatio(j): range of `cons_bars` candles starting 3 bars
    further back than offset j, as a fraction of ATR -- how tight price was
    coiled just before the FVG's older candle."""
    start = i - (j + 3)
    end = start - (cons_bars - 1)
    if end < 0 or np.isnan(atr_i) or atr_i == 0:
        return np.nan
    idx = range(end, start + 1)
    hh = max(high[k] for k in idx)
    ll = min(low[k] for k in idx)
    return (hh - ll) / atr_i


def _bear_ob_found(open_: np.ndarray, close: np.ndarray, atr_v: np.ndarray, i: int, leg_len: int) -> bool:
    """Last bullish candle immediately before a strong bearish candle that
    closes below its open -- scans from the most recent pair backward,
    takes the first match (Pine's `for j=1 to legLen ... break`)."""
    for j in range(1, leg_len + 1):
        older, newer = i - j, i - (j - 1)
        if older < 0 or newer < 0:
            continue
        if (
            close[older] > open_[older]
            and close[newer] < open_[newer]
            and close[newer] < open_[older]
            and abs(close[newer] - open_[newer]) >= 0.6 * atr_v[i]
        ):
            return True
    return False


def _bull_ob_found(open_: np.ndarray, close: np.ndarray, atr_v: np.ndarray, i: int, leg_len: int) -> bool:
    for j in range(1, leg_len + 1):
        older, newer = i - j, i - (j - 1)
        if older < 0 or newer < 0:
            continue
        if (
            close[older] < open_[older]
            and close[newer] > open_[newer]
            and close[newer] > open_[older]
            and abs(close[newer] - open_[newer]) >= 0.6 * atr_v[i]
        ):
            return True
    return False


def run_trade_state_machine(
    df: pd.DataFrame,
    atr: pd.Series,
    last_int_hi: pd.Series,
    last_int_lo: pd.Series,
    grab_up: pd.Series,
    grab_dn: pd.Series,
    grab_up_price: pd.Series,
    grab_dn_price: pd.Series,
    grab_up_type: pd.Series,
    grab_dn_type: pd.Series,
    grab_up_age: pd.Series,
    grab_dn_age: pd.Series,
    kz_hi: pd.Series,
    kz_lo: pd.Series,
    range_width_atr: pd.Series,
    efficiency_ratio: pd.Series,
    grab_seq: pd.Series,
    sessions: pd.DataFrame,
    daily_bias: pd.DataFrame,
    params: GateParams = GateParams(),
) -> list[dict]:
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    atr_v = atr.to_numpy()
    body_v = np.abs(close - open_)
    with np.errstate(invalid="ignore"):
        disp = body_v >= params.mom_atr * atr_v
    disp = np.where(np.isnan(atr_v), False, disp)

    last_int_hi_v = last_int_hi.to_numpy()
    last_int_lo_v = last_int_lo.to_numpy()
    grab_up_v, grab_dn_v = grab_up.to_numpy(), grab_dn.to_numpy()
    grab_up_price_v, grab_dn_price_v = grab_up_price.to_numpy(), grab_dn_price.to_numpy()
    grab_up_type_v, grab_dn_type_v = grab_up_type.to_numpy(), grab_dn_type.to_numpy()
    grab_up_age_v, grab_dn_age_v = grab_up_age.to_numpy(), grab_dn_age.to_numpy()
    kz_hi_v, kz_lo_v = kz_hi.to_numpy(), kz_lo.to_numpy()
    rng_width_v = range_width_atr.to_numpy()
    er_v = efficiency_ratio.to_numpy()
    grab_seq_v = grab_seq.to_numpy()
    in_ny = sessions["in_ny"].to_numpy()
    in_london = sessions["in_london"].to_numpy()
    in_asia = sessions["in_asia"].to_numpy()
    hour_london = sessions["hour_london"].to_numpy()
    bias_bull_v = daily_bias["bias_bull"].to_numpy()
    bias_bear_v = daily_bias["bias_bear"].to_numpy()
    premium_v = daily_bias["premium"].to_numpy()

    def _session_label(i: int) -> str:
        if in_ny[i]:
            return "NY"
        if in_london[i]:
            return "LONDON"
        if in_asia[i]:
            return "ASIA"
        return "OTHER"

    trades: list[dict] = []
    be_st = 0
    bu_st = 0
    be: dict = {}
    bu: dict = {}

    for i in range(n):
        # ---- BEAR (short), armed by a raid of the session-range HIGH ----
        if be_st == 0:
            if grab_up_v[i] and not np.isnan(last_int_lo_v[i]):
                be = dict(
                    lvl=grab_up_price_v[i],
                    ty=int(grab_up_type_v[i]),
                    age=int(grab_up_age_v[i]),
                    hi=high[i],
                    body=max(open_[i], close[i]),
                    edge=kz_lo_v[i],
                    grab_bar=i,
                    range_width_atr=rng_width_v[i],
                    efficiency_ratio=er_v[i - 1] if i >= 1 else np.nan,
                    grab_seq=int(grab_seq_v[i]),
                    cnt=0,
                )
                be_st = 1

        elif be_st == 1:
            if high[i] > be["hi"]:
                be["hi"] = high[i]
                be["body"] = max(open_[i], close[i])
            be["cnt"] += 1
            if close[i] < be["lvl"]:
                wick = be["hi"] - be["body"]
                if wick < params.wick_min * atr_v[i]:
                    be_st = 0
                else:
                    be["wick_atr"] = wick / atr_v[i] if atr_v[i] else np.nan
                    be["prot"] = last_int_lo_v[i]
                    be["run_lo"] = low[i]
                    be["bar"] = i
                    be["close_back_bar"] = i
                    be["cnt"] = 0
                    be_st = 0 if np.isnan(be["prot"]) else 2
            elif be["cnt"] > params.sweep_win:
                be_st = 0

        elif be_st == 2:
            be["run_lo"] = min(be["run_lo"], low[i])
            be["cnt"] += 1
            bos_hit = (close[i] < be["prot"]) if params.bos_use_close else (low[i] < be["prot"])
            if bos_hit and disp[i]:
                be["rng_hi"] = be["hi"]
                be["rng_lo"] = be["run_lo"]
                rng = be["rng_hi"] - be["rng_lo"]
                leg_len = min(i - be["bar"] + 2, 60)
                t, b, d, n_found, j = _find_bear_fvg(low, high, i, be["rng_lo"], rng, leg_len)
                if t is None:
                    be_st = 0
                else:
                    be["fvg_t"], be["fvg_b"], be["ret"] = t, b, d
                    be["mss_bar"] = i
                    be["body_atr"] = body_v[i] / atr_v[i] if atr_v[i] else np.nan
                    be["imbalance_count"] = n_found
                    be["ob_found"] = _bear_ob_found(open_, close, atr_v, i, leg_len)
                    be["coil_atr"] = _coil_ratio(low, high, i, j, atr_v[i], params.cons_bars)
                    be["cnt"] = 0
                    be_st = 3
            elif be["cnt"] > params.max_bars_to_bos:
                be_st = 0

        elif be_st == 3:
            be["cnt"] += 1
            lvl = {"Edge": be["fvg_b"], "50%": (be["fvg_t"] + be["fvg_b"]) / 2, "Full fill": be["fvg_t"]}[
                params.retest_mode
            ]
            if high[i] >= lvl:
                e = lvl
                s = be["rng_hi"] if params.stop_mode == "Sweep extreme" else be["fvg_t"]
                if params.target_mode == "Session range edge" and not np.isnan(be["edge"]):
                    g = be["edge"]
                elif params.target_mode == "Opposing liquidity":
                    g = be["rng_lo"]
                else:
                    g = e - params.fixed_r * (s - e)
                if s > e > g:
                    be["ent"], be["stp"], be["tgt"] = e, s, g
                    be["rr"] = (e - g) / (s - e)
                    be["entry_bar"] = i
                    be["session"] = _session_label(i)
                    be["hour_london"] = int(hour_london[i])
                    be["with_daily_bias"] = bool(bias_bear_v[i])
                    be["right_premium_discount"] = bool(premium_v[i])  # premium favors shorts
                    be["ote"] = bool(be["ret"] is not None and params.ote_lo <= be["ret"] < params.ote_hi)
                    be["cnt"] = 0
                    be_st = 4
                else:
                    be_st = 0
            elif be["cnt"] > params.max_bars_to_retest:
                be_st = 0

        elif be_st == 4:
            be["cnt"] += 1
            risk = be["stp"] - be["ent"]
            if high[i] >= be["stp"]:
                r = -1.0 - params.cost_pts / risk
                trades.append(_finish_trade(df, be, "short", i, "stop", r, risk))
                be_st = 0
            elif low[i] <= be["tgt"]:
                r = (be["ent"] - be["tgt"]) / risk - params.cost_pts / risk
                trades.append(_finish_trade(df, be, "short", i, "target", r, risk))
                be_st = 0
            elif be["cnt"] > params.max_hold:
                r = (be["ent"] - close[i]) / risk - params.cost_pts / risk
                trades.append(_finish_trade(df, be, "short", i, "timeout", r, risk))
                be_st = 0

        # ---- BULL (long), armed by a raid of the session-range LOW ----
        if bu_st == 0:
            if grab_dn_v[i] and not np.isnan(last_int_hi_v[i]):
                bu = dict(
                    lvl=grab_dn_price_v[i],
                    ty=int(grab_dn_type_v[i]),
                    age=int(grab_dn_age_v[i]),
                    lo=low[i],
                    body=min(open_[i], close[i]),
                    edge=kz_hi_v[i],
                    grab_bar=i,
                    range_width_atr=rng_width_v[i],
                    efficiency_ratio=er_v[i - 1] if i >= 1 else np.nan,
                    grab_seq=int(grab_seq_v[i]),
                    cnt=0,
                )
                bu_st = 1

        elif bu_st == 1:
            if low[i] < bu["lo"]:
                bu["lo"] = low[i]
                bu["body"] = min(open_[i], close[i])
            bu["cnt"] += 1
            if close[i] > bu["lvl"]:
                wick = bu["body"] - bu["lo"]
                if wick < params.wick_min * atr_v[i]:
                    bu_st = 0
                else:
                    bu["wick_atr"] = wick / atr_v[i] if atr_v[i] else np.nan
                    bu["prot"] = last_int_hi_v[i]
                    bu["run_hi"] = high[i]
                    bu["bar"] = i
                    bu["close_back_bar"] = i
                    bu["cnt"] = 0
                    bu_st = 0 if np.isnan(bu["prot"]) else 2
            elif bu["cnt"] > params.sweep_win:
                bu_st = 0

        elif bu_st == 2:
            bu["run_hi"] = max(bu["run_hi"], high[i])
            bu["cnt"] += 1
            bos_hit = (close[i] > bu["prot"]) if params.bos_use_close else (high[i] > bu["prot"])
            if bos_hit and disp[i]:
                bu["rng_lo"] = bu["lo"]
                bu["rng_hi"] = bu["run_hi"]
                rng = bu["rng_hi"] - bu["rng_lo"]
                leg_len = min(i - bu["bar"] + 2, 60)
                t, b, d, n_found, j = _find_bull_fvg(low, high, i, bu["rng_hi"], rng, leg_len)
                if t is None:
                    bu_st = 0
                else:
                    bu["fvg_t"], bu["fvg_b"], bu["ret"] = t, b, d
                    bu["mss_bar"] = i
                    bu["body_atr"] = body_v[i] / atr_v[i] if atr_v[i] else np.nan
                    bu["imbalance_count"] = n_found
                    bu["ob_found"] = _bull_ob_found(open_, close, atr_v, i, leg_len)
                    bu["coil_atr"] = _coil_ratio(low, high, i, j, atr_v[i], params.cons_bars)
                    bu["cnt"] = 0
                    bu_st = 3
            elif bu["cnt"] > params.max_bars_to_bos:
                bu_st = 0

        elif bu_st == 3:
            bu["cnt"] += 1
            lvl = {"Edge": bu["fvg_t"], "50%": (bu["fvg_t"] + bu["fvg_b"]) / 2, "Full fill": bu["fvg_b"]}[
                params.retest_mode
            ]
            if low[i] <= lvl:
                e = lvl
                s = bu["rng_lo"] if params.stop_mode == "Sweep extreme" else bu["fvg_b"]
                if params.target_mode == "Session range edge" and not np.isnan(bu["edge"]):
                    g = bu["edge"]
                elif params.target_mode == "Opposing liquidity":
                    g = bu["rng_hi"]
                else:
                    g = e + params.fixed_r * (e - s)
                if s < e < g:
                    bu["ent"], bu["stp"], bu["tgt"] = e, s, g
                    bu["rr"] = (g - e) / (e - s)
                    bu["entry_bar"] = i
                    bu["session"] = _session_label(i)
                    bu["hour_london"] = int(hour_london[i])
                    bu["with_daily_bias"] = bool(bias_bull_v[i])
                    bu["right_premium_discount"] = bool(not premium_v[i])  # discount favors longs
                    bu["ote"] = bool(bu["ret"] is not None and params.ote_lo <= bu["ret"] < params.ote_hi)
                    bu["cnt"] = 0
                    bu_st = 4
                else:
                    bu_st = 0
            elif bu["cnt"] > params.max_bars_to_retest:
                bu_st = 0

        elif bu_st == 4:
            bu["cnt"] += 1
            risk = bu["ent"] - bu["stp"]
            if low[i] <= bu["stp"]:
                r = -1.0 - params.cost_pts / risk
                trades.append(_finish_trade(df, bu, "long", i, "stop", r, risk))
                bu_st = 0
            elif high[i] >= bu["tgt"]:
                r = (bu["tgt"] - bu["ent"]) / risk - params.cost_pts / risk
                trades.append(_finish_trade(df, bu, "long", i, "target", r, risk))
                bu_st = 0
            elif bu["cnt"] > params.max_hold:
                r = (close[i] - bu["ent"]) / risk - params.cost_pts / risk
                trades.append(_finish_trade(df, bu, "long", i, "timeout", r, risk))
                bu_st = 0

    return trades


def _finish_trade(df: pd.DataFrame, state: dict, direction: str, exit_bar: int, exit_reason: str, r: float, risk: float) -> dict:
    ts = df["timestamp"]
    return {
        "direction": direction,
        "grab_bar": state["grab_bar"],
        "grab_time": ts.iloc[state["grab_bar"]],
        "grab_price": state["lvl"],
        "grab_type": state["ty"],
        "grab_age_bars": state["age"],
        "range_width_atr": state["range_width_atr"],
        "efficiency_ratio": state["efficiency_ratio"],
        "grab_seq": state["grab_seq"],
        "close_back_bar": state["close_back_bar"],
        "wick_atr": state["wick_atr"],
        "mss_bar": state["mss_bar"],
        "mss_time": ts.iloc[state["mss_bar"]],
        "protected_level": state["prot"],
        "leg_high": state["rng_hi"],
        "leg_low": state["rng_lo"],
        "body_atr": state["body_atr"],
        "imbalance_count": state["imbalance_count"],
        "stacked_imbalance": state["imbalance_count"] > 1,
        "valid_ob": state["ob_found"],
        "coil_atr": state["coil_atr"],
        "fvg_top": state["fvg_t"],
        "fvg_bottom": state["fvg_b"],
        "fvg_retrace_depth": state["ret"],
        "ote": state["ote"],
        "entry_bar": state["entry_bar"],
        "entry_time": ts.iloc[state["entry_bar"]],
        "entry_price": state["ent"],
        "stop_price": state["stp"],
        "target_price": state["tgt"],
        "risk": risk,
        "planned_rr": state["rr"],
        "session_edge_target": state["edge"],
        "session": state["session"],
        "hour_london": state["hour_london"],
        "with_daily_bias": state["with_daily_bias"],
        "right_premium_discount": state["right_premium_discount"],
        "first_grab_of_day": state["grab_seq"] <= 1,
        "exit_bar": exit_bar,
        "exit_time": ts.iloc[exit_bar],
        "exit_reason": exit_reason,
        "r_multiple": r,
    }
