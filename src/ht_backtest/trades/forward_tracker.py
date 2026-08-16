"""Forward reach tracker: independent of the trade's own stop/target/timeout
resolution, follow every entry for mfe_win bars and ask, for each target in
0.5R..3R, "was this reached before the 1R adverse stop?" The 1R adverse
level is the SAME distance as the trade's own risk (entry-to-stop), so this
answers a different question than the live trade's outcome -- e.g. a trade
whose own planned target is only 0.7R can still be asked whether price would
have reached 2R if you'd let it run past that target.

INTRABAR AMBIGUITY (confirmed with the user): OHLC bars don't reveal whether
the stop or a further favorable excursion happened first when a single bar's
range would touch both. This tracker resolves that conservatively: if the
1R adverse stop is touched on a bar, that bar's OWN favorable excursion is
NOT used to credit any new target reach (only the running max-favorable from
STRICTLY EARLIER bars counts) -- and the bar is flagged, not dropped, so it
stays inspectable rather than silently biasing the reach rate optimistic.

If a trade runs off the end of the available data before either the 1R stop
or mfe_win bars are reached, that trade is marked "insufficient_data" rather
than a genuine timeout -- but any target already reached within the bars
that DID exist is still credited (real signal, not discarded), just not the
targets that remain unknown.

Phase C: trade loop uses NumPy column arrays (no DataFrame.iterrows).
Per-trade bar scan is unchanged so reach outcomes stay bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TARGETS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def track_forward_reach(
    trades_df: pd.DataFrame,
    df: pd.DataFrame,
    mfe_win: int = 100,
    targets: tuple[float, ...] = DEFAULT_TARGETS,
) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    high = df["high"].to_numpy(dtype=np.float64, copy=False)
    low = df["low"].to_numpy(dtype=np.float64, copy=False)
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    n_bars = len(df)

    directions = trades_df["direction"].to_numpy()
    entry_bars = trades_df["entry_bar"].to_numpy(dtype=np.int64, copy=False)
    entry_prices = trades_df["entry_price"].to_numpy(dtype=np.float64, copy=False)
    risks = trades_df["risk"].to_numpy(dtype=np.float64, copy=False)
    n_trades = len(trades_df)

    resolutions = np.empty(n_trades, dtype=object)
    mfes = np.empty(n_trades, dtype=np.float64)
    end_rs = np.empty(n_trades, dtype=np.float64)
    ambiguous_flags = np.empty(n_trades, dtype=bool)
    ambiguous_bars = np.empty(n_trades, dtype=object)
    reach = {T: np.empty(n_trades, dtype=object) for T in targets}

    for t in range(n_trades):
        direction = directions[t]
        entry_bar = int(entry_bars[t])
        entry_price = float(entry_prices[t])
        risk = float(risks[t])

        bars_available = n_bars - 1 - entry_bar
        window = min(mfe_win, bars_available)

        running_mfe = 0.0
        ambiguous = False
        ambiguous_bar = None
        resolution = None
        end_r = np.nan

        if window <= 0:
            resolution = "insufficient_data"
        else:
            stopped = False
            is_short = direction == "short"
            for k in range(1, window + 1):
                i = entry_bar + k
                if is_short:
                    fav = (entry_price - low[i]) / risk
                    adv = (high[i] - entry_price) / risk
                else:
                    fav = (high[i] - entry_price) / risk
                    adv = (entry_price - low[i]) / risk

                if adv >= 1.0:
                    if fav > running_mfe:
                        ambiguous = True
                        ambiguous_bar = i
                    resolution = "stop"
                    stopped = True
                    break
                if fav > running_mfe:
                    running_mfe = fav

            if not stopped:
                final_i = entry_bar + window
                close_final = close[final_i]
                end_r = (
                    (entry_price - close_final) / risk if is_short else (close_final - entry_price) / risk
                )
                resolution = "timeout" if window == mfe_win else "insufficient_data"

        resolutions[t] = resolution
        mfes[t] = running_mfe
        end_rs[t] = end_r
        ambiguous_flags[t] = ambiguous
        ambiguous_bars[t] = ambiguous_bar
        for T in targets:
            if running_mfe >= T:
                reach[T][t] = True
            elif resolution == "insufficient_data":
                reach[T][t] = np.nan
            else:
                reach[T][t] = False

    out = trades_df.copy()
    out["shadow_resolution"] = resolutions
    out["shadow_mfe_r"] = mfes
    out["shadow_end_r"] = end_rs
    out["intrabar_ambiguous"] = ambiguous_flags
    out["intrabar_ambiguous_bar"] = ambiguous_bars
    for T in targets:
        out[f"reach_{T}R"] = reach[T]
    return out
