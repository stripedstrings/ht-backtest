"""Ten surviving hypothesis strategies (independent of HT).

IDs match the locked survivor list: 1, 3, 5, 6, 9, 11, 13, 16, 17, 18.
Strategy 18 is BTC sweeper / SOL holder / enter SOL (train-evaluable pair).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ht_backtest.strategies.base import StrategyContext, StrategyMetadata, TradeCandidate
from ht_backtest.strategies.hypothesis_helpers import (
    BTC,
    ETH,
    FAIL_WIN,
    RECLAIM_WIN,
    SOL,
    TARGET_R,
    VOL_LOOKBACK,
    asia_width_is_tight,
    collect_raid_events,
    ensure_primitives,
    ffill_asia_levels,
    first_fade_in_session,
    hash_params,
    make_fade_trade,
    make_reclaim_trade,
    reclaim_bar,
    session_range_frame,
    smt_divergences,
)

STRATEGY_VERSION = "1.0.0"


def _empty_tags(_trade: TradeCandidate, _bars: pd.DataFrame, _ctx: StrategyContext) -> Mapping[str, Any]:
    return {}


@dataclass
class KzFirstRaidReclaimStrategy:
    """Hypothesis 1 — Judas / first raid of the killzone."""

    reclaim_win: int = RECLAIM_WIN
    version: str = STRATEGY_VERSION
    theoretical_category: str = "timing"
    signal_type: str = "first_kz_raid_reclaim"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="kz_first_raid_reclaim",
            version=self.version,
            parameter_hash=hash_params({"id": "kz_first_raid_reclaim", "reclaim_win": self.reclaim_win}),
            description=(
                "Why it might beat a coin: ICT’s judas-swing claim — the first raid of a "
                "killzone is the real stop-run that clears resting liquidity; later raids "
                "are often continuation or noise. A reclaim after that first raid traps the "
                "traders who chased the sweep, so the move back through the edge has a "
                "defined pool of losers funding the reversal. "
                "Mechanics: in London or NY, take only the first session-range-edge raid of "
                f"that killzone instance; enter on close back inside within {self.reclaim_win} "
                "bars; stop beyond the sweep extreme; target the far session edge (or 2R)."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        for e in collect_raid_events(bars, prim, sr):
            if not e.first_of_kz or e.reclaim_bar is None:
                continue
            t = make_reclaim_trade(bars, ctx, sid, e)
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class KzFirst30mRaidReclaimStrategy:
    """Hypothesis 3 — first raid only if it fires in the opening 30 minutes."""

    reclaim_win: int = RECLAIM_WIN
    version: str = STRATEGY_VERSION
    theoretical_category: str = "timing"
    signal_type: str = "first_30m_raid_reclaim"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="kz_first_30m_raid_reclaim",
            version=self.version,
            parameter_hash=hash_params({"reclaim_win": self.reclaim_win, "window": "first_30m"}),
            description=(
                "Why it might beat a coin: the opening half-hour of London or NY is where "
                "inventory is thinnest and stops sit closest to overnight levels — the "
                "highest-manipulation window. If judas swings exist, they should concentrate "
                "here, not merely as ‘first raid sometime in the session.’ "
                "Mechanics: same first-raid-and-reclaim as kz_first_raid_reclaim, but only "
                "when the raid bar falls in London 07:00–07:30 or NY 12:30–13:00 Europe/London; "
                f"enter on reclaim within {self.reclaim_win} bars; stop beyond sweep extreme."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        for e in collect_raid_events(bars, prim, sr):
            if not e.first_of_kz or not e.first_30 or e.reclaim_bar is None:
                continue
            t = make_reclaim_trade(bars, ctx, sid, e)
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class HighVolGrabReclaimStrategy:
    """Hypothesis 5 — high-volume grab as absorption."""

    reclaim_win: int = RECLAIM_WIN
    vol_lookback: int = VOL_LOOKBACK
    version: str = STRATEGY_VERSION
    theoretical_category: str = "volume"
    signal_type: str = "high_vol_grab_reclaim"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="high_vol_grab_reclaim",
            version=self.version,
            parameter_hash=hash_params(
                {"reclaim_win": self.reclaim_win, "vol_lookback": self.vol_lookback, "rule": "vol>=p50"}
            ),
            description=(
                "Why it might beat a coin: absorption — a stop-run that prints with "
                "above-median volume is more likely institutions buying (or selling) into "
                "the raid, filling aggressive flow and defending the level. That is why "
                "price can reclaim and reverse: the other side got absorbed, not vacuumed. "
                "This axis uses real exchange volume, not OHLC geometry alone. "
                "Mechanics: session-range edge raid whose grab-bar volume ≥ 50th percentile "
                f"(median) of the prior {self.vol_lookback} bars; enter on reclaim within "
                f"{self.reclaim_win} bars; stop beyond sweep extreme; target far edge or 2R."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        for e in collect_raid_events(bars, prim, sr):
            if e.reclaim_bar is None or np.isnan(e.vol_median):
                continue
            if e.grab_volume < e.vol_median:
                continue
            t = make_reclaim_trade(bars, ctx, sid, e)
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class LowVolGrabReclaimStrategy:
    """Hypothesis 6 — low-volume grab as vacuum (rival to 5)."""

    reclaim_win: int = RECLAIM_WIN
    vol_lookback: int = VOL_LOOKBACK
    version: str = STRATEGY_VERSION
    theoretical_category: str = "volume"
    signal_type: str = "low_vol_grab_reclaim"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="low_vol_grab_reclaim",
            version=self.version,
            parameter_hash=hash_params(
                {"reclaim_win": self.reclaim_win, "vol_lookback": self.vol_lookback, "rule": "vol<=p50"}
            ),
            description=(
                "Why it might beat a coin: vacuum / thin-air sweep — a raid on quiet volume "
                "means little opposition at the level, so the break is fragile and snaps "
                "back once the wick is done. Rival to high_vol_grab_reclaim: same geometry, "
                "symmetric volume split at the median — the data adjudicates absorption vs vacuum. "
                "Mechanics: session-range edge raid with grab-bar volume ≤ 50th percentile "
                f"(median) of the prior {self.vol_lookback} bars; enter on reclaim within "
                f"{self.reclaim_win} bars; stop beyond sweep extreme; target far edge or 2R."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        for e in collect_raid_events(bars, prim, sr):
            if e.reclaim_bar is None or np.isnan(e.vol_median):
                continue
            if e.grab_volume > e.vol_median:
                continue
            t = make_reclaim_trade(bars, ctx, sid, e)
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class TightAsiaSpringStrategy:
    """Hypothesis 9 — tight Asia coiled spring into London/NY raid-reclaim."""

    reclaim_win: int = RECLAIM_WIN
    version: str = STRATEGY_VERSION
    theoretical_category: str = "range"
    signal_type: str = "tight_asia_raid_reclaim"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="tight_asia_spring",
            version=self.version,
            parameter_hash=hash_params({"reclaim_win": self.reclaim_win, "asia": "tight_p40"}),
            description=(
                "Why it might beat a coin: Wyckoff coiled-spring — a tight overnight Asia "
                "range is a defined box of resting inventory. When London or NY raids that "
                "Asia edge and reclaims, the move is a spring out of compression, not a "
                "random wick in an already-expanded range. Tight ranges give the spring "
                "something real to reverse into. "
                "Mechanics: require prior completed Asia width ≤ 40th percentile of the last "
                "20 Asia widths; take a session-range raid that tags the Asia high/low and "
                f"reclaims within {self.reclaim_win} bars; stop beyond the raid; target the "
                "far Asia edge (or 2R)."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        asia_hi, asia_lo = ffill_asia_levels(prim)
        atr = prim.atr.to_numpy()
        tight = asia_width_is_tight(asia_hi, asia_lo)
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        for e in collect_raid_events(bars, prim, sr):
            if e.reclaim_bar is None or not tight[e.grab_bar]:
                continue
            a_hi, a_lo = asia_hi[e.grab_bar], asia_lo[e.grab_bar]
            if np.isnan(a_hi) or np.isnan(a_lo):
                continue
            # Raid must interact with Asia edge (level near asia hi/lo).
            tol = 0.15 * float(atr[e.grab_bar]) if not np.isnan(atr[e.grab_bar]) else 0.0
            near_asia = abs(e.level - a_hi) <= tol or abs(e.level - a_lo) <= tol
            if not near_asia:
                continue
            t = make_reclaim_trade_asia(bars, ctx, sid, e, a_hi=a_hi, a_lo=a_lo)
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


def make_reclaim_trade_asia(
    bars: pd.DataFrame,
    ctx: StrategyContext,
    strategy_id: str,
    event,
    *,
    a_hi: float,
    a_lo: float,
) -> TradeCandidate | None:
    t = make_reclaim_trade(bars, ctx, strategy_id, event, use_far_edge=False, target_r=TARGET_R)
    if t is None:
        return None
    entry = t.entry_price
    risk = t.risk
    if event.direction == "up":
        planned = a_lo if a_lo < entry else entry - TARGET_R * risk
    else:
        planned = a_hi if a_hi > entry else entry + TARGET_R * risk
    return TradeCandidate(
        direction=t.direction,
        entry_bar=t.entry_bar,
        entry_price=t.entry_price,
        stop_price=t.stop_price,
        risk=t.risk,
        strategy_id=strategy_id,
        symbol=ctx.symbol,
        planned_target=planned,
        extras={**t.extras, "asia_high": a_hi, "asia_low": a_lo},
    )


@dataclass
class AsiaMidBiasRaidStrategy:
    """Hypothesis 11 — premium/discount vs Asia mid at session open."""

    reclaim_win: int = RECLAIM_WIN
    version: str = STRATEGY_VERSION
    theoretical_category: str = "range"
    signal_type: str = "asia_mid_bias_raid_reclaim"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="asia_mid_bias_raid",
            version=self.version,
            parameter_hash=hash_params({"reclaim_win": self.reclaim_win, "bias": "asia_mid"}),
            description=(
                "Why it might beat a coin: before London opens, the Asia mid marks overnight "
                "equilibrium. Price above that mid is premium (sellers have the better side); "
                "price below is discount (buyers). Raids that fade with that context are "
                "trading into value, not randomly. "
                "Mechanics: at London or NY killzone open, compare price to the prior Asia "
                "mid. Only take first-of-session raid-and-reclaims aligned with that bias "
                "(premium → short after high raid; discount → long after low raid); "
                f"reclaim within {self.reclaim_win} bars; stop beyond sweep extreme."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        asia_hi, asia_lo = ffill_asia_levels(prim)
        asia_mid = (asia_hi + asia_lo) / 2.0
        close = bars["close"].to_numpy()
        sessions = prim.sessions
        kz_start = (sessions["london_start"] | sessions["ny_start"]).to_numpy()

        # Bias frozen at each killzone open.
        bias_at = np.full(len(bars), "", dtype=object)
        last_bias = ""
        for i in range(len(bars)):
            if kz_start[i] and not np.isnan(asia_mid[i]):
                last_bias = "premium" if close[i] > asia_mid[i] else "discount"
            bias_at[i] = last_bias

        sid = self.metadata().id
        out: list[TradeCandidate] = []
        for e in collect_raid_events(bars, prim, sr):
            if not e.first_of_kz or e.reclaim_bar is None:
                continue
            bias = bias_at[e.grab_bar]
            if bias == "premium" and e.direction != "up":
                continue
            if bias == "discount" and e.direction != "dn":
                continue
            if bias not in ("premium", "discount"):
                continue
            t = make_reclaim_trade(bars, ctx, sid, e)
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class LondonNySameDirectionStrategy:
    """Hypothesis 13 — London reclaim then same-direction NY reclaim."""

    reclaim_win: int = RECLAIM_WIN
    version: str = STRATEGY_VERSION
    theoretical_category: str = "pattern"
    signal_type: str = "london_then_ny_same_dir"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="london_ny_same_direction",
            version=self.version,
            parameter_hash=hash_params({"id": "london_ny_same_direction", "reclaim_win": self.reclaim_win}),
            description=(
                "Why it might beat a coin: two-session institutional commitment — if London "
                "already raided and reclaimed one side, a same-direction NY raid later that "
                "UTC day is confirmation of the day’s draw, not a fresh coin flip. London "
                "sets the narrative; NY funds it. "
                "Mechanics: completed London raid+reclaim, then a later NY raid+reclaim in "
                "the same direction on the same UTC calendar day; enter on the NY reclaim; "
                "stop beyond the NY sweep extreme; target far session edge or 2R."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        events = [e for e in collect_raid_events(bars, prim, sr) if e.reclaim_bar is not None]
        lon = [e for e in events if e.session == "LONDON"]
        ny = [e for e in events if e.session == "NY"]
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        used_ny: set[int] = set()
        for L in lon:
            for N in ny:
                if N.utc_day != L.utc_day or N.direction != L.direction:
                    continue
                if N.grab_bar <= L.reclaim_bar:  # type: ignore[operator]
                    continue
                if N.reclaim_bar in used_ny:
                    continue
                t = make_reclaim_trade(bars, ctx, sid, N)
                if t is not None:
                    out.append(t)
                    used_ny.add(N.reclaim_bar)  # type: ignore[arg-type]
                    break
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class FailedRaidNextSessionFadeStrategy:
    """Hypothesis 16 — failed reclaim, fade when next session closes back through."""

    fail_win: int = FAIL_WIN
    version: str = STRATEGY_VERSION
    theoretical_category: str = "pattern"
    signal_type: str = "failed_raid_next_kz_fade"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="failed_raid_next_session_fade",
            version=self.version,
            parameter_hash=hash_params({"fail_win": self.fail_win}),
            description=(
                "Why it might beat a coin: inverted HT — a raid that never reclaims within "
                f"{self.fail_win} bars took real liquidity and held. The next killzone’s "
                "first close back through that failed edge is the delayed mean-reversion "
                "when trapped breakout traders finally unwind, not another manipulative "
                "wick. Orthogonal to setups that require an immediate MSS. "
                "Mechanics: session-range raid with no reclaim inside the fail window; in "
                "the next London/NY killzone, enter on the first close back through the "
                "failed edge; stop beyond the original sweep extreme; target 2R."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        prim = ensure_primitives(bars, ctx)
        sr = session_range_frame(bars, prim)
        close = bars["close"].to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        sessions = prim.sessions
        kz_start = (sessions["london_start"] | sessions["ny_start"]).to_numpy()
        in_london = sessions["in_london"].to_numpy()
        in_ny = sessions["in_ny"].to_numpy()
        events = collect_raid_events(bars, prim, sr)
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        n = len(bars)

        for e in events:
            if reclaim_bar(close, e.grab_bar, e.direction, e.level, self.fail_win) is not None:
                continue
            nxt = None
            for j in range(e.grab_bar + 1, n):
                if kz_start[j]:
                    nxt = j
                    break
            if nxt is None:
                continue
            if sessions["london_start"].iloc[nxt]:
                in_sess = in_london
            elif sessions["ny_start"].iloc[nxt]:
                in_sess = in_ny
            else:
                continue
            fade = first_fade_in_session(close, in_sess, nxt, e.direction, e.level)
            if fade is None:
                continue
            direction = "short" if e.direction == "up" else "long"
            extreme = float(high[e.grab_bar]) if e.direction == "up" else float(low[e.grab_bar])
            t = make_fade_trade(
                bars,
                ctx,
                sid,
                entry_bar=fade,
                direction=direction,
                level=e.level,
                extreme=extreme,
            )
            if t is not None:
                out.append(t)
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class SmtFadeSweeperStrategy:
    """Hypothesis 17 — SMT: fade the sweeper (BTC vs ETH)."""

    lookback: int = 20
    reclaim_win: int = RECLAIM_WIN
    version: str = STRATEGY_VERSION
    requires_symbols: tuple[str, ...] = (ETH,)
    theoretical_category: str = "cross-asset"
    signal_type: str = "smt_fade_sweeper"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="smt_fade_sweeper_btc_eth",
            version=self.version,
            parameter_hash=hash_params(
                {"lookback": self.lookback, "reclaim_win": self.reclaim_win, "pair": "BTC/ETH", "role": "fade_sweeper"}
            ),
            description=(
                "Why it might beat a coin: SMT divergence — when BTC sweeps a swing extreme "
                "and ETH refuses to confirm, the sweeper is the weaker name. Fading BTC on "
                "the reclaim trades that relative weakness: the stop-run failed to drag the "
                "correlated twin, so BTC’s sweep is more likely exhaustion than leadership. "
                "Mechanics: only on BTC with ETH in aux_bars; when BTC prints a new "
                f"{self.lookback}-bar high/low and ETH does not confirm, enter BTC on reclaim "
                f"of that level within {self.reclaim_win} bars; stop beyond the sweep; target 2R. "
                "BTC/ETH kept for train (fade BTC) and later holdout confirmation on ETH."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        if ctx.symbol != BTC:
            return []
        aux = (ctx.aux_bars or {}).get(ETH)
        if aux is None or aux.empty or "low" not in aux.columns:
            return []
        # Align length: aux is left-joined to primary index already.
        divs = smt_divergences(bars, aux, lookback=self.lookback)
        close = bars["close"].to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        cooldown = 0
        for d in divs:
            i = d["bar"]
            if i < cooldown:
                continue
            level = d["sweep_level"]
            if d["side"] == "low":
                rec = None
                for k in range(1, self.reclaim_win + 1):
                    j = i + k
                    if j >= len(close):
                        break
                    if close[j] > level:
                        rec = j
                        break
                if rec is None:
                    continue
                entry = float(close[rec])
                stop = float(min(low[i : rec + 1]))
                if not (stop < entry):
                    continue
                risk = entry - stop
                out.append(
                    TradeCandidate(
                        direction="long",
                        entry_bar=rec,
                        entry_price=entry,
                        stop_price=stop,
                        risk=risk,
                        strategy_id=sid,
                        symbol=ctx.symbol,
                        planned_target=entry + TARGET_R * risk,
                        extras={"entry_time": int(bars["timestamp"].iloc[rec]), "smt_side": "low"},
                    )
                )
                cooldown = rec + self.lookback
            else:
                rec = None
                for k in range(1, self.reclaim_win + 1):
                    j = i + k
                    if j >= len(close):
                        break
                    if close[j] < level:
                        rec = j
                        break
                if rec is None:
                    continue
                entry = float(close[rec])
                stop = float(max(high[i : rec + 1]))
                if not (stop > entry):
                    continue
                risk = stop - entry
                out.append(
                    TradeCandidate(
                        direction="short",
                        entry_bar=rec,
                        entry_price=entry,
                        stop_price=stop,
                        risk=risk,
                        strategy_id=sid,
                        symbol=ctx.symbol,
                        planned_target=entry - TARGET_R * risk,
                        extras={"entry_time": int(bars["timestamp"].iloc[rec]), "smt_side": "high"},
                    )
                )
                cooldown = rec + self.lookback
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


@dataclass
class SmtTradeHolderBtcSolStrategy:
    """Hypothesis 18 — SMT: trade the holder (BTC sweeper / SOL holder / enter SOL)."""

    lookback: int = 20
    version: str = STRATEGY_VERSION
    requires_symbols: tuple[str, ...] = (BTC,)
    theoretical_category: str = "cross-asset"
    signal_type: str = "smt_trade_holder"

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            id="smt_trade_holder_btc_sol",
            version=self.version,
            parameter_hash=hash_params(
                {"lookback": self.lookback, "pair": "BTC/SOL", "role": "trade_holder", "enter": "SOL"}
            ),
            description=(
                "Why it might beat a coin: trade strength, not the raid — when BTC sweeps a "
                "swing and SOL refuses to confirm, SOL is the relative leader. Entering SOL "
                "in the divergent direction follows the name that held liquidity, rather "
                "than fading the weaker sweeper. Same SMT observation as the fade-sweeper "
                "rival; opposite execution. Pair is BTC/SOL (both train symbols) so the "
                "hypothesis is evaluable without touching holdout ETH; BTC/ETH reserved "
                "for later confirmation if this clears. "
                "Mechanics: only on SOL with BTC in aux_bars; when BTC prints a new "
                f"{self.lookback}-bar high/low and SOL does not confirm, enter SOL in the "
                "bounce direction at the divergence bar (or next confirming close); stop "
                "beyond SOL’s held extreme; target 2R."
            ),
        )

    def generate_trades(self, bars: pd.DataFrame, ctx: StrategyContext) -> list[TradeCandidate]:
        if ctx.symbol != SOL:
            return []
        aux = (ctx.aux_bars or {}).get(BTC)
        if aux is None or aux.empty or "low" not in aux.columns:
            return []
        divs = smt_divergences(aux, bars, lookback=self.lookback)
        close = bars["close"].to_numpy()
        open_ = bars["open"].to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        sid = self.metadata().id
        out: list[TradeCandidate] = []
        cooldown = 0
        for d in divs:
            i = d["bar"]
            if i < cooldown or i >= len(close):
                continue
            held = d["holder_extreme"]
            if d["side"] == "low":
                # Long SOL: prefer bounce close, else next up-close within 3 bars.
                entry_bar = None
                if close[i] > open_[i] and not np.isnan(close[i]):
                    entry_bar = i
                else:
                    for k in range(1, 4):
                        j = i + k
                        if j >= len(close):
                            break
                        if close[j] > close[j - 1]:
                            entry_bar = j
                            break
                if entry_bar is None:
                    continue
                entry = float(close[entry_bar])
                stop = float(min(held, float(np.nanmin(low[i : entry_bar + 1]))))
                if not (stop < entry):
                    continue
                risk = entry - stop
                out.append(
                    TradeCandidate(
                        direction="long",
                        entry_bar=entry_bar,
                        entry_price=entry,
                        stop_price=stop,
                        risk=risk,
                        strategy_id=sid,
                        symbol=ctx.symbol,
                        planned_target=entry + TARGET_R * risk,
                        extras={"entry_time": int(bars["timestamp"].iloc[entry_bar]), "smt_side": "low"},
                    )
                )
                cooldown = entry_bar + self.lookback
            else:
                entry_bar = None
                if close[i] < open_[i] and not np.isnan(close[i]):
                    entry_bar = i
                else:
                    for k in range(1, 4):
                        j = i + k
                        if j >= len(close):
                            break
                        if close[j] < close[j - 1]:
                            entry_bar = j
                            break
                if entry_bar is None:
                    continue
                entry = float(close[entry_bar])
                stop = float(max(held, float(np.nanmax(high[i : entry_bar + 1]))))
                if not (stop > entry):
                    continue
                risk = stop - entry
                out.append(
                    TradeCandidate(
                        direction="short",
                        entry_bar=entry_bar,
                        entry_price=entry,
                        stop_price=stop,
                        risk=risk,
                        strategy_id=sid,
                        symbol=ctx.symbol,
                        planned_target=entry - TARGET_R * risk,
                        extras={"entry_time": int(bars["timestamp"].iloc[entry_bar]), "smt_side": "high"},
                    )
                )
                cooldown = entry_bar + self.lookback
        return out

    def tags(self, trade: TradeCandidate, bars: pd.DataFrame, ctx: StrategyContext) -> Mapping[str, Any]:
        return _empty_tags(trade, bars, ctx)


def register_hypotheses(register_fn) -> None:
    """Register the ten survivors under their strategy ids."""
    factories = {
        "kz_first_raid_reclaim": KzFirstRaidReclaimStrategy,
        "kz_first_30m_raid_reclaim": KzFirst30mRaidReclaimStrategy,
        "high_vol_grab_reclaim": HighVolGrabReclaimStrategy,
        "low_vol_grab_reclaim": LowVolGrabReclaimStrategy,
        "tight_asia_spring": TightAsiaSpringStrategy,
        "asia_mid_bias_raid": AsiaMidBiasRaidStrategy,
        "london_ny_same_direction": LondonNySameDirectionStrategy,
        "failed_raid_next_session_fade": FailedRaidNextSessionFadeStrategy,
        "smt_fade_sweeper_btc_eth": SmtFadeSweeperStrategy,
        "smt_trade_holder_btc_sol": SmtTradeHolderBtcSolStrategy,
    }
    for name, cls in factories.items():
        register_fn(name, cls)
