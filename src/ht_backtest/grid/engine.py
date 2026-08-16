"""Depth-2/3 condition grid over base trades + FDR (BH / BY).

None handling: a trade counts in combo n only if every condition in the combo
is True. None excludes the trade (None is not False).
"""

from __future__ import annotations

import itertools
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ht_backtest.conditions.features import enrich_condition_features
from ht_backtest.conditions.registry import (
    ALL_CONDITIONS,
    MUTEX_PAIRS,
    is_mutex_combo,
    library_version,
)
from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import SplitManifest
from ht_backtest.reports.reach import reach_vs_random_walk
from ht_backtest.reports.universe_report import generate_pooled_trades
from ht_backtest.strategies.registry import get_strategy

# Excluded from combo generation for the session-range reclaim base grid
GRID_EXCLUDE_IDS: frozenset[str] = frozenset({"asia_session"})

# Extra mutexes for this grid (also registered globally in registry)
GRID_EXTRA_MUTEX: tuple[tuple[str, str], ...] = (
    ("london_session", "london_raided_high"),
    ("london_session", "london_raided_low"),
)

MIN_N = 200
FDR_Q = 0.05
SURVIVOR_MIN_DIFF_PP = 3.0
PRIMARY_TARGET = 1.0


def grid_condition_ids() -> list[str]:
    return [c.id for c in ALL_CONDITIONS if c.id not in GRID_EXCLUDE_IDS]


def all_mutex_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(dict.fromkeys([*MUTEX_PAIRS, *GRID_EXTRA_MUTEX]))


def is_mutex_combo_grid(condition_ids: Sequence[str]) -> bool:
    s = set(condition_ids)
    for a, b in all_mutex_pairs():
        if a in s and b in s:
            return True
    return False


def signature(base_id: str, conds: Sequence[str]) -> str:
    parts = sorted(conds)
    return f"base={base_id}|d={len(parts)}|c={'+'.join(parts)}"


def one_sided_binom_pvalue(hits: int, n: int, p0: float = 0.5) -> float:
    """P(X >= hits) under X~Binomial(n, p0). Exact via survival sum."""
    if n <= 0:
        return 1.0
    hits = max(0, min(int(hits), int(n)))
    # Use regularized incomplete beta / recursive probabilities for stability
    # P(X=k) = C(n,k) p^k (1-p)^{n-k}
    # Compute SF from hits..n in log-space
    log_p = math.log(p0)
    log_q = math.log(1.0 - p0)
    # start at k=hits
    # log C(n,k) via lgamma
    def log_pmf(k: int) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_p
            + (n - k) * log_q
        )

    # Sum carefully
    max_log = max(log_pmf(k) for k in range(hits, n + 1))
    total = 0.0
    for k in range(hits, n + 1):
        total += math.exp(log_pmf(k) - max_log)
    return float(min(1.0, math.exp(max_log) * total))


def bh_reject(p_values: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Return boolean mask of BH rejections at level q (independent/PRDS)."""
    m = len(p_values)
    out = np.zeros(m, dtype=bool)
    if m == 0:
        return out
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if not below.any():
        return out
    k_max = int(np.where(below)[0].max())
    out[order[: k_max + 1]] = True
    return out


def by_reject(p_values: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini–Yekutieli (any dependence): q / H_m with H_m = sum_{i=1}^m 1/i."""
    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool)
    h_m = sum(1.0 / i for i in range(1, m + 1))
    return bh_reject(p_values, q=q / h_m)


@dataclass
class TradeMatrix:
    """Base train trades + boolean/object condition columns."""

    trades: pd.DataFrame  # includes reach_* and condition eval cols
    condition_ids: list[str]
    base_id: str = "kz_first_raid_reclaim"

    @property
    def n_base(self) -> int:
        return len(self.trades)


def build_trade_matrix(
    *,
    split_path: str | Path = "specs/splits/v1.json",
    cache_dir: str | Path = "data/raw",
    funding_dir: str | Path = "data/funding",
    workers: int = 4,
    condition_ids: Sequence[str] | None = None,
    extra_condition_cols: dict[str, np.ndarray] | None = None,
    log_fn=print,
) -> TradeMatrix:
    split = SplitManifest.load(split_path)
    strategy = get_strategy("kz_first_raid_reclaim")
    trades, _ = generate_pooled_trades(
        strategy,
        split,
        timeframe="15m",
        workers=workers,
        strategy_name="kz_first_raid_reclaim",
        split_path=str(split_path),
        cache_dir=str(cache_dir),
        funding_dir=str(funding_dir),
        attach_funding=True,
        log_fn=lambda *a, **k: None,
    )
    train = trades[trades["split"] == "train"].copy().reset_index(drop=True)
    ids = list(condition_ids) if condition_ids is not None else grid_condition_ids()

    # Eval conditions per symbol at entry_bar
    dl = OHLCVDownloader(cache_dir=cache_dir)
    cond_maps: dict[str, list[Any]] = {cid: [None] * len(train) for cid in ids}

    for sym, g in train.groupby("symbol"):
        bars = dl.cached_range(sym, "15m", 0, 4_102_444_800_000)
        if bars.empty:
            continue
        enriched = enrich_condition_features(
            bars,
            sym,
            cache_dir=cache_dir,
            funding_dir=funding_dir,
        )
        from ht_backtest.conditions.registry import CONDITION_BY_ID

        entry_bars = g["entry_bar"].astype(int)
        for cid in ids:
            if cid not in CONDITION_BY_ID:
                continue
            series = CONDITION_BY_ID[cid].eval(enriched)
            for idx, eb in zip(g.index, entry_bars):
                if 0 <= int(eb) < len(series):
                    cond_maps[cid][idx] = series.iloc[int(eb)]

    for cid, vals in cond_maps.items():
        train[cid] = vals

    if extra_condition_cols:
        for cid, arr in extra_condition_cols.items():
            if len(arr) != len(train):
                raise ValueError(f"extra col {cid} length mismatch")
            train[cid] = list(arr)
            if cid not in ids:
                ids.append(cid)

    return TradeMatrix(trades=train, condition_ids=ids, base_id="kz_first_raid_reclaim")


def _is_true(v: Any) -> bool:
    """True only for boolean True; False and None excluded (None is not False)."""
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return False


def combo_mask(matrix: TradeMatrix, conds: Sequence[str]) -> np.ndarray:
    """True where every condition is True; False/None → exclude."""
    mask = np.ones(len(matrix.trades), dtype=bool)
    for cid in conds:
        col = matrix.trades[cid].to_numpy()
        ok = np.fromiter((_is_true(v) for v in col), dtype=bool, count=len(col))
        mask &= ok
    return mask


def combo_n(matrix: TradeMatrix, conds: Sequence[str]) -> int:
    return int(combo_mask(matrix, conds).sum())


def score_combo(matrix: TradeMatrix, conds: Sequence[str]) -> dict[str, Any]:
    mask = combo_mask(matrix, conds)
    n = int(mask.sum())
    sub = matrix.trades.loc[mask]
    if n == 0:
        return {
            "n": 0,
            "status": "skipped_low_n",
            "reach_1r_pct": float("nan"),
            "diff_pp_1r": float("nan"),
            "hits_1r": 0,
            "n_1r": 0,
            "p_value_1r": 1.0,
            "reach_json": "[]",
        }
    table = reach_vs_random_walk(sub)
    row1 = table.loc[table["target_R"] == PRIMARY_TARGET]
    if row1.empty:
        reach_pct = float("nan")
        diff_pp = float("nan")
        hits = 0
        n1 = 0
        pval = 1.0
    else:
        r = row1.iloc[0]
        reach_pct = float(r["reach_pct"])
        diff_pp = float(r["diff_pp"])
        n1 = int(r["n"])
        col = "reach_1.0R"
        if n1 and col in sub.columns:
            known = sub[col].notna()
            hits = int(sub.loc[known, col].sum())
        else:
            hits = 0
        pval = one_sided_binom_pvalue(hits, n1, p0=0.5) if n1 else 1.0

    status = "tested" if n >= MIN_N else "skipped_low_n"
    if status == "skipped_low_n":
        pval = float("nan")

    return {
        "n": n,
        "status": status,
        "reach_1r_pct": reach_pct,
        "diff_pp_1r": diff_pp,
        "hits_1r": hits,
        "n_1r": n1,
        "p_value_1r": pval,
        "reach_json": table.to_json(orient="records"),
    }


def enumerate_combos(
    condition_ids: Sequence[str],
    depth: int,
) -> tuple[list[tuple[str, ...]], int, int]:
    """Returns (valid_combos, n_generated, n_mutex_skipped)."""
    generated = 0
    mutex_skipped = 0
    valid: list[tuple[str, ...]] = []
    for combo in itertools.combinations(condition_ids, depth):
        generated += 1
        if is_mutex_combo_grid(combo):
            mutex_skipped += 1
            continue
        valid.append(tuple(sorted(combo)))
    return valid, generated, mutex_skipped


def _init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE grid_runs (
            run_id TEXT PRIMARY KEY,
            base_strategy_id TEXT,
            library_version TEXT,
            depth_max INTEGER,
            q_fdr REAL,
            m_generated INTEGER,
            m_mutex_skipped INTEGER,
            m_low_n INTEGER,
            m_tested INTEGER,
            n_bh_survivors INTEGER,
            n_by_survivors INTEGER,
            n_by_significant INTEGER,
            started_at_utc TEXT,
            finished_at_utc TEXT,
            notes TEXT
        );
        CREATE TABLE grid_results (
            run_id TEXT,
            signature TEXT,
            depth INTEGER,
            c1 TEXT,
            c2 TEXT,
            c3 TEXT,
            categories TEXT,
            n INTEGER,
            status TEXT,
            reach_1r_pct REAL,
            diff_pp_1r REAL,
            hits_1r INTEGER,
            n_1r INTEGER,
            p_value_1r REAL,
            bh_significant INTEGER,
            by_significant INTEGER,
            bh_rank INTEGER,
            PRIMARY KEY (run_id, signature)
        );
        CREATE TABLE grid_survivors (
            run_id TEXT,
            signature TEXT,
            depth INTEGER,
            c1 TEXT,
            c2 TEXT,
            c3 TEXT,
            n INTEGER,
            reach_1r_pct REAL,
            diff_pp_1r REAL,
            p_value_1r REAL,
            bh_significant INTEGER,
            by_significant INTEGER,
            PRIMARY KEY (run_id, signature)
        );
        """
    )
    return conn


def apply_fdr_and_survivors(conn: sqlite3.Connection, run_id: str) -> tuple[int, int, int]:
    rows = list(
        conn.execute(
            "SELECT signature, p_value_1r, diff_pp_1r, n, depth, c1, c2, c3, reach_1r_pct "
            "FROM grid_results WHERE run_id=? AND status='tested' AND p_value_1r IS NOT NULL",
            (run_id,),
        )
    )
    if not rows:
        return 0, 0, 0
    sigs = [r[0] for r in rows]
    pvals = np.array([float(r[1]) for r in rows], dtype=float)
    bh = bh_reject(pvals, FDR_Q)
    by = by_reject(pvals, FDR_Q)
    order = np.argsort(pvals)
    rank = np.empty(len(pvals), dtype=int)
    rank[order] = np.arange(1, len(pvals) + 1)

    for i, sig in enumerate(sigs):
        conn.execute(
            "UPDATE grid_results SET bh_significant=?, by_significant=?, bh_rank=? "
            "WHERE run_id=? AND signature=?",
            (int(bh[i]), int(by[i]), int(rank[i]), run_id, sig),
        )

    conn.execute("DELETE FROM grid_survivors WHERE run_id=?", (run_id,))
    n_bh_surv = 0
    n_by_surv = 0
    n_by_sig = int(by.sum())
    for i, r in enumerate(rows):
        sig, pval, diff_pp, n, depth, c1, c2, c3, reach = r
        if by[i] and diff_pp is not None and float(diff_pp) > SURVIVOR_MIN_DIFF_PP:
            n_by_surv += 1
        if bh[i] and diff_pp is not None and float(diff_pp) > SURVIVOR_MIN_DIFF_PP:
            n_bh_surv += 1
            conn.execute(
                "INSERT INTO grid_survivors "
                "(run_id, signature, depth, c1, c2, c3, n, reach_1r_pct, diff_pp_1r, "
                "p_value_1r, bh_significant, by_significant) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    sig,
                    depth,
                    c1,
                    c2,
                    c3,
                    n,
                    reach,
                    diff_pp,
                    pval,
                    1,
                    int(by[i]),
                ),
            )
    conn.commit()
    return n_bh_surv, n_by_surv, n_by_sig


def run_grid(
    matrix: TradeMatrix,
    *,
    depth: int = 2,
    db_path: str | Path = "data/grid/condition_grid.sqlite",
    run_id: str | None = None,
    notes: str = "",
    log_fn=print,
) -> dict[str, Any]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    db_path = Path(db_path)
    conn = _init_db(db_path)

    ids = [c for c in matrix.condition_ids if c not in GRID_EXCLUDE_IDS or c.startswith("synthetic_")]
    # synthetic_* always included if present
    ids = list(dict.fromkeys(ids))

    valid, n_gen, n_mutex = enumerate_combos(ids, depth)
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_fn(f"GRID START run_id={run_id} depth={depth} generated={n_gen} mutex_skip={n_mutex} valid={len(valid)}")
    t0 = time.perf_counter()

    from ht_backtest.conditions.registry import CONDITION_BY_ID

    n_low = 0
    n_tested = 0
    for combo in valid:
        sig = signature(matrix.base_id, combo)
        scored = score_combo(matrix, combo)
        if scored["status"] == "skipped_low_n":
            n_low += 1
        else:
            n_tested += 1
        cats = sorted(
            {
                CONDITION_BY_ID[c].category
                for c in combo
                if c in CONDITION_BY_ID
            }
            | ({"synthetic"} if any(c.startswith("synthetic_") for c in combo) else set())
        )
        c1, c2, c3 = (list(combo) + [None, None, None])[:3]
        conn.execute(
            "INSERT INTO grid_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                sig,
                depth,
                c1,
                c2,
                c3,
                ",".join(cats),
                scored["n"],
                scored["status"],
                scored["reach_1r_pct"],
                scored["diff_pp_1r"],
                scored["hits_1r"],
                scored["n_1r"],
                scored["p_value_1r"],
                0,
                0,
                None,
            ),
        )

    n_bh, n_by_surv, n_by_sig = apply_fdr_and_survivors(conn, run_id)
    finished = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn.execute(
        "INSERT INTO grid_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            matrix.base_id,
            library_version()[:500],
            depth,
            FDR_Q,
            n_gen,
            n_mutex,
            n_low,
            n_tested,
            n_bh,
            n_by_surv,
            n_by_sig,
            started,
            finished,
            notes,
        ),
    )
    conn.commit()
    elapsed = time.perf_counter() - t0
    log_fn(
        f"GRID FINISH run_id={run_id} elapsed={elapsed:.1f}s tested={n_tested} "
        f"low_n={n_low} bh_survivors={n_bh} by_survivors={n_by_surv} by_sig={n_by_sig}"
    )
    summary = {
        "run_id": run_id,
        "db_path": str(db_path),
        "generated": n_gen,
        "mutex_skipped": n_mutex,
        "low_n": n_low,
        "tested": n_tested,
        "bh_survivors": n_bh,
        "by_survivors": n_by_surv,
        "by_significant": n_by_sig,
        "started": started,
        "finished": finished,
        "elapsed_s": elapsed,
    }
    conn.close()
    return summary


def load_survivors(db_path: str | Path, run_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT * FROM grid_survivors WHERE run_id=? ORDER BY diff_pp_1r DESC",
        conn,
        params=(run_id,),
    )
    conn.close()
    return df


def inject_synthetic_70pct(
    matrix: TradeMatrix,
    *,
    n_select: int = 500,
    reach_rate: float = 0.70,
    seed: int = 42,
    col: str = "synthetic_edge_70",
) -> TradeMatrix:
    """Mark ``col`` True on a fixed set of trades with exact reach_rate at 1R."""
    rng = np.random.default_rng(seed)
    df = matrix.trades
    known = df["reach_1.0R"].notna().to_numpy()
    reaches = df["reach_1.0R"].fillna(False).astype(bool).to_numpy()
    win_idx = np.flatnonzero(known & reaches)
    lose_idx = np.flatnonzero(known & ~reaches)
    n_win = int(round(n_select * reach_rate))
    n_lose = n_select - n_win
    if len(win_idx) < n_win or len(lose_idx) < n_lose:
        raise RuntimeError("not enough winners/losers for synthetic injection")
    pick_w = rng.choice(win_idx, size=n_win, replace=False)
    pick_l = rng.choice(lose_idx, size=n_lose, replace=False)
    chosen = set(int(i) for i in np.concatenate([pick_w, pick_l]))
    flags = np.array([True if i in chosen else False for i in range(len(df))], dtype=object)
    # Use False (not None) when not selected so AND with other conditions is well-defined
    out = matrix.trades.copy()
    out[col] = flags
    ids = list(matrix.condition_ids) + [col]
    return TradeMatrix(trades=out, condition_ids=ids, base_id=matrix.base_id)
