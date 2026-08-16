"""Persistent hypothesis memory — what categories/signals have been tested.

Every batch appends one row per strategy to ``data/memory/hypothesis_log.csv``.
Before implementing a new strategy, call ``prior_for_category`` / ``query_log``
to see whether the same theoretical category and signal type already ran and
what it produced.

Categories (locked vocabulary):
  timing | volume | range | cross-asset | pattern
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

CATEGORIES = ("timing", "volume", "range", "cross-asset", "pattern")

LOG_COLUMNS = [
    "strategy_id",
    "parameter_hash",
    "theoretical_category",
    "key_parameters",
    "training_result_summary",
    "holdout_result",
    "promoted",
    "date",
    "plain_English_description",
]

# Default category + short signal type for known strategies (override via
# strategy.theoretical_category / strategy.signal_type / strategy.key_parameters).
_STRATEGY_META: dict[str, dict[str, Any]] = {
    "kz_first_raid_reclaim": {
        "category": "timing",
        "signal_type": "first_kz_raid_reclaim",
        "key_parameters": {"reclaim_win": 3},
    },
    "kz_first_30m_raid_reclaim": {
        "category": "timing",
        "signal_type": "first_30m_raid_reclaim",
        "key_parameters": {"reclaim_win": 3, "window": "first_30m"},
    },
    "high_vol_grab_reclaim": {
        "category": "volume",
        "signal_type": "high_vol_grab_reclaim",
        "key_parameters": {"vol_rule": ">=p50", "vol_lookback": 50},
    },
    "low_vol_grab_reclaim": {
        "category": "volume",
        "signal_type": "low_vol_grab_reclaim",
        "key_parameters": {"vol_rule": "<=p50", "vol_lookback": 50},
    },
    "tight_asia_spring": {
        "category": "range",
        "signal_type": "tight_asia_raid_reclaim",
        "key_parameters": {"asia_width": "<=p40_of_20"},
    },
    "asia_mid_bias_raid": {
        "category": "range",
        "signal_type": "asia_mid_bias_raid_reclaim",
        "key_parameters": {"bias": "asia_mid_at_kz_open"},
    },
    "london_ny_same_direction": {
        "category": "pattern",
        "signal_type": "london_then_ny_same_dir",
        "key_parameters": {"sessions": "London->NY", "same_utc_day": True},
    },
    "failed_raid_next_session_fade": {
        "category": "pattern",
        "signal_type": "failed_raid_next_kz_fade",
        "key_parameters": {"fail_win": 20},
    },
    "smt_fade_sweeper_btc_eth": {
        "category": "cross-asset",
        "signal_type": "smt_fade_sweeper",
        "key_parameters": {"pair": "BTC/ETH", "enter": "BTC", "lookback": 20},
    },
    "smt_trade_holder_btc_sol": {
        "category": "cross-asset",
        "signal_type": "smt_trade_holder",
        "key_parameters": {"pair": "BTC/SOL", "enter": "SOL", "lookback": 20},
    },
    "holy_trinity_v10": {
        "category": "pattern",
        "signal_type": "session_range_mss_fvg",
        "key_parameters": {"gate": "holy_trinity_v10"},
    },
}

DEFAULT_LOG_PATH = Path("data/memory/hypothesis_log.csv")


def default_log_path(repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        # src/ht_backtest/memory/hypothesis_log.py -> repo root
        return Path(__file__).resolve().parents[3] / "data" / "memory" / "hypothesis_log.csv"
    return Path(repo_root) / "data" / "memory" / "hypothesis_log.csv"


def resolve_category(strategy: Any, strategy_id: str | None = None) -> str:
    sid = strategy_id or getattr(getattr(strategy, "metadata", lambda: None)(), "id", None) or ""
    raw = getattr(strategy, "theoretical_category", None)
    if raw is None and sid in _STRATEGY_META:
        raw = _STRATEGY_META[sid]["category"]
    cat = str(raw or "pattern").strip().lower()
    if cat not in CATEGORIES:
        raise ValueError(f"theoretical_category must be one of {CATEGORIES}, got {cat!r}")
    return cat


def resolve_key_parameters(strategy: Any, strategy_id: str | None = None) -> str:
    """JSON object string: always includes signal_type plus strategy params."""
    sid = strategy_id or ""
    base = dict(_STRATEGY_META.get(sid, {}).get("key_parameters", {}))
    signal = getattr(strategy, "signal_type", None) or _STRATEGY_META.get(sid, {}).get("signal_type", sid)
    override = getattr(strategy, "key_parameters", None)
    if isinstance(override, Mapping):
        base.update(dict(override))
    payload = {"signal_type": signal, **base}
    return json.dumps(payload, sort_keys=True, default=str)


def load_log(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else default_log_path()
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Normalize boolean-ish promoted
    df["promoted"] = df["promoted"].astype(str)
    return df[LOG_COLUMNS]


def append_records(records: list[dict[str, Any]], path: str | Path | None = None) -> Path:
    p = Path(path) if path else default_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_log(p)
    rows = []
    for r in records:
        row = {c: "" for c in LOG_COLUMNS}
        for c in LOG_COLUMNS:
            v = r.get(c, "")
            if c == "promoted":
                row[c] = "True" if v in (True, "True", "true", "1") else "False"
            else:
                row[c] = "" if v is None else str(v)
        rows.append(row)
    new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    out = pd.concat([existing, new], ignore_index=True)
    out.to_csv(p, index=False)
    return p


def training_summary_from_comparison(comp_rows: pd.DataFrame) -> tuple[str, bool]:
    """Best train diff_pp across horizons; promoted if any horizon promoted."""
    if comp_rows.empty:
        return "no_trades", False
    best_i = comp_rows["diff_pp"].idxmax()
    best = comp_rows.loc[best_i]
    summary = (
        f"best_diff_pp={float(best['diff_pp']):+.2f} "
        f"at {float(best['target_R']):.1f}R n={int(best['n'])}"
    )
    promoted = bool(comp_rows["promoted"].any()) if "promoted" in comp_rows.columns else False
    return summary, promoted


def format_prior_results_summary(path: str | Path | None = None) -> str:
    """Category coverage: exhausted (tested, none promoted) vs unexplored vs promoted."""
    df = load_log(path)
    lines = ["PRIOR RESULTS (hypothesis memory)", "=" * 72]
    if df.empty:
        lines.append("(log empty — all categories unexplored)")
        for cat in CATEGORIES:
            lines.append(f"  {cat:<12}  UNEXPLORED")
        lines.append("=" * 72)
        return "\n".join(lines)

    # Normalize promoted
    promo = df["promoted"].astype(str).str.lower().isin(("true", "1", "yes"))

    for cat in CATEGORIES:
        sub = df[df["theoretical_category"].astype(str).str.lower() == cat]
        if sub.empty:
            lines.append(f"  {cat:<12}  UNEXPLORED")
            continue
        n = len(sub)
        any_promo = bool(promo.loc[sub.index].any())
        # Parse best train edge from summaries when possible
        best_edge = None
        best_sid = ""
        for _, row in sub.iterrows():
            m = re.search(r"best_diff_pp=([+-]?\d+(?:\.\d+)?)", str(row["training_result_summary"]))
            if m:
                v = float(m.group(1))
                if best_edge is None or v > best_edge:
                    best_edge = v
                    best_sid = str(row["strategy_id"])
        hold_n = int(
            sub["holdout_result"].astype(str).str.strip().replace("nan", "").replace("None", "").str.len().gt(0).sum()
        )
        if any_promo:
            status = "PROMOTED (do not re-test lightly)"
        else:
            status = "EXHAUSTED (tested, none promoted)"
        edge_s = f"{best_edge:+.2f}pp ({best_sid})" if best_edge is not None else "n/a"
        lines.append(
            f"  {cat:<12}  {status}  | n_hypotheses={n}  best_train={edge_s}  "
            f"holdout_logged={hold_n}"
        )
        # Compact per-strategy lines
        for _, row in sub.iterrows():
            st = str(row.get("key_parameters", ""))
            try:
                sig = json.loads(st).get("signal_type", "")
            except Exception:  # noqa: BLE001
                sig = ""
            lines.append(
                f"      - {row['strategy_id']}  signal={sig or '?'}  "
                f"{row['training_result_summary']}  "
                f"holdout={row['holdout_result'] or '—'}  "
                f"promoted={row['promoted']}"
            )

    lines.append("=" * 72)
    return "\n".join(lines)


def query_log(
    *,
    category: str | None = None,
    signal_type: str | None = None,
    strategy_id: str | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Filter the log before implementing a new strategy."""
    df = load_log(path)
    if df.empty:
        return df
    out = df
    if category:
        out = out[out["theoretical_category"].astype(str).str.lower() == category.lower()]
    if strategy_id:
        out = out[out["strategy_id"].astype(str) == strategy_id]
    if signal_type:
        mask = []
        for raw in out["key_parameters"].astype(str):
            try:
                mask.append(json.loads(raw).get("signal_type") == signal_type)
            except Exception:  # noqa: BLE001
                mask.append(False)
        out = out[mask]
    return out.reset_index(drop=True)


def prior_for_category(category: str, path: str | Path | None = None) -> str:
    """Human summary of prior tests in one category (for agent pre-implementation checks)."""
    cat = category.strip().lower()
    if cat not in CATEGORIES:
        return f"unknown category {category!r}; known: {CATEGORIES}"
    sub = query_log(category=cat, path=path)
    if sub.empty:
        return f"category {cat!r}: UNEXPLORED — no prior hypotheses in the log."
    lines = [f"category {cat!r}: {len(sub)} prior hypothesis(es)"]
    for _, row in sub.iterrows():
        lines.append(
            f"  {row['strategy_id']} hash={row['parameter_hash']}: "
            f"{row['training_result_summary']}; holdout={row['holdout_result'] or 'none'}; "
            f"promoted={row['promoted']}"
        )
    return "\n".join(lines)


def records_from_batch_comparison(
    comparison: pd.DataFrame,
    *,
    strategies: Mapping[str, Any],
    date: str | None = None,
    holdout_by_strategy: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build log rows from a batch comparison table (one row per strategy_id)."""
    if comparison.empty:
        return []
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    holdout_by_strategy = dict(holdout_by_strategy or {})
    records: list[dict[str, Any]] = []
    for sid, grp in comparison.groupby("strategy_id"):
        strategy = strategies.get(str(sid))
        meta = strategy.metadata() if strategy is not None else None
        param_hash = ""
        desc = ""
        if meta is not None:
            param_hash = meta.parameter_hash
            desc = meta.description
        elif "strategy_parameter_hash" in grp.columns and len(grp):
            param_hash = str(grp["strategy_parameter_hash"].iloc[0])
        if "description" in grp.columns and len(grp) and not desc:
            desc = str(grp["description"].iloc[0])
        summary, promoted = training_summary_from_comparison(grp)
        cat = resolve_category(strategy, str(sid)) if strategy is not None else (
            _STRATEGY_META.get(str(sid), {}).get("category", "pattern")
        )
        keys = resolve_key_parameters(strategy, str(sid)) if strategy is not None else json.dumps(
            {"signal_type": _STRATEGY_META.get(str(sid), {}).get("signal_type", sid)},
            sort_keys=True,
        )
        records.append(
            {
                "strategy_id": str(sid),
                "parameter_hash": param_hash,
                "theoretical_category": cat,
                "key_parameters": keys,
                "training_result_summary": summary,
                "holdout_result": holdout_by_strategy.get(str(sid), ""),
                "promoted": promoted,
                "date": date,
                "plain_English_description": desc,
            }
        )
    return records


def update_holdout_result(
    strategy_id: str,
    holdout_result: str,
    *,
    parameter_hash: str | None = None,
    path: str | Path | None = None,
) -> None:
    """Patch the latest matching row's holdout_result (e.g. after a pre-registered test)."""
    p = Path(path) if path else default_log_path()
    df = load_log(p)
    if df.empty:
        return
    mask = df["strategy_id"].astype(str) == strategy_id
    if parameter_hash:
        mask &= df["parameter_hash"].astype(str) == parameter_hash
    idxs = df.index[mask]
    if len(idxs) == 0:
        return
    df.loc[idxs[-1], "holdout_result"] = str(holdout_result)
    df.to_csv(p, index=False)
