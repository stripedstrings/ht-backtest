"""Human-facing audit report from completed candidate artifacts."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ht_backtest.data.split import SplitManifest

COIN_ZONE_PP = 2.0
POSITIVE_SIGNAL_PP = 2.0  # outside coin zone to the upside


def _load_completed(completed_dir: str | Path) -> dict[str, Any]:
    d = Path(completed_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"completed dir not found: {d}")
    reach_path = d / "training_reach_table.csv"
    if not reach_path.exists():
        raise FileNotFoundError(f"missing training_reach_table.csv in {d}")
    reach = pd.read_csv(reach_path)
    meta: dict[str, Any] = {}
    if (d / "metadata.json").exists():
        import json

        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    candidate: dict[str, Any] = {}
    if (d / "candidate.yaml").exists():
        candidate = yaml.safe_load((d / "candidate.yaml").read_text(encoding="utf-8")) or {}
    promo = {}
    if (d / "promotion.json").exists():
        import json

        promo = json.loads((d / "promotion.json").read_text(encoding="utf-8"))
    return {"dir": d, "reach": reach, "meta": meta, "candidate": candidate, "promo": promo}


def _cell_class(diff_pp: float) -> str:
    if abs(diff_pp) <= COIN_ZONE_PP:
        return "coin"
    if diff_pp > 0:
        return "pos"
    return "neg"


def _verdict(reach: pd.DataFrame) -> tuple[str, str]:
    """Return (headline, one_paragraph)."""
    best = float(reach["diff_pp"].max())
    n = int(reach["n"].iloc[0])
    if best > POSITIVE_SIGNAL_PP:
        headline = "This strategy showed a positive signal on training data — here is what that means."
        para = (
            f"On {n:,} training trades, at least one profit target cleared the coin zone "
            f"(more than {POSITIVE_SIGNAL_PP:.0f} percentage points above the random-walk reach rate). "
            "That is a training-sample finding only: it is not a guarantee of future results, "
            "and it has not been confirmed on sealed holdout data."
        )
    else:
        headline = "This strategy is statistically indistinguishable from a coin flip."
        para = (
            f"Across {n:,} training trades, reach rates at every horizon sat at or below the "
            f"random-walk benchmark, or inside the ±{COIN_ZONE_PP:.0f}pp coin zone. "
            "There is no training evidence here of an edge large enough to separate skill from noise."
        )
    return headline, para


def _expected_outcome_1r(reach: pd.DataFrame, *, risk_pct: float = 1.0, n_trades: int = 100) -> dict[str, Any]:
    """Bernoulli +1R / −1R model using empirical 1R reach rate."""
    row = reach[reach["target_R"] == 1.0]
    if row.empty:
        raise ValueError("reach table missing 1.0R row")
    p = float(row.iloc[0]["reach_pct"]) / 100.0
    exp_r_per_trade = 2.0 * p - 1.0
    exp_pct = n_trades * exp_r_per_trade * (risk_pct / 100.0) * 100.0  # percent of equity
    return {
        "reach_1r_pct": p * 100.0,
        "exp_r_per_trade": exp_r_per_trade,
        "exp_equity_pct": exp_pct,
        "risk_pct": risk_pct,
        "n_trades": n_trades,
    }


def render_html_report(
    completed_dir: str | Path,
    *,
    split_path: str | Path = "specs/splits/v1.json",
    out_path: str | Path | None = None,
) -> Path:
    data = _load_completed(completed_dir)
    reach = data["reach"]
    meta = data["meta"]
    candidate = data["candidate"]
    split = SplitManifest.load(split_path)

    strategy_id = str(meta.get("strategy_id") or candidate.get("candidate_id") or data["dir"].name)
    title = str(candidate.get("title") or strategy_id)
    param_hash = str(meta.get("parameter_hash") or "")
    n = int(reach["n"].iloc[0])
    n_train_symbols = len(split.train_symbols)
    n_universe = len(split.universe)
    test_date = str(meta.get("completed_at_utc") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    # Human date
    try:
        test_date_human = datetime.strptime(test_date[:15], "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")
    except ValueError:
        test_date_human = test_date[:10]

    start = pd.Timestamp(split.overall_start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    end = pd.Timestamp(split.overall_end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    date_range = f"{start} → {end}"

    headline, verdict_para = _verdict(reach)
    impl = _expected_outcome_1r(reach)
    sign = "+" if impl["exp_equity_pct"] >= 0 else ""
    implication = (
        f"Trading this strategy at {impl['risk_pct']:.0f}% risk per trade over {impl['n_trades']} trades, "
        f"the expected outcome based on these results is {sign}{impl['exp_equity_pct']:.2f}% of equity "
        f"(model: each trade gains +1R with probability equal to the 1R reach rate "
        f"({impl['reach_1r_pct']:.1f}%), otherwise −1R)."
    )
    caveat = (
        "Caveat: this is an arithmetic expectation from training-sample reach rates only — "
        "not a forecast, not holdout-validated, and not financial advice. Past performance "
        "does not predict future results; costs, slippage, and regime change are ignored here."
    )

    rows_html = []
    for _, row in reach.sort_values("target_R").iterrows():
        diff = float(row["diff_pp"])
        cls = _cell_class(diff)
        rows_html.append(
            "<tr>"
            f"<td>{float(row['target_R']):.1f}R</td>"
            f"<td class='n'>{int(row['n']):,}</td>"
            f"<td>{float(row['reach_pct']):.1f}%</td>"
            f"<td>{float(row['random_walk_pct']):.1f}%</td>"
            f"<td class='{cls}'>{diff:+.1f}pp</td>"
            "</tr>"
        )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Audit report — {html.escape(strategy_id)}</title>
  <style>
    :root {{
      --ink: #14201a;
      --paper: #f4efe4;
      --muted: #5c6b62;
      --line: #c9c0b0;
      --pos: #1b7a4a;
      --pos-bg: #d9f0e3;
      --neg: #a31d2a;
      --neg-bg: #f7dce0;
      --coin: #6b6b6b;
      --coin-bg: #e8e6e1;
    }}
    @page {{ margin: 18mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 2.5rem 1.5rem 3rem;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      background: linear-gradient(165deg, #ebe4d6 0%, var(--paper) 40%, #e7efe8 100%);
      color: var(--ink); line-height: 1.45;
    }}
    .sheet {{
      max-width: 44rem; margin: 0 auto; background: rgba(255,252,246,0.92);
      border: 1px solid var(--line); padding: 2rem 2.1rem 1.5rem;
      box-shadow: 0 18px 40px rgba(20,32,26,0.08);
    }}
    header {{ border-bottom: 1px solid var(--line); padding-bottom: 1rem; margin-bottom: 1.5rem; }}
    header .name {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; }}
    header .meta {{ margin-top: 0.45rem; color: var(--muted); font-size: 0.92rem; }}
    header .meta span {{ margin-right: 1rem; white-space: nowrap; }}
    h2 {{
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em;
      color: var(--muted); margin: 1.75rem 0 0.65rem; font-weight: 700;
    }}
    .verdict-head {{
      font-size: 1.65rem; line-height: 1.25; font-weight: 700;
      margin: 0 0 0.75rem; letter-spacing: -0.02em;
    }}
    .verdict-p, .impl-p, .caveat, .rw-note {{ margin: 0 0 0.75rem; font-size: 1.02rem; }}
    .caveat {{ color: var(--muted); font-size: 0.95rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 0.75rem 0 0.5rem; font-size: 0.98rem; }}
    th, td {{ padding: 0.55rem 0.45rem; text-align: right; border-bottom: 1px solid var(--line); }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    td.n {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
    td.pos {{ background: var(--pos-bg); color: var(--pos); font-weight: 700; }}
    td.neg {{ background: var(--neg-bg); color: var(--neg); font-weight: 700; }}
    td.coin {{ background: var(--coin-bg); color: var(--coin); font-weight: 700; }}
    footer {{
      margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--line);
      font-size: 0.82rem; color: var(--muted); line-height: 1.4;
    }}
    @media print {{
      body {{ background: white; padding: 0; }}
      .sheet {{ box-shadow: none; border: none; max-width: none; }}
    }}
  </style>
</head>
<body>
  <article class="sheet">
    <header>
      <div class="name">{html.escape(title)}</div>
      <div class="meta">
        <span>id <strong>{html.escape(strategy_id)}</strong></span>
        <span>hash <strong>{html.escape(param_hash)}</strong></span>
        <span>tested <strong>{html.escape(test_date_human)}</strong></span>
        <span>n <strong>{n:,}</strong></span>
        <span>train symbols <strong>{n_train_symbols}</strong></span>
      </div>
    </header>

    <h2>Section 1 — The verdict</h2>
    <p class="verdict-head">{html.escape(headline)}</p>
    <p class="verdict-p">{html.escape(verdict_para)}</p>

    <h2>Section 2 — The table</h2>
    <p class="rw-note">The random-walk column is the chance a fair coin would hit the same
    profit multiple before ruin of the stop: for target T (in R), that rate is 1/(1+T).
    Diff is live reach minus that benchmark, in percentage points.</p>
    <table>
      <thead>
        <tr><th>Target</th><th>n</th><th>Reach</th><th>RW 1/(1+T)</th><th>Diff</th></tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>

    <h2>Section 3 — The implication</h2>
    <p class="impl-p">{html.escape(implication)}</p>
    <p class="caveat">{html.escape(caveat)}</p>

    <footer>
      Tested using reach-vs-random-walk benchmark across {n_universe} crypto perpetual pairs
      on Binance, {html.escape(date_range)}. Full methodology at
      github.com/stripedstrings/ht-backtest.
    </footer>
  </article>
</body>
</html>
"""

    out = Path(out_path) if out_path else data["dir"] / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out


def generate_report(strategy_id: str, *, candidates_root: str | Path = "data/candidates", **kwargs: Any) -> Path:
    completed = Path(candidates_root) / "completed" / strategy_id
    return render_html_report(completed, **kwargs)
