from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import SplitManifest, build_split_manifest
from ht_backtest.data.universe import pull_and_validate_universe
from ht_backtest.data.validator import validate_ohlcv
from ht_backtest.reports.batch_runner import load_batch_config, run_batch
from ht_backtest.reports.reach import format_reach_table, reach_vs_random_walk
from ht_backtest.reports.universe_report import generate_pooled_trades
from ht_backtest.strategies.registry import get_strategy, list_strategies


def _parse_date(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def cmd_download(args: argparse.Namespace) -> None:
    dl = OHLCVDownloader(exchange_id=args.exchange, cache_dir=args.cache_dir)
    since_ms = _parse_date(args.since)
    until_ms = _parse_date(args.until) if args.until else None
    result = dl.download(args.symbol, args.timeframe, since_ms, until_ms)
    print(
        f"{result.symbol} {result.timeframe}: {result.rows} bars cached "
        f"[{result.start} .. {result.end}], {result.fetched_new_bars} new bars fetched this run"
    )


def cmd_validate(args: argparse.Namespace) -> None:
    dl = OHLCVDownloader(exchange_id=args.exchange, cache_dir=args.cache_dir)
    since_ms = _parse_date(args.since)
    until_ms = _parse_date(args.until) if args.until else int(pd.Timestamp.utcnow().timestamp() * 1000)
    df = dl.cached_range(args.symbol, args.timeframe, since_ms, until_ms)
    if df.empty:
        print("No cached data for this range. Run `download` first.", file=sys.stderr)
        sys.exit(1)
    report = validate_ohlcv(df, args.symbol, args.timeframe)
    print(report.summary())
    if args.json_out:
        payload = {
            "symbol": report.symbol,
            "timeframe": report.timeframe,
            "total_bars": report.total_bars,
            "duplicates": len(report.duplicates),
            "gaps": report.gaps.to_dict(orient="records"),
            "missing_bars": report.missing_bar_count(),
            "impossible_ohlc": report.impossible_ohlc[["timestamp", "open", "high", "low", "close"]].to_dict(orient="records"),
            "non_positive_price": report.non_positive_price[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records"),
            "extreme_jumps": report.extreme_jumps[["timestamp", "close", "log_return", "zscore"]].to_dict(orient="records"),
            "outlier_wicks": report.outlier_wicks[["timestamp", "high", "low", "wick_up", "wick_dn", "atr"]].to_dict(orient="records"),
            "dead_runs": report.dead_runs.to_dict(orient="records"),
            "dead_bars": report.dead_bar_count(),
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote detailed report to {args.json_out}")


def cmd_universe(args: argparse.Namespace) -> None:
    df = pull_and_validate_universe(
        universe_path=args.universe_file,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        cache_dir=args.cache_dir,
        reports_dir=args.reports_dir,
    )
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.reports_dir) / f"universe_summary_{args.timeframe}.csv"
    df.to_csv(csv_path, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    print(f"\nWrote {csv_path}")


def cmd_split_build(args: argparse.Namespace) -> None:
    manifest = build_split_manifest(
        universe_path=args.universe_file,
        timeframe=args.timeframe,
        seed=args.seed,
        symbol_holdout_fraction=args.symbol_holdout_fraction,
        date_holdout_fraction=args.date_holdout_fraction,
        exchange_id=args.exchange,
        cache_dir=args.cache_dir,
    )
    out_path = Path(args.splits_dir) / f"{args.name}.json"
    manifest.save(out_path)
    print(f"Wrote split manifest '{args.name}' to {out_path}")
    print(f"  seed                 : {manifest.seed}")
    print(f"  universe             : {len(manifest.universe)} symbols")
    print(f"  holdout symbols ({len(manifest.holdout_symbols)}) : {', '.join(manifest.holdout_symbols)}")
    print(f"  train symbols   ({len(manifest.train_symbols)}) : {', '.join(manifest.train_symbols)}")
    print(f"  overall range        : {manifest.to_dict()['overall_start']} .. {manifest.to_dict()['overall_end']}")
    print(f"  date holdout cutoff  : {manifest.to_dict()['date_holdout_start']}  (most recent {args.date_holdout_fraction:.0%} of span)")


def cmd_batch(args: argparse.Namespace) -> None:
    config = load_batch_config(args.config)
    print(f"batch config: {args.config}")
    print(f"strategies ({len(config.strategies)}): {', '.join(config.strategies)}")
    print(f"split={config.split}  workers={config.workers}  timeframe={config.timeframe}")
    run_batch(config)


def cmd_memory(args: argparse.Namespace) -> None:
    from ht_backtest.memory.hypothesis_log import (
        format_prior_results_summary,
        prior_for_category,
        query_log,
    )

    if args.category:
        print(prior_for_category(args.category, path=args.log))
        if args.signal_type:
            q = query_log(category=args.category, signal_type=args.signal_type, path=args.log)
            print()
            print(q.to_string(index=False) if not q.empty else "(no matching rows)")
        return
    print(format_prior_results_summary(args.log))


def cmd_intake(args: argparse.Namespace) -> None:
    from ht_backtest.discovery.pipeline import run_intake_pipeline

    result = run_intake_pipeline(
        args.intake_file,
        fixture=args.fixture,
        skip_dry_count=args.skip_dry_count,
        max_symbols=args.max_symbols,
        candidates_root=args.candidates_dir,
        split_path=args.split_path,
        cache_dir=args.cache_dir,
        model=args.model,
    )
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "candidate"}, indent=2))
    raise SystemExit(0 if result.get("accepted") else 2)


def cmd_intake_serve(args: argparse.Namespace) -> None:
    from ht_backtest.discovery.web_form import serve

    serve(host=args.host, port=args.port, fixture_default=args.fixture)


def cmd_worker(args: argparse.Namespace) -> None:
    from ht_backtest.discovery.worker import run_queue

    if not args.run_queue and not args.loop:
        print("Specify --run-queue (once) or --loop", file=sys.stderr)
        raise SystemExit(2)
    results = run_queue(
        root=args.candidates_dir,
        limit=args.limit,
        loop=args.loop,
        interval_s=args.interval,
        split_path=args.split_path,
        cache_dir=args.cache_dir,
        workers=args.workers,
        mfe_win=args.mfe_win,
        min_edge_pp=args.min_edge_pp,
        min_n=args.min_n,
        skip_dry_count=args.skip_dry_count,
    )
    n_ok = sum(1 for r in results if r.get("status") == "completed")
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    print(f"\nworker done: completed={n_ok} failed={n_fail}")
    raise SystemExit(0 if n_fail == 0 else 1)


def cmd_run(args: argparse.Namespace) -> None:
    strategy = get_strategy(args.strategy)
    meta = strategy.metadata()
    split_path = Path(args.splits_dir) / f"{args.split}.json"
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}", file=sys.stderr)
        sys.exit(1)
    split = SplitManifest.load(split_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / f"{meta.id}_{args.split}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_meta = {
        "strategy": meta.to_dict(),
        "split": args.split,
        "split_path": str(split_path),
        "timeframe": args.timeframe,
        "exchange": args.exchange,
        "cache_dir": args.cache_dir,
        "mfe_win": args.mfe_win,
        "created_at_utc": stamp,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)

    print(f"strategy : {meta.id} v{meta.version}  hash={meta.parameter_hash}")
    print(f"desc     : {meta.description}")
    print(f"split    : {args.split}  ({split_path})")
    print(f"out      : {out_dir}")
    print()

    trades, timings = generate_pooled_trades(
        strategy=strategy,
        split=split,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        cache_dir=args.cache_dir,
        mfe_win=args.mfe_win,
        workers=args.workers,
        strategy_name=args.strategy,
        split_path=split_path,
        use_primitive_cache=not args.no_primitive_cache,
    )
    trades_path = out_dir / "trades.parquet"
    trades.to_parquet(trades_path, index=False)
    print(f"\nWrote {trades_path}  ({len(trades)} trades)")

    train = trades[trades["split"] == "train"] if not trades.empty else trades
    reach = reach_vs_random_walk(train)
    reach_path = out_dir / "training_reach_table.csv"
    reach.to_csv(reach_path, index=False)
    print(f"Wrote {reach_path}")
    print()
    print(format_reach_table(reach, f"TRAIN reach vs RW — {meta.id}"))
    print(f"\nHoldout rows present in artifact but not scored here: {(trades['split'] == 'holdout').sum() if not trades.empty else 0}")
    print("\nstage timings:")
    for k, v in timings.to_dict().items():
        if k.endswith("_s") or k in ("symbols", "trades"):
            print(f"  {k}: {v}")

    artifact_meta = {
        "strategy": meta.to_dict(),
        "split": args.split,
        "split_path": str(split_path),
        "timeframe": args.timeframe,
        "exchange": args.exchange,
        "cache_dir": args.cache_dir,
        "mfe_win": args.mfe_win,
        "created_at_utc": stamp,
        "stage_timings": timings.to_dict(),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ht-backtest")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Fetch and cache OHLCV bars")
    p_dl.add_argument("symbol")
    p_dl.add_argument("timeframe")
    p_dl.add_argument("--since", required=True, help="e.g. 2019-01-01")
    p_dl.add_argument("--until", default=None, help="defaults to now")
    p_dl.add_argument("--exchange", default="binanceusdm")
    p_dl.add_argument("--cache-dir", default="data/raw")
    p_dl.set_defaults(func=cmd_download)

    p_val = sub.add_parser("validate", help="Validate cached OHLCV bars")
    p_val.add_argument("symbol")
    p_val.add_argument("timeframe")
    p_val.add_argument("--since", required=True)
    p_val.add_argument("--until", default=None)
    p_val.add_argument("--exchange", default="binanceusdm")
    p_val.add_argument("--cache-dir", default="data/raw")
    p_val.add_argument("--json-out", default=None)
    p_val.set_defaults(func=cmd_validate)

    p_uni = sub.add_parser("universe", help="Download + validate the full symbol universe")
    p_uni.add_argument("timeframe")
    p_uni.add_argument("--universe-file", default="specs/universe.json")
    p_uni.add_argument("--exchange", default="binanceusdm")
    p_uni.add_argument("--cache-dir", default="data/raw")
    p_uni.add_argument("--reports-dir", default="data/reports")
    p_uni.set_defaults(func=cmd_universe)

    p_split = sub.add_parser("split-build", help="Build the structural train/holdout split manifest")
    p_split.add_argument("name", help="split name, e.g. 'v1' -- report commands must reference this explicitly")
    p_split.add_argument("timeframe")
    p_split.add_argument("--universe-file", default="specs/universe.json")
    p_split.add_argument("--seed", type=int, required=True)
    p_split.add_argument("--symbol-holdout-fraction", type=float, default=0.30)
    p_split.add_argument("--date-holdout-fraction", type=float, default=0.20)
    p_split.add_argument("--exchange", default="binanceusdm")
    p_split.add_argument("--cache-dir", default="data/raw")
    p_split.add_argument("--splits-dir", default="specs/splits")
    p_split.set_defaults(func=cmd_split_build)

    p_run = sub.add_parser(
        "run",
        help="Run a registered strategy over a locked split; write reach-vs-RW train table + metadata",
    )
    p_run.add_argument(
        "--strategy",
        required=True,
        help=f"registered strategy id ({', '.join(list_strategies())})",
    )
    p_run.add_argument("--split", required=True, help="split manifest name, e.g. v1")
    p_run.add_argument("--timeframe", default="15m")
    p_run.add_argument("--exchange", default="binanceusdm")
    p_run.add_argument("--cache-dir", default="data/raw")
    p_run.add_argument("--splits-dir", default="specs/splits")
    p_run.add_argument("--out-dir", default="data/runs")
    p_run.add_argument("--mfe-win", type=int, default=100)
    p_run.add_argument("--workers", type=int, default=1, help="process-pool size over symbols")
    p_run.add_argument(
        "--no-primitive-cache",
        action="store_true",
        help="recompute ATR/sessions/swings every time (for profiling cold path)",
    )
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser(
        "batch",
        help="Run multiple registered strategies from a YAML queue; write train comparison vs RW",
    )
    p_batch.add_argument("config", help="path to batch YAML, e.g. specs/batch/example_10.yaml")
    p_batch.set_defaults(func=cmd_batch)

    p_mem = sub.add_parser(
        "memory",
        help="Show hypothesis memory: exhausted vs unexplored categories (query before new strategies)",
    )
    p_mem.add_argument("--log", default=None, help="path to hypothesis_log.csv (default data/memory/...)")
    p_mem.add_argument("--category", default=None, help="filter: timing|volume|range|cross-asset|pattern")
    p_mem.add_argument("--signal-type", default=None, help="optional signal_type inside key_parameters")
    p_mem.set_defaults(func=cmd_memory)

    p_intake = sub.add_parser(
        "intake",
        help="Audit intake → YAML candidate (Claude or --fixture) → dry-count → queue/reject",
    )
    p_intake.add_argument("intake_file", help="YAML/JSON intake file (see specs/candidates_schema/examples/)")
    p_intake.add_argument(
        "--fixture",
        action="store_true",
        help="offline deterministic translator (no ANTHROPIC_API_KEY)",
    )
    p_intake.add_argument("--skip-dry-count", action="store_true", help="translate+risk only (dev)")
    p_intake.add_argument("--max-symbols", type=int, default=None, help="limit train symbols for dry-count")
    p_intake.add_argument("--candidates-dir", default="data/candidates")
    p_intake.add_argument("--split-path", default="specs/splits/v1.json")
    p_intake.add_argument("--cache-dir", default="data/raw")
    p_intake.add_argument("--model", default=None, help="Anthropic model id override")
    p_intake.set_defaults(func=cmd_intake)

    p_web = sub.add_parser("intake-serve", help="Local web form for audit intake (stdlib HTTP)")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument("--fixture", action="store_true", help="default form submissions to fixture translator")
    p_web.set_defaults(func=cmd_intake_serve)

    p_worker = sub.add_parser(
        "worker",
        help="Process data/candidates/queued → compile → train → memory → completed/failed",
    )
    p_worker.add_argument("--run-queue", action="store_true", help="process queued candidates once and exit")
    p_worker.add_argument("--loop", action="store_true", help="poll the queue on an interval")
    p_worker.add_argument("--interval", type=float, default=60.0, help="seconds between polls in --loop mode")
    p_worker.add_argument("--limit", type=int, default=None, help="max candidates this run")
    p_worker.add_argument("--candidates-dir", default="data/candidates")
    p_worker.add_argument("--split-path", default="specs/splits/v1.json")
    p_worker.add_argument("--cache-dir", default="data/raw")
    p_worker.add_argument("--workers", type=int, default=4)
    p_worker.add_argument("--mfe-win", type=int, default=100)
    p_worker.add_argument("--min-edge-pp", type=float, default=5.0)
    p_worker.add_argument("--min-n", type=int, default=200)
    p_worker.add_argument("--skip-dry-count", action="store_true")
    p_worker.set_defaults(func=cmd_worker)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
