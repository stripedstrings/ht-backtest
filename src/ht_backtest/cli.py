from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ht_backtest.data.downloader import OHLCVDownloader
from ht_backtest.data.split import build_split_manifest
from ht_backtest.data.universe import pull_and_validate_universe
from ht_backtest.data.validator import validate_ohlcv


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
