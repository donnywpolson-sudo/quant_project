from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.phase1_raw_contract import REQUIRED_DATASET


ALLOWED_AUDIT_SCHEMAS = {"trades", "mbp-1"}
AUDIT_REASON = "validate_whether_ohlcv_gap_has_trade_or_book_activity"


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_iso(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _timestamp_column(frame: pd.DataFrame) -> pd.Series:
    if "ts_event" in frame.columns:
        return pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    if "ts" in frame.columns:
        return pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(frame.index, utc=True, errors="coerce"), index=frame.index)
    raise ValueError("raw parquet missing timestamp column/index")


def _load_gap_audit(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"missing gap audit: {_relative_path(path)}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"unreadable gap audit: {exc}"]
    if not isinstance(payload, dict):
        return None, ["gap audit top-level JSON is not an object"]
    return payload, []


def _candidate_gaps(
    gap_audit: dict[str, Any],
    markets: list[str],
    years: list[int],
    max_windows: int,
) -> list[dict[str, Any]]:
    market_order = {market: idx for idx, market in enumerate(markets)}
    year_set = set(years)
    candidates: list[dict[str, Any]] = []
    for entry in gap_audit.get("entries", []):
        if not isinstance(entry, dict):
            continue
        market = str(entry.get("market", ""))
        year = int(entry.get("year", 0) or 0)
        if market not in market_order or year not in year_set:
            continue
        for gap in entry.get("largest_gaps", []):
            if not isinstance(gap, dict):
                continue
            candidates.append(
                {
                    "market": market,
                    "year": year,
                    "gap": gap,
                    "gap_size_minutes": int(gap.get("gap_size_minutes", 0) or 0),
                    "market_order": market_order[market],
                }
            )
    candidates.sort(
        key=lambda item: (
            -int(item["gap_size_minutes"]),
            int(item["market_order"]),
            int(item["year"]),
            str(item["gap"].get("first_synthetic_ts", "")),
        )
    )
    return candidates[:max_windows]


def _resolve_adjacent_raw_context(
    raw_path: Path,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
) -> tuple[dict[str, Any] | None, str | None]:
    if not raw_path.exists():
        return None, f"missing raw parquet: {_relative_path(raw_path)}"
    frame = pd.read_parquet(raw_path)
    if "instrument_id" not in frame.columns:
        return None, f"raw parquet missing instrument_id: {_relative_path(raw_path)}"
    ts = _timestamp_column(frame)
    work = frame.copy()
    work["_ts"] = ts
    work = work.dropna(subset=["_ts"]).sort_values("_ts", kind="mergesort")
    before = work[work["_ts"] < gap_start].tail(1)
    after = work[work["_ts"] > gap_end].head(1)
    adjacent = pd.concat([before, after], ignore_index=True)
    ids = pd.to_numeric(adjacent.get("instrument_id"), errors="coerce").dropna().astype("int64")
    unique_ids = sorted(set(int(value) for value in ids.tolist()))
    if len(unique_ids) != 1:
        return None, f"adjacent instrument_id unresolved: {_relative_path(raw_path)}"
    context = {
        "instrument_id": unique_ids[0],
        "raw_ohlcv_source_file": None,
        "raw_ohlcv_source_hash": None,
    }
    if "source_file" in adjacent.columns and adjacent["source_file"].notna().any():
        context["raw_ohlcv_source_file"] = str(adjacent["source_file"].dropna().iloc[0])
    if "source_sha256" in adjacent.columns and adjacent["source_sha256"].notna().any():
        context["raw_ohlcv_source_hash"] = str(adjacent["source_sha256"].dropna().iloc[0])
    return context, None


def _build_task(
    candidate: dict[str, Any],
    schema: str,
    raw_root: Path,
    buffer_minutes: int,
    max_window_minutes: int,
) -> tuple[dict[str, Any] | None, str | None]:
    gap = candidate["gap"]
    market = str(candidate["market"])
    year = int(candidate["year"])
    first = pd.Timestamp(str(gap.get("first_synthetic_ts")))
    last = pd.Timestamp(str(gap.get("last_synthetic_ts")))
    if first.tzinfo is None:
        first = first.tz_localize("UTC")
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    query_start = first - pd.Timedelta(minutes=buffer_minutes)
    query_end = last + pd.Timedelta(minutes=buffer_minutes + 1)
    window_minutes = int((query_end - query_start).total_seconds() // 60)
    if window_minutes > max_window_minutes:
        return (
            None,
            f"{market} {year} {schema}: window {window_minutes}m exceeds max {max_window_minutes}m",
        )
    raw_path = raw_root / market / f"{year}.parquet"
    context, failure = _resolve_adjacent_raw_context(raw_path, first, last)
    if failure is not None or context is None:
        return None, f"{market} {year} {schema}: {failure}"
    return (
        {
            "market": market,
            "year": year,
            "schema": schema,
            "dataset": REQUIRED_DATASET,
            "stype_in": "instrument_id",
            "instrument_id": context["instrument_id"],
            "start": _utc_iso(query_start),
            "end": _utc_iso(query_end),
            "pre_buffer_minutes": buffer_minutes,
            "post_buffer_minutes": buffer_minutes,
            "source_gap_timestamps": {
                "synthetic_gap_id": gap.get("synthetic_gap_id"),
                "first_synthetic_ts": _utc_iso(first),
                "last_synthetic_ts": _utc_iso(last),
                "gap_size_minutes": gap.get("gap_size_minutes"),
                "synthetic_rows": gap.get("synthetic_rows"),
            },
            "raw_ohlcv_source_file": context["raw_ohlcv_source_file"],
            "raw_ohlcv_source_hash": context["raw_ohlcv_source_hash"],
            "reason": AUDIT_REASON,
        },
        None,
    )


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    schemas = [str(schema) for schema in args.schemas]
    unsupported = sorted(set(schemas) - ALLOWED_AUDIT_SCHEMAS)
    failures = [f"unsupported audit schemas: {unsupported}"] if unsupported else []
    gap_audit, load_failures = _load_gap_audit(Path(args.gap_audit_json))
    failures.extend(load_failures)

    tasks: list[dict[str, Any]] = []
    if gap_audit is not None and not unsupported:
        candidates = _candidate_gaps(
            gap_audit,
            [str(market) for market in args.markets],
            [int(year) for year in args.years],
            int(args.max_windows),
        )
        if not candidates:
            failures.append("no matching largest gaps found")
        for candidate in candidates:
            for schema in schemas:
                task, failure = _build_task(
                    candidate,
                    schema,
                    Path(args.raw_root),
                    int(args.buffer_minutes),
                    int(args.max_window_minutes),
                )
                if failure:
                    failures.append(failure)
                elif task is not None:
                    tasks.append(task)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "FAIL" if failures else "PASS",
        "dry_run_only": True,
        "dataset": REQUIRED_DATASET,
        "gap_audit_json": _relative_path(Path(args.gap_audit_json)),
        "raw_root": _relative_path(Path(args.raw_root)),
        "markets": [str(market) for market in args.markets],
        "years": [int(year) for year in args.years],
        "schemas": schemas,
        "allowed_schemas": sorted(ALLOWED_AUDIT_SCHEMAS),
        "max_windows": int(args.max_windows),
        "max_window_minutes": int(args.max_window_minutes),
        "buffer_minutes": int(args.buffer_minutes),
        "failures": failures,
        "tasks": tasks,
    }


def write_plan(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-audit-json", default="reports/pipeline_audit/raw_session_gap_audit.json")
    parser.add_argument("--markets", nargs="+", required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--schemas", nargs="+", default=["trades"])
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--max-window-minutes", type=int, default=90)
    parser.add_argument("--buffer-minutes", type=int, default=0)
    parser.add_argument("--plan-out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.max_windows <= 0:
        raise SystemExit("--max-windows must be > 0")
    if args.max_window_minutes <= 0:
        raise SystemExit("--max-window-minutes must be > 0")
    if args.buffer_minutes < 0:
        raise SystemExit("--buffer-minutes must be >= 0")
    plan = build_plan(args)
    write_plan(plan, Path(args.plan_out))
    if plan["status"] != "PASS":
        print(f"FAIL tick/source gap audit plan: failures={len(plan['failures'])}")
        return 1
    print(f"PASS tick/source gap audit plan: tasks={len(plan['tasks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
