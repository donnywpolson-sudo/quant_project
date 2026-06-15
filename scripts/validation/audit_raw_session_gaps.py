from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.phase2_causal_base.build_causal_base_data import (
    _session_metadata,
    load_session_calendar,
)


SESSION_EDGE_SHARE_THRESHOLD = 0.5


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_timestamps(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_parquet(path)
    if "ts" in frame.columns:
        ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    elif "ts_event" in frame.columns:
        ts = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        ts = pd.Series(pd.to_datetime(frame.index, utc=True, errors="coerce"), index=frame.index)
    else:
        raise ValueError(f"missing timestamp column/index in {_relative_path(path)}")
    return frame, ts


def _top_counts(series: pd.Series, name: str, limit: int = 10) -> list[dict[str, Any]]:
    counts = series.dropna().value_counts().head(limit)
    return [{name: str(index), "rows": int(rows)} for index, rows in counts.items()]


def _gap_size_counts(synthetic: pd.DataFrame) -> list[dict[str, int]]:
    if "synthetic_gap_size_minutes" not in synthetic.columns:
        return []
    sizes = pd.to_numeric(synthetic["synthetic_gap_size_minutes"], errors="coerce")
    gap_frame = pd.DataFrame({"gap_size_minutes": sizes})
    if "synthetic_gap_id" in synthetic.columns:
        gap_frame["synthetic_gap_id"] = synthetic["synthetic_gap_id"].to_numpy()
        gap_frame = gap_frame.dropna(subset=["synthetic_gap_id", "gap_size_minutes"])
        gap_frame = gap_frame.drop_duplicates(["synthetic_gap_id", "gap_size_minutes"])
    else:
        gap_frame = gap_frame.dropna(subset=["gap_size_minutes"])
    counts = Counter(int(value) for value in gap_frame["gap_size_minutes"])
    return [
        {"gap_size_minutes": size, "gaps": count}
        for size, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _session_bucket(row: pd.Series) -> str:
    if not bool(row.get("inside_session", False)):
        return "outside_configured_session"
    since_open = row.get("minutes_since_session_open")
    until_close = row.get("minutes_until_session_close")
    hour = int(row["ct_hour"])
    if pd.notna(since_open) and float(since_open) < 60:
        return "first_60m_after_configured_open"
    if pd.notna(until_close) and float(until_close) <= 60:
        return "last_60m_before_configured_close"
    if hour == 18:
        return "configured_evening_17_18_ct"
    if hour >= 19 or hour < 5:
        return "overnight_19_05_ct"
    if hour in {6, 7}:
        return "pre_us_06_07_ct"
    if 8 <= hour <= 13:
        return "us_day_08_13_ct"
    if hour in {14, 15}:
        return "late_day_14_15_ct"
    return "other_in_session"


def _bucket_synthetic_timestamps(
    synthetic_ts: pd.Series,
    market: str,
    session_config: Path,
) -> pd.DataFrame:
    if synthetic_ts.empty:
        return pd.DataFrame()
    calendar = load_session_calendar(market, session_config)
    metadata = _session_metadata(synthetic_ts.reset_index(drop=True), calendar)
    local = synthetic_ts.reset_index(drop=True).dt.tz_convert(calendar.timezone)
    metadata["ct_hour"] = local.dt.hour.astype("int64")
    metadata["session_bucket"] = metadata.apply(_session_bucket, axis=1)
    metadata["ts"] = synthetic_ts.reset_index(drop=True)
    return metadata


def _largest_gaps(synthetic: pd.DataFrame, synthetic_ts: pd.Series, metadata: pd.DataFrame) -> list[dict[str, Any]]:
    if "synthetic_gap_id" not in synthetic.columns or "synthetic_gap_size_minutes" not in synthetic.columns:
        return []
    gap_frame = pd.DataFrame(
        {
            "synthetic_gap_id": synthetic["synthetic_gap_id"].to_numpy(),
            "synthetic_gap_size_minutes": pd.to_numeric(
                synthetic["synthetic_gap_size_minutes"], errors="coerce"
            ).to_numpy(),
            "ts": synthetic_ts.to_numpy(),
            "session_date": metadata.get("session_date", pd.Series(pd.NA, index=metadata.index)).to_numpy(),
        }
    ).dropna(subset=["synthetic_gap_id", "synthetic_gap_size_minutes", "ts"])
    if gap_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for gap_id, group in gap_frame.groupby("synthetic_gap_id", sort=False):
        size = int(group["synthetic_gap_size_minutes"].max())
        rows.append(
            {
                "synthetic_gap_id": str(gap_id),
                "gap_size_minutes": size,
                "synthetic_rows": int(len(group)),
                "first_synthetic_ts": pd.Timestamp(group["ts"].min()).isoformat(),
                "last_synthetic_ts": pd.Timestamp(group["ts"].max()).isoformat(),
                "session_date": str(group["session_date"].dropna().iloc[0])
                if group["session_date"].notna().any()
                else None,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["gap_size_minutes"]), row["first_synthetic_ts"]))[:10]


def audit_market_year(
    market: str,
    year: int,
    raw_root: Path,
    causal_root: Path,
    session_config: Path,
) -> dict[str, Any]:
    raw_path = raw_root / market / f"{year}.parquet"
    causal_path = causal_root / market / f"{year}.parquet"
    failures: list[str] = []
    if not raw_path.exists():
        failures.append(f"missing raw input: {_relative_path(raw_path)}")
    if not causal_path.exists():
        failures.append(f"missing causal input: {_relative_path(causal_path)}")
    if not session_config.exists():
        failures.append(f"missing session config: {_relative_path(session_config)}")
    if failures:
        return {
            "market": market,
            "year": year,
            "status": "FAIL",
            "failures": failures,
            "paths": {
                "raw": _relative_path(raw_path),
                "causal": _relative_path(causal_path),
            },
        }

    raw, raw_ts = _read_timestamps(raw_path)
    causal, causal_ts = _read_timestamps(causal_path)
    if "is_synthetic" not in causal.columns:
        return {
            "market": market,
            "year": year,
            "status": "FAIL",
            "failures": [f"missing is_synthetic column: {_relative_path(causal_path)}"],
            "paths": {
                "raw": _relative_path(raw_path),
                "causal": _relative_path(causal_path),
            },
        }

    synthetic_mask = causal["is_synthetic"].fillna(False).astype(bool)
    synthetic = causal.loc[synthetic_mask].copy()
    synthetic_ts = causal_ts.loc[synthetic_mask]
    raw_timestamp_set = set(raw_ts.dropna().astype("int64").tolist())
    synthetic_timestamp_values = synthetic_ts.dropna().astype("int64")
    present_mask = synthetic_timestamp_values.isin(raw_timestamp_set)
    present_count = int(present_mask.sum())
    missing_count = int(len(synthetic_timestamp_values) - present_count)
    metadata = _bucket_synthetic_timestamps(synthetic_ts, market, session_config)

    session_bucket_counts = _top_counts(metadata.get("session_bucket", pd.Series(dtype="string")), "bucket")
    session_edge_rows = sum(
        row["rows"]
        for row in session_bucket_counts
        if row["bucket"] in {"first_60m_after_configured_open", "last_60m_before_configured_close"}
    )
    edge_share = float(session_edge_rows / len(synthetic_ts)) if len(synthetic_ts) else 0.0
    if missing_count and not present_count:
        raw_gap_call = "confirmed_absent_from_raw_parquet"
    elif present_count:
        raw_gap_call = "mixed_presence_in_raw_parquet"
    else:
        raw_gap_call = "no_synthetic_rows"
    session_template_call = (
        "not_primarily_session_edge_under_local_config"
        if edge_share <= SESSION_EDGE_SHARE_THRESHOLD
        else "primarily_session_edge_under_local_config"
    )

    return {
        "market": market,
        "year": year,
        "status": "PASS",
        "failures": [],
        "paths": {
            "raw": _relative_path(raw_path),
            "causal": _relative_path(causal_path),
        },
        "raw_gap_call": raw_gap_call,
        "session_template_call": session_template_call,
        "validation_call": "requires_tick_or_source_validation_before_semantics_change",
        "raw_rows": int(len(raw)),
        "synthetic_rows": int(len(synthetic)),
        "synthetic_timestamps_missing_from_raw": missing_count,
        "synthetic_timestamps_present_in_raw": present_count,
        "share_synthetic_rows_first_or_last_60m_configured_session": edge_share,
        "gap_size_buckets": _gap_size_counts(synthetic),
        "ct_hour_buckets": _top_counts(
            metadata.get("ct_hour", pd.Series(dtype="int64")).astype("string"), "ct_hour"
        ),
        "session_buckets": session_bucket_counts,
        "top_session_dates": _top_counts(
            metadata.get("session_date", pd.Series(dtype="string")), "session_date", limit=5
        ),
        "largest_gaps": _largest_gaps(synthetic, synthetic_ts, metadata),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = Path(args.raw_root)
    causal_root = Path(args.causal_root)
    session_config = Path(args.session_config)
    entries = [
        audit_market_year(market, int(year), raw_root, causal_root, session_config)
        for market in args.markets
        for year in args.years
    ]
    failures = [
        f"{entry['market']} {entry['year']}: {failure}"
        for entry in entries
        for failure in entry.get("failures", [])
    ]
    summary = [
        {
            "market": entry["market"],
            "year": entry["year"],
            "status": entry["status"],
            "raw_gap_call": entry.get("raw_gap_call"),
            "session_template_call": entry.get("session_template_call"),
            "validation_call": entry.get("validation_call"),
            "synthetic_rows": entry.get("synthetic_rows"),
            "synthetic_timestamps_missing_from_raw": entry.get(
                "synthetic_timestamps_missing_from_raw"
            ),
            "synthetic_timestamps_present_in_raw": entry.get(
                "synthetic_timestamps_present_in_raw"
            ),
        }
        for entry in entries
    ]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "method": "compare Phase 2 synthetic rows to raw parquet timestamps and bucket gaps by configured session metadata",
        "raw_root": _relative_path(raw_root),
        "causal_root": _relative_path(causal_root),
        "session_config": _relative_path(session_config),
        "summary": summary,
        "entries": entries,
    }


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Raw Session Gap Audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        f"Status: `{report['status']}`",
        "",
        "| Market | Year | Status | Synthetic rows | Missing from raw | Present in raw | Raw gap call | Session call |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in report["summary"]:
        lines.append(
            "| `{market}` | {year} | `{status}` | {synthetic_rows} | "
            "{missing} | {present} | `{raw_gap_call}` | `{session_call}` |".format(
                market=row["market"],
                year=row["year"],
                status=row["status"],
                synthetic_rows=row.get("synthetic_rows", "NA"),
                missing=row.get("synthetic_timestamps_missing_from_raw", "NA"),
                present=row.get("synthetic_timestamps_present_in_raw", "NA"),
                raw_gap_call=row.get("raw_gap_call") or "NA",
                session_call=row.get("session_template_call") or "NA",
            )
        )
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in report["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--causal-root", default="data/causally_gated_normalized")
    parser.add_argument("--session-config", default="configs/market_sessions.yaml")
    parser.add_argument("--json-out", default="reports/pipeline_audit/raw_session_gap_audit.json")
    parser.add_argument("--md-out", default="reports/pipeline_audit/raw_session_gap_audit.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(args)
    write_json_report(report, Path(args.json_out))
    write_markdown_report(report, Path(args.md_out))
    if report["status"] != "PASS":
        print(f"FAIL raw session gap audit: failures={len(report['failures'])}")
        return 1
    print(f"PASS raw session gap audit: entries={len(report['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
