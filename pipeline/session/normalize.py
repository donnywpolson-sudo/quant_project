from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from pipeline.common.io_safe import atomic_write_json, write_csv_rows
from pipeline.data_gate.manifest import build_data_manifest


SESSION_KEYS = {
    "timezone",
    "input_timezone",
    "week_start_day",
    "week_start_time",
    "week_end_day",
    "week_end_time",
    "daily_break",
    "daily_breaks",
    "holiday_calendar",
    "holiday_early_close_time",
    "closed_dates",
    "early_closes",
    "late_opens",
    "session_calendar_accuracy",
    "approximate_calendar_failure_mode",
    "approximate_reason",
    "session_start_local",
    "session_end_local",
    "session_break_start_local",
    "session_break_end_local",
}


def load_session_config(sessions_path: str | Path, market: str = "") -> dict:
    """Read either the legacy simple session shape or merged raw validation config."""
    path = Path(sessions_path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "markets" in raw and market in (raw.get("markets") or {}):
        return {k: v for k, v in (raw["markets"][market] or {}).items() if k in SESSION_KEYS}
    if "default" in raw:
        return {k: v for k, v in (raw["default"] or {}).items() if k in SESSION_KEYS}
    return {k: v for k, v in raw.items() if k in SESSION_KEYS}


def normalize_session_df(df: pl.DataFrame, market: str = "", session_config: dict | None = None) -> pl.DataFrame:
    if "ts_event" not in df.columns:
        raise ValueError("missing ts_event")
    session_config = session_config or {}
    out = df.sort("ts_event").unique("ts_event", keep="first")
    if out["ts_event"].dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        session_expr = (pl.col("ts_event") // 1440).cast(pl.Utf8)
    else:
        session_expr = pl.col("ts_event").dt.date().cast(pl.Utf8)
    return out.with_columns(
        session_expr.alias("session_id"),
        session_expr.alias("session_date"),
        pl.lit(market).alias("market"),
        pl.lit(str(session_config.get("timezone", ""))).alias("session_timezone"),
        pl.lit(str(session_config.get("session_calendar_accuracy", ""))).alias("session_calendar_accuracy"),
    )


def session_normalize_root(in_root: str | Path = "data/validated", out_root: str | Path = "data/session_normalized", sessions_path: str | Path = "configs/raw_data_validation.yaml") -> dict:
    in_root = Path(in_root)
    out_root = Path(out_root)
    rows = []
    failures = []
    for p in sorted(in_root.glob("*/*.parquet")):
        try:
            out = out_root / p.parent.name / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            session_cfg = load_session_config(sessions_path, p.parent.name)
            normalize_session_df(pl.read_parquet(p), p.parent.name, session_cfg).write_parquet(out)
            rows.append({
                "input": str(p),
                "output": str(out),
                "status": "PASS",
                "timezone": str(session_cfg.get("timezone", "")),
                "session_calendar_accuracy": str(session_cfg.get("session_calendar_accuracy", "")),
            })
        except Exception as exc:
            failures.append(str(p))
            rows.append({"input": str(p), "output": "", "status": "FAIL", "note": str(exc)})
    report = {"status": "FAIL" if failures else "PASS", "sessions_path": str(sessions_path), "files": rows, "failures": failures}
    atomic_write_json("reports/session_normalization/session_normalization_report.json", report)
    write_csv_rows("reports/session_normalization/session_normalization_summary.csv", rows or [{"input": "", "output": "", "status": "WARN", "note": "no files"}])
    build_data_manifest(out_root, stage="session_normalized")
    return report
