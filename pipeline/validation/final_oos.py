from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pipeline.validation.final_lineage import file_sha256, lineage_path, read_json_or_sidecar, write_lineage


FINAL_WFA_BACKTEST = Path("reports/validation/stage_24_final_wfa_backtest_results.parquet")
FINAL_OOS_PREDICTIONS = Path("reports/validation/stage_25_final_oos_predictions.parquet")

FINAL_OOS_REQUIRED_COLUMNS = (
    "run_id",
    "profile",
    "symbol",
    "split",
    "timestamp",
    "prediction",
    "target_15m_ret",
)


def validate_final_oos_predictions(
    path: str | Path = FINAL_OOS_PREDICTIONS,
    *,
    target_col: str = "target_15m_ret",
    expected_symbols: list[str] | None = None,
    expected_splits: int | None = None,
    source_path: str | Path = FINAL_WFA_BACKTEST,
) -> dict[str, Any]:
    path = Path(path)
    required = [c if c != "target_15m_ret" else target_col for c in FINAL_OOS_REQUIRED_COLUMNS]
    if not path.exists():
        return _status("MISSING", path, [], required, "artifact missing")
    try:
        schema = pl.scan_parquet(path).collect_schema()
        cols = schema.names()
    except Exception as exc:
        return _status("FAIL", path, [], required, f"unreadable parquet: {exc}")
    missing = [c for c in required if c not in cols]
    if missing:
        return _status("FAIL", path, cols, required, "missing required columns: " + ",".join(missing))
    try:
        lf = pl.scan_parquet(path)
        rows = int(lf.select(pl.len().alias("rows")).collect()["rows"][0] or 0)
    except Exception:
        rows = 0
    if rows <= 0:
        return _status("FAIL", path, cols, required, "row_count=0")
    failures: list[str] = []
    try:
        nonnull = lf.filter(pl.col("prediction").is_not_null()).select(pl.len().alias("rows")).collect()["rows"][0]
        if int(nonnull or 0) != rows:
            failures.append(f"prediction null rows={rows - int(nonnull or 0)}")
        actual_symbols = sorted(str(x) for x in lf.select("symbol").unique().collect()["symbol"].to_list())
        actual_splits = sorted(str(x) for x in lf.select("split").unique().collect()["split"].to_list())
        keys = lf.select(["symbol", "split"]).unique().collect()
        actual_key_count = keys.height
    except Exception as exc:
        return _status("FAIL", path, cols, required, f"coverage validation error: {exc}", row_count=rows)
    if expected_symbols:
        missing_symbols = sorted(set(map(str, expected_symbols)) - set(actual_symbols))
        extra_symbols = sorted(set(actual_symbols) - set(map(str, expected_symbols)))
        if missing_symbols or extra_symbols:
            failures.append(f"symbol coverage mismatch missing={missing_symbols} extra={extra_symbols}")
    if expected_splits:
        expected_slots = len(expected_symbols or actual_symbols) * int(expected_splits)
        if rows < expected_slots:
            failures.append(f"row_count too small actual_rows={rows} expected_min_rows={expected_slots}")
        if actual_key_count < expected_slots:
            failures.append(f"(symbol,split) coverage incomplete actual_slots={actual_key_count} expected_slots={expected_slots}")
        if len(actual_splits) < int(expected_splits):
            failures.append(f"split coverage incomplete actual_splits={len(actual_splits)} expected_splits={expected_splits}")
    lineage = _validate_stage25_lineage(path, source_path)
    if lineage["status"] != "PASS":
        failures.append(lineage["reason"])
    if failures:
        return _status("FAIL", path, cols, required, "; ".join(failures), row_count=rows)
    return _status("PASS", path, cols, required, "ok", row_count=rows)


def materialize_final_oos_predictions(
    *,
    run_id: str,
    profile: str,
    source_path: str | Path | None = None,
    out_path: str | Path = FINAL_OOS_PREDICTIONS,
    target_col: str = "target_15m_ret",
) -> dict[str, Any]:
    source = Path(source_path) if source_path else FINAL_WFA_BACKTEST
    out = Path(out_path)
    if not source.exists():
        raise RuntimeError(f"FINAL OOS FAIL: source artifact missing: {source}")
    df = pl.read_parquet(source)
    if "prediction" not in df.columns:
        raise RuntimeError(f"FINAL OOS FAIL: source={source} missing prediction column; available={df.columns}")
    if target_col not in df.columns:
        if "target_value" in df.columns:
            df = df.with_columns(pl.col("target_value").alias(target_col))
        else:
            raise RuntimeError(f"FINAL OOS FAIL: source={source} missing target column {target_col}; available={df.columns}")
    if "timestamp" not in df.columns:
        if "ts_event" not in df.columns:
            raise RuntimeError(f"FINAL OOS FAIL: source={source} missing timestamp/ts_event; available={df.columns}")
        df = df.with_columns(pl.col("ts_event").alias("timestamp"))
    if "split" not in df.columns:
        df = df.with_columns(
            pl.col("split_id").cast(pl.Utf8).alias("split") if "split_id" in df.columns else pl.lit("").alias("split")
        )
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit("").alias("symbol"))
    df = df.with_columns(
        pl.lit(str(run_id)).alias("run_id"),
        pl.lit(str(profile)).alias("profile"),
    )
    keep = [
        "run_id",
        "profile",
        "symbol",
        "split",
        "timestamp",
        "ts_event",
        "prediction_time",
        "execution_time",
        "prediction",
        "prediction_prob",
        target_col,
        "target_col",
        "target_value",
        "raw_signal",
        "position",
        "position_after",
        "position_before",
        "position_delta",
        "pnl",
        "gross_pnl",
        "net_pnl",
        "costs",
        "feature_set_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "modeling_mode",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.select([c for c in keep if c in df.columns]).write_parquet(out)
    actual_rows = int(pl.scan_parquet(out).select(pl.len().alias("rows")).collect()["rows"][0] or 0)
    write_lineage(
        out,
        run_id=run_id,
        profile=profile,
        source_stage="stage_24_final_wfa",
        source_artifact_path=source,
        selected_feature_count=_selected_feature_count(),
        actual_rows=actual_rows,
    )
    return validate_final_oos_predictions(out, target_col=target_col, source_path=source)


def _validate_stage25_lineage(path: Path, source_path: str | Path) -> dict[str, Any]:
    sidecar = lineage_path(path)
    if not sidecar.exists():
        return {"status": "FAIL", "reason": f"missing lineage sidecar: {sidecar}"}
    meta = read_json_or_sidecar(path)
    required = ["run_id", "profile", "created_at", "source_stage", "source_artifact_path", "source_artifact_checksum"]
    missing = [k for k in required if not meta.get(k)]
    if missing:
        return {"status": "FAIL", "reason": "missing lineage fields: " + ",".join(missing)}
    source = Path(source_path)
    if Path(str(meta.get("source_artifact_path"))).as_posix() != source.as_posix():
        return {"status": "FAIL", "reason": f"lineage source path mismatch recorded={meta.get('source_artifact_path')} expected={source}"}
    if not source.exists():
        return {"status": "FAIL", "reason": f"lineage source missing: {source}"}
    if str(meta.get("source_artifact_checksum")) != file_sha256(source):
        return {"status": "FAIL", "reason": "lineage source checksum mismatch"}
    return {"status": "PASS", "reason": "ok"}


def _selected_feature_count() -> int | str:
    manifest = Path("data/frozen_features/phase5_v1/manifest.json")
    if not manifest.exists():
        return ""
    try:
        import json

        return int(json.loads(manifest.read_text(encoding="utf-8")).get("selected_feature_count") or 0)
    except Exception:
        return ""


def _status(
    status: str,
    path: Path,
    available: list[str],
    required: list[str],
    reason: str,
    *,
    row_count: int | str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "artifact_path": str(path),
        "available_columns": ",".join(available),
        "required_columns": ",".join(required),
        "producing_stage": "Stage 25 FINAL OOS PREDICTIONS from Stage 24 FINAL WFA WITH FROZEN FEATURES",
        "regenerate_command": "no supported standalone final WFA regeneration command found; rerun the supported final pipeline when implemented",
        "reason": reason,
        "row_count": row_count,
    }
