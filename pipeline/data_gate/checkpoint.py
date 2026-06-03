from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows
from pipeline.data_gate.manifest import build_data_manifest


SUPPORTED_START_STAGES = {
    "raw", "validated", "session_normalized", "causally_gated_normalized",
    "labeled", "baseline_feature_matrix", "expanded_feature_matrix",
}
REQUIRED_OHLCV = {"ts_event", "open", "high", "low", "close", "volume"}
REQUIRED_SESSION = {"session_id", "session_date", "market", "session_timezone", "session_calendar_accuracy"}
REQUIRED_CAUSAL = {"prediction_time", "earliest_execution_time", "non_model_metadata_columns"}


class CheckpointGateError(RuntimeError):
    pass


def _files(root: Path, symbols: list[str] | None, start_year: int | None, end_year: int | None) -> list[Path]:
    out = []
    for p in sorted(root.glob("*/*.parquet")):
        if symbols and p.parent.name not in symbols:
            continue
        try:
            year = int(p.stem)
        except ValueError:
            year = None
        if start_year and year and year < start_year:
            continue
        if end_year and year and year > end_year:
            continue
        out.append(p)
    return out


def _manifest_ok(root: Path, config: Any) -> tuple[bool, str]:
    if (root / "manifest.json").exists() and (root / "_manifest.csv").exists():
        return True, ""
    if list(root.glob("*/*.parquet")) and getattr(getattr(config, "data", object()), "allow_manifest_rebuild", False):
        build_data_manifest(root, stage=root.name)
        return True, "manifest rebuilt"
    return False, "missing manifest.json or _manifest.csv"


def _check_common(df: pl.DataFrame, stage: str) -> list[str]:
    failures = []
    missing = sorted(REQUIRED_OHLCV - set(df.columns))
    if missing:
        failures.append(f"missing required columns: {missing}")
    if "ts_event" in df.columns and df["ts_event"].n_unique() != df.height:
        failures.append("duplicate ts_event")
    if {"open", "high", "low", "close"}.issubset(df.columns):
        bad = df.filter((pl.col("high") < pl.col("low")) | (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low")) | (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))).height
        if bad:
            failures.append(f"invalid OHLC rows={bad}")
    if stage in {"session_normalized", "causally_gated_normalized", "labeled"}:
        missing_session = sorted(REQUIRED_SESSION - set(df.columns))
        if missing_session:
            failures.append(f"missing required session columns: {missing_session}")
        if "session_id" not in df.columns:
            failures.append("missing session_id; start from validated or run session normalization")
        elif df.filter(pl.col("session_id").is_null()).height:
            failures.append("null session_id")
    if stage in {"causally_gated_normalized", "labeled"}:
        missing_causal = sorted(REQUIRED_CAUSAL - set(df.columns))
        if missing_causal:
            failures.append(f"missing required causal columns: {missing_causal}")
        if "prediction_time" not in df.columns:
            failures.append("missing prediction_time; if this is session_normalized, start from session_normalized and run causal gating")
        exec_col = "earliest_execution_time" if "earliest_execution_time" in df.columns else ("execution_time" if "execution_time" in df.columns else None)
        if exec_col is None:
            failures.append("missing earliest_execution_time/execution_time")
        elif "prediction_time" in df.columns and df.filter(pl.col("prediction_time") > pl.col(exec_col)).height:
            failures.append("prediction_time > earliest execution time")
        forbidden = [c for c in df.columns if c.startswith(("target_", "label_", "future_")) and stage == "causally_gated_normalized"]
        if forbidden:
            failures.append(f"forbidden feature columns present: {forbidden}")
        for c in [c for c in df.columns if c.endswith("_available_at") and "prediction_time" in df.columns]:
            if df.filter(pl.col(c) > pl.col("prediction_time")).height:
                failures.append(f"availability timestamp after prediction_time: {c}")
    if stage == "labeled":
        target_cols = [c for c in df.columns if c.startswith("target_")]
        if not target_cols:
            failures.append("missing configured target column")
        elif df.filter(~pl.col(target_cols[0]).is_finite()).height == df.height:
            failures.append("no finite labels")
    return failures


def validate_checkpoint_stage(stage: str, root: str, config: Any, symbols: list[str] | None = None, start_year: int | None = None, end_year: int | None = None) -> dict:
    if stage not in SUPPORTED_START_STAGES - {"raw"}:
        raise ValueError(f"unsupported checkpoint stage={stage}")
    root_path = Path(root)
    rows = []
    failures = []
    manifest_ok, manifest_note = _manifest_ok(root_path, config)
    if not manifest_ok:
        failures.append(manifest_note)
    files = _files(root_path, symbols, start_year, end_year)
    if not files:
        failures.append(f"no parquet files under {root_path}/{{market}}/{{year}}.parquet")
    if stage == "baseline_feature_matrix" and not (root_path / "column_registry.json").exists():
        failures.append("missing column_registry.json")
    for p in files:
        try:
            df = pl.read_parquet(p)
            f = _check_common(df, stage)
            if stage == "baseline_feature_matrix":
                targets = [c for c in df.columns if c.startswith("target_")]
                features = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric() and not c.startswith(("target_", "label_", "future_"))]
                if not targets:
                    f.append("missing target column")
                if not features:
                    f.append("missing model feature columns")
            status = "FAIL" if f else "PASS"
            failures.extend([f"{p}: {x}" for x in f])
            rows.append({"path": str(p), "stage": stage, "status": status, "rows": df.height, "failures": "; ".join(f)})
        except Exception as exc:
            failures.append(f"{p}: {exc}")
            rows.append({"path": str(p), "stage": stage, "status": "FAIL", "rows": 0, "failures": str(exc)})
    remediation = _remediation(stage, failures)
    report = {
        "status": "FAIL" if failures else "PASS",
        "stage": stage,
        "root": str(root_path),
        "files": rows,
        "failures": failures,
        "manifest_note": manifest_note,
        "remediation": remediation,
    }
    atomic_write_json(f"reports/validation/checkpoint_gate_{stage}.json", report)
    write_csv_rows(f"reports/validation/checkpoint_gate_{stage}.csv", rows or [{"path": "", "stage": stage, "status": "FAIL", "failures": "; ".join(failures)}])
    return report


def _remediation(stage: str, failures: list[str] | None = None) -> str:
    text = "; ".join(failures or [])
    if stage == "causally_gated_normalized" and "missing session_id" in text:
        return (
            "This checkpoint is not causally gated normalized. It appears to be validated/raw-style OHLCV data.\n"
            "Recommended:\n"
            "  python -m pipeline.data.adopt_checkpoint --stage validated --source <root> --target data/validated --copy\n"
            "  python run.py --from-stage validated --data-root data/validated"
        )
    if stage == "causally_gated_normalized" and ("missing prediction_time" in text or "missing earliest_execution_time" in text):
        return (
            "This checkpoint has session-like data but missing causal timing columns.\n"
            "Recommended:\n"
            "  python -m pipeline.data.adopt_checkpoint --stage session_normalized --source <root> --target data/session_normalized --copy\n"
            "  python run.py --from-stage session_normalized --data-root data/session_normalized"
        )
    if stage == "causally_gated_normalized" and "future_" in text:
        return "Recommended: remove future/label/target columns or start from a pre-label stage and regenerate causal labels."
    if stage == "session_normalized":
        return "python -m pipeline.causal.gate --in-root data/session_normalized --out-root data/causally_gated_normalized"
    if stage == "causally_gated_normalized":
        return "If prediction_time is missing, start from session_normalized instead."
    return "Run the previous pipeline stage or rebuild the checkpoint manifest."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--symbols")
    p.add_argument("--start-year", type=int)
    p.add_argument("--end-year", type=int)
    args = p.parse_args()
    from pipeline.common.config import RootConfig
    symbols = args.symbols.split(",") if args.symbols else None
    report = validate_checkpoint_stage(args.stage, args.root, RootConfig(), symbols, args.start_year, args.end_year)
    print(f"checkpoint_gate={report['status']} stage={args.stage} root={args.root}")
    if report["status"] == "FAIL":
        raise SystemExit(f"CHECKPOINT GATE FAIL: stage={args.stage} root={args.root} reason={'; '.join(report['failures'][:3])}\n{report['remediation']}")


if __name__ == "__main__":
    main()
