from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows


ORDER = {
    "unknown": 0,
    "raw": 1,
    "validated_candidate": 2,
    "validated": 2,
    "session_normalized_candidate": 3,
    "session_normalized": 3,
    "causally_gated_normalized": 4,
    "labeled": 5,
    "baseline_feature_matrix": 6,
}
OHLCV = {"ts_event", "open", "high", "low", "close", "volume"}
SESSION = {"session_id", "session_date", "market", "session_timezone", "session_calendar_accuracy"}
CAUSAL = {"prediction_time", "earliest_execution_time", "non_model_metadata_columns"}
FORBIDDEN = ("future_",)


def classify_checkpoint(source: str | Path, target_col: str = "target_15m_ret") -> dict[str, Any]:
    source = Path(source)
    files = sorted(source.rglob("*.parquet"))
    rows: list[dict[str, Any]] = []
    for p in files:
        try:
            df = pl.read_parquet(p, n_rows=500)
            stage, reason = classify_frame(df, source, target_col)
            rows.append({"path": str(p), "inferred_stage": stage, "reason": reason, "columns": df.columns})
        except Exception as exc:
            rows.append({"path": str(p), "inferred_stage": "unknown", "reason": str(exc), "columns": []})

    if not rows:
        inferred = "unknown"
        status = "FAIL"
        reason = f"no parquet files found under {source}"
    else:
        min_order = min(ORDER.get(r["inferred_stage"], 0) for r in rows)
        candidates = [k for k, v in ORDER.items() if v == min_order]
        inferred = next((r["inferred_stage"] for r in rows if r["inferred_stage"] in candidates), "unknown")
        disagree = len({r["inferred_stage"] for r in rows}) > 1
        status = "WARN" if disagree else ("FAIL" if inferred == "unknown" else "PASS")
        reason = "multiple files disagree; using lowest common valid stage" if disagree else rows[0]["reason"]

    report = {"status": status, "source": str(source), "inferred_stage": inferred, "reason": reason, "files": rows}
    atomic_write_json("reports/validation/checkpoint_classification_report.json", report)
    write_csv_rows(
        "reports/validation/checkpoint_classification_summary.csv",
        [{"path": r["path"], "inferred_stage": r["inferred_stage"], "reason": r["reason"]} for r in rows]
        or [{"path": "", "inferred_stage": "unknown", "reason": reason}],
    )
    return report


def classify_frame(df: pl.DataFrame, root: Path | None = None, target_col: str = "target_15m_ret") -> tuple[str, str]:
    cols = set(df.columns)
    has_ohlcv = OHLCV.issubset(cols)
    has_target = target_col in cols or any(c.startswith(("target_", "label_")) for c in cols)
    numeric_features = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric() and not c.startswith(("target_", "label_", "future_"))]
    has_registry = bool(root and ((root / "column_registry.json").exists() or (root / "registry.json").exists()))
    if has_target and numeric_features and (has_registry or not has_ohlcv):
        return "baseline_feature_matrix", "target and model features found; registry exists or schema is sufficient"
    if not has_ohlcv:
        return "unknown", "required OHLCV columns missing"
    if "session_id" not in cols:
        return "validated_candidate", "OHLCV found, but session_id/prediction_time missing. Start from validated or run session normalization."
    missing_session = sorted(SESSION - cols)
    if missing_session:
        return "validated_candidate", f"session_id found, but required session-normalized columns missing: {missing_session}. Rerun session normalization."
    session_note = "session columns found."
    exec_col = "earliest_execution_time" if "earliest_execution_time" in cols else ("execution_time" if "execution_time" in cols else None)
    missing_causal = sorted(CAUSAL - cols)
    if missing_causal:
        return "session_normalized_candidate", f"{session_note} Session columns found, but causal columns missing: {missing_causal}. Start from session_normalized or run causal gating."
    if df.filter(pl.col("prediction_time") > pl.col(exec_col)).height:
        return "session_normalized_candidate", "prediction_time exceeds earliest execution time; rerun causal gating."
    forbidden = [c for c in df.columns if c.startswith(FORBIDDEN)]
    if forbidden:
        return "session_normalized_candidate", f"forbidden future columns present: {forbidden}; rerun from pre-label/pre-feature stage."
    for c in [c for c in df.columns if c.endswith("_available_at")]:
        if df.filter(pl.col(c) > pl.col("prediction_time")).height:
            return "session_normalized_candidate", f"availability timestamp after prediction_time: {c}; rerun causal gating."
    if has_target:
        target = target_col if target_col in cols else next(c for c in df.columns if c.startswith(("target_", "label_")))
        finite = df.filter(pl.col(target).is_finite()).height
        if finite:
            return "labeled", "causal timing columns and finite target found."
    return "causally_gated_normalized", "causal timing columns found and leakage checks passed."


def canonical_stage(stage: str) -> str:
    return {
        "validated_candidate": "validated",
        "session_normalized_candidate": "session_normalized",
    }.get(stage, stage)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target-col", default="target_15m_ret")
    args = p.parse_args()
    report = classify_checkpoint(args.source, args.target_col)
    print(f"inferred_stage={report['inferred_stage']} status={report['status']} reason={report['reason']}")
    if report["inferred_stage"] == "unknown":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
