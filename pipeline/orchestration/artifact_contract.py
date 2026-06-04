from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from pipeline.validation.diagnostic_io import write_csv_json


ARTIFACT_KEYS = [
    "backtest_results",
    "oos_predictions",
    "leakage_report",
    "execution_trace_report",
    "metrics_report",
    "stress_report",
    "acceptance_report",
]


def safe_window_name(start: Any, end: Any) -> str:
    raw = "_".join(str(x or "none") for x in [start, end])
    return "".join(ch if ch.isalnum() else "-" for ch in raw)[:80]


def expected_artifact_paths(symbol: str, command: str, out_dir: Path, test_start, test_end, *, profile: str) -> dict:
    window = safe_window_name(
        test_start.isoformat() if hasattr(test_start, "isoformat") else test_start,
        test_end.isoformat() if hasattr(test_end, "isoformat") else test_end,
    )
    return {
        "backtest_results": str(out_dir / ("backtest_results_hmm.parquet" if command == "run-hmm" else "backtest_results.parquet")),
        "oos_predictions": str(out_dir / "oos_predictions.parquet"),
        "leakage_report": str(Path("reports") / "leakage" / f"{profile}_{symbol}_{command}_{window}.json"),
        "execution_trace_report": str(out_dir / "execution_trace_report.json"),
        "metrics_report": str(Path("reports") / "metrics" / f"{profile}_{symbol}_{command}_{window}_metrics_report.json"),
        "stress_report": str(Path("reports") / "stress" / f"{profile}_{symbol}_{command}_{window}_stress_report.json"),
        "acceptance_report": str(Path("reports") / "acceptance" / f"{profile}_{symbol}_{command}_{window}_acceptance_gate.json"),
    }


def prediction_pnl_missing_reason(result_row: dict, artifact_row: dict, col: str, checksum_col: str) -> str:
    value = result_row.get(checksum_col)
    if value not in ("missing", "all_nan", None, ""):
        return ""
    bt_path = Path(artifact_row.get("backtest_results") or result_row.get("path", ""))
    if not bt_path.exists():
        return f"{checksum_col} missing: backtest_results missing at {bt_path}"
    try:
        df = pl.read_parquet(bt_path)
    except Exception as exc:
        return f"{checksum_col} missing: cannot read backtest_results at {bt_path}: {exc}"
    if col not in df.columns:
        return f"{checksum_col} missing: column {col} absent in {bt_path}"
    return f"{checksum_col} missing: column {col} all non-finite/null in {bt_path}"


def artifact_failure_reasons(result_row: dict, artifact_row: dict) -> list[str]:
    reasons = []
    if result_row.get("error"):
        reasons.append(str(result_row["error"]))
    for key in ARTIFACT_KEYS:
        path = artifact_row.get(key)
        if path and not Path(path).exists():
            reasons.append(f"missing {key}: {path}")
    for col, checksum_col in [("prediction_prob", "pred_cs"), ("pnl", "pnl_cs")]:
        reason = prediction_pnl_missing_reason(result_row, artifact_row, col, checksum_col)
        if reason:
            reasons.append(reason)
    if artifact_row.get("error") and artifact_row.get("error") not in reasons:
        reasons.append(str(artifact_row["error"]))
    return reasons


def write_failure_reasons(rows: list[dict], out_dir: str | Path = "reports/validation") -> tuple[Path, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "failure_reasons.csv"
    json_path = out_path / "failure_reasons.json"
    fields = [
        "symbol",
        "split",
        "status",
        "rc",
        "command",
        "exception_type",
        "exception_message",
        "stdout_tail",
        "stderr_tail",
        "pred_cs",
        "pnl_cs",
        "failure_reason",
        "backtest_results",
        "oos_predictions",
        "metrics_report",
        "acceptance_report",
    ]
    return write_csv_json(rows, csv_path=csv_path, json_path=json_path, fields=fields)
