from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.stage_contract import stage_contracts
from pipeline.validation.diagnostic_io import write_csv_json
from pipeline.features.frozen import validate_frozen_feature_set
from pipeline.validation.final_oos import FINAL_OOS_PREDICTIONS, FINAL_WFA_BACKTEST, validate_final_oos_predictions
from pipeline.validation.final_lineage import validate_lineage_freshness


PIPELINE_FLOW_AUDIT_CSV = Path("reports/validation/pipeline_flow_audit.csv")
PIPELINE_FLOW_AUDIT_JSON = Path("reports/validation/pipeline_flow_audit.json")

PIPELINE_FLOW_AUDIT_FIELDS = [
    "stage_index",
    "stage_name",
    "status",
    "expected_inputs_present",
    "expected_outputs_present",
    "required_columns_present",
    "required_columns_missing",
    "row_count",
    "min_ts",
    "max_ts",
    "upstream_stage_status",
    "reason",
    "expected_input",
    "expected_output",
    "validation_status",
    "freshness",
    "repair_command",
]


def build_pipeline_status(config: Any, *, data_root: str | None = None) -> list[dict[str, Any]]:
    target_col = str(getattr(getattr(config, "walkforward", object()), "walkforward_target", "target_15m_ret"))
    rows: list[dict[str, Any]] = []
    status_by_stage: dict[int, str] = {}

    for contract in stage_contracts(target_col):
        input_paths = [_resolve_path(p, config, data_root) for p in contract.expected_input_paths]
        output_paths = [_resolve_path(p, config, data_root) for p in contract.expected_output_paths]
        inputs_present = _all_paths_present(input_paths)
        outputs_present = _all_paths_present(output_paths)

        data_paths = _parquet_files(output_paths, config) or _parquet_files(input_paths, config)
        stats = _dataset_stats(data_paths, contract.required_columns)
        missing_cols = stats["missing_columns"]
        blocking_missing = [c for c in missing_cols if c != "target_valid"]
        display_missing = [f"{c}(optional)" if c == "target_valid" else c for c in missing_cols]
        required_present = [c for c in contract.required_columns if c not in missing_cols]
        freshness = _freshness(input_paths, output_paths, blocking_missing)

        upstream = _upstream_status(contract.upstream_dependencies, status_by_stage)
        if data_root and contract.stage_index in {12, 13}:
            upstream = "PASS"
        status, reason = _stage_status(
            inputs_present=inputs_present,
            outputs_present=outputs_present,
            blocking_missing=blocking_missing,
            required_columns=contract.required_columns,
            data_paths=data_paths,
            upstream=upstream,
            stats=stats,
        )
        if contract.stage_index == 23 and upstream == "PASS":
            frozen = validate_frozen_feature_set(config=config)
            status = str(frozen.get("status"))
            reason = str(frozen.get("reason"))
            outputs_present = status != "MISSING"
            freshness = "valid" if status == "PASS" else ("missing" if status == "MISSING" else "invalid")
        if contract.stage_index == 25 and upstream == "PASS":
            final_oos = validate_final_oos_predictions(
                target_col=target_col,
                expected_symbols=[str(s) for s in getattr(config, "symbols", []) or []],
                expected_splits=_expected_split_count(),
                source_path=FINAL_WFA_BACKTEST,
            )
            status = str(final_oos.get("status"))
            reason = _final_oos_reason(final_oos)
            outputs_present = status != "MISSING"
            freshness = "valid" if status == "PASS" else ("missing" if status == "MISSING" else "invalid")
            if final_oos.get("row_count") not in ("", None):
                stats["row_count"] = str(final_oos.get("row_count"))
        if contract.stage_index == 26 and upstream == "PASS":
            lineage = validate_lineage_freshness(
                artifact_path="reports/validation/stage_26_final_metrics_diagnostics_audit_report.json",
                source_artifact_path=FINAL_OOS_PREDICTIONS,
                stage_name="Stage 26 FINAL METRICS + DIAGNOSTICS",
            )
            if lineage["status"] != "PASS":
                status = str(lineage["status"])
                reason = f"{lineage['reason']}; repair_command={_repair_command(contract.stage_index, target_col)}"
                freshness = "stale" if status == "STALE" else freshness
        if contract.stage_index == 27 and upstream == "PASS":
            lineage = validate_lineage_freshness(
                artifact_path="reports/validation/stage_27_strategy_acceptance_audit_report.json",
                source_artifact_path="reports/validation/stage_26_final_metrics_diagnostics_audit_report.json",
                stage_name="Stage 27 STRATEGY ACCEPT / REJECT GATE",
            )
            if lineage["status"] != "PASS":
                status = str(lineage["status"])
                reason = f"{lineage['reason']}; repair_command={_repair_command(contract.stage_index, target_col)}"
                freshness = "stale" if status == "STALE" else freshness
        status_by_stage[contract.stage_index] = status

        row = {
            "stage": str(contract.stage_index),
            "stage_index": str(contract.stage_index),
            "stage_name": contract.stage_name,
            "expected_input": "|".join(str(p) for p in input_paths) or "-",
            "expected_output": "|".join(str(p) for p in output_paths) or "-",
            "present": "present" if outputs_present else "missing",
            "status": status,
            "validation_status": status,
            "expected_inputs_present": str(bool(inputs_present)),
            "expected_outputs_present": str(bool(outputs_present)),
            "row_count": stats["row_count"],
            "min_ts": stats["min_ts"],
            "max_ts": stats["max_ts"],
            "required_columns_present": ",".join(required_present) if required_present else "-",
            "required_columns_missing": ",".join(display_missing) if display_missing else "-",
            "upstream_stage_status": upstream,
            "reason": reason,
            "freshness": freshness,
            "repair_command": _repair_command(contract.stage_index, target_col),
        }
        rows.append(row)
    return rows


def write_pipeline_flow_audit(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    return write_csv_json(
        rows,
        csv_path=PIPELINE_FLOW_AUDIT_CSV,
        json_path=PIPELINE_FLOW_AUDIT_JSON,
        fields=PIPELINE_FLOW_AUDIT_FIELDS,
        key_fields={"stage_index"},
    )


def print_pipeline_status(rows: list[dict[str, Any]]) -> None:
    columns = [
        "stage",
        "stage_name",
        "expected_input",
        "expected_output",
        "present",
        "validation_status",
        "row_count",
        "min_ts",
        "max_ts",
        "required_columns_present",
        "required_columns_missing",
        "freshness",
        "reason",
    ]
    print(" | ".join(columns), flush=True)
    print(" | ".join("-" * len(c) for c in columns), flush=True)
    for row in rows:
        print(" | ".join(str(row.get(c, "")) for c in columns), flush=True)

    stage12 = _status_for(rows, 12)
    stage13 = _status_for(rows, 13)
    final_ready = all(_status_for(rows, i) == "PASS" for i in range(20, 28))
    if stage12 == "PASS" and stage13 == "PASS":
        print("Pipeline is baseline-WFA-ready", flush=True)
    else:
        print("Pipeline is NOT baseline-WFA-ready", flush=True)
    if final_ready:
        print("Pipeline is final-strategy-ready", flush=True)
    else:
        print(f"Pipeline is NOT final-strategy-ready: {_final_not_ready_reason(rows)}", flush=True)


def _status_for(rows: list[dict[str, Any]], stage_index: int) -> str:
    for row in rows:
        if str(row.get("stage_index")) == str(stage_index):
            return str(row.get("status"))
    return "MISSING"


def _row_for(rows: list[dict[str, Any]], stage_index: int) -> dict[str, Any]:
    for row in rows:
        if str(row.get("stage_index")) == str(stage_index):
            return row
    return {}


def _final_not_ready_reason(rows: list[dict[str, Any]]) -> str:
    if any(_status_for(rows, i) == "STALE" for i in (26, 27)):
        return "final metrics/gate stale relative to Stage 25"
    for i in range(24, 28):
        status = _status_for(rows, i)
        if status != "PASS":
            row = _row_for(rows, i)
            return f"stage {i} {status}: {row.get('reason', '')}"
    return "unknown"


def _resolve_path(raw: str, config: Any, data_root: str | None) -> Path:
    norm = raw.replace("\\", "/")
    data = getattr(config, "data", object())
    if norm == "data/raw":
        return Path(getattr(data, "raw_root", norm))
    if norm == "data/validated":
        return Path(getattr(data, "validated_root", norm))
    if norm == "data/session_normalized":
        return Path(getattr(data, "session_normalized_root", norm))
    if norm == "data/causally_gated_normalized":
        return Path(getattr(data, "causally_gated_root", norm))
    if norm == "data/feature_matrices/baseline" and data_root:
        return Path(data_root)
    if norm.startswith("data/feature_matrices/baseline/") and data_root:
        return Path(data_root) / norm.split("data/feature_matrices/baseline/", 1)[1]
    return Path(raw)


def _expand(path: Path) -> list[Path]:
    text = str(path)
    if any(ch in text for ch in "*?[]"):
        return sorted(Path().glob(text))
    return [path]


def _all_paths_present(paths: list[Path]) -> bool:
    if not paths:
        return True
    return all(any(p.exists() for p in _expand(path)) for path in paths)


def _any_path_present(paths: list[Path]) -> bool:
    return any(any(p.exists() for p in _expand(path)) for path in paths)


def _parquet_files(paths: list[Path], config: Any) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        for expanded in _expand(path):
            if expanded.is_file() and expanded.suffix == ".parquet":
                if _in_config_scope(expanded, config):
                    out.append(expanded)
            elif expanded.is_dir():
                out.extend(p for p in sorted(expanded.glob("*.parquet")) if _in_config_scope(p, config))
                out.extend(p for p in sorted(expanded.glob("*/*.parquet")) if _in_config_scope(p, config))
    return sorted(set(out))


def _in_config_scope(path: Path, config: Any) -> bool:
    symbols = set(str(s) for s in getattr(config, "symbols", []) or [])
    parts = set(path.parts)
    if symbols and ("data" in parts or "output" in parts) and not (symbols & parts):
        return False
    try:
        year = int(path.stem)
    except ValueError:
        return True
    start_year = getattr(config, "start_year", None)
    end_year = getattr(config, "end_year", None)
    if start_year is not None and year < int(start_year):
        return False
    if end_year is not None and year > int(end_year):
        return False
    return True


def _dataset_stats(paths: list[Path], required_columns: tuple[str, ...]) -> dict[str, Any]:
    if not paths:
        return {"row_count": "", "min_ts": "", "max_ts": "", "missing_columns": list(required_columns)}
    try:
        all_cols: set[str] = set()
        row_count = 0
        min_ts = ""
        max_ts = ""
        for path in paths:
            lf = pl.scan_parquet(path)
            cols = set(lf.collect_schema().names())
            all_cols |= cols
            exprs = [pl.len().alias("rows")]
            if "ts_event" in cols:
                exprs.extend([pl.col("ts_event").min().alias("min_ts"), pl.col("ts_event").max().alias("max_ts")])
            stats = lf.select(exprs).collect()
            row_count += int(stats["rows"][0])
            if "min_ts" in stats.columns and stats["min_ts"][0] is not None:
                cur_min = str(stats["min_ts"][0])
                cur_max = str(stats["max_ts"][0])
                min_ts = cur_min if not min_ts or cur_min < min_ts else min_ts
                max_ts = cur_max if not max_ts or cur_max > max_ts else max_ts
        missing = [c for c in required_columns if c not in all_cols]
        return {"row_count": str(row_count), "min_ts": min_ts, "max_ts": max_ts, "missing_columns": missing}
    except Exception as exc:
        return {"row_count": "", "min_ts": "", "max_ts": "", "missing_columns": list(required_columns), "error": str(exc)}


def _freshness(input_paths: list[Path], output_paths: list[Path], missing_cols: list[str]) -> str:
    if missing_cols:
        return "invalid"
    if not _all_paths_present(output_paths):
        return "missing"
    if {str(p) for p in input_paths} & {str(p) for p in output_paths}:
        return "valid"
    input_mtimes = _mtimes(input_paths)
    output_mtimes = _mtimes(output_paths)
    if input_mtimes and output_mtimes and max(input_mtimes) > min(output_mtimes):
        return "stale"
    return "valid"


def _mtimes(paths: list[Path]) -> list[float]:
    mtimes = []
    for path in paths:
        for expanded in _expand(path):
            if expanded.is_file():
                mtimes.append(expanded.stat().st_mtime)
            elif expanded.is_dir():
                mtimes.extend(p.stat().st_mtime for p in expanded.rglob("*") if p.is_file())
    return mtimes


def _upstream_status(deps: tuple[int, ...], status_by_stage: dict[int, str]) -> str:
    if not deps:
        return "none"
    statuses = [status_by_stage.get(dep, "MISSING") for dep in deps]
    if all(s == "PASS" for s in statuses):
        return "PASS"
    if any(s in {"FAIL", "STALE"} for s in statuses):
        return "FAIL"
    if any(s in {"MISSING", "SKIPPED"} for s in statuses):
        return "MISSING"
    return "WARN"


def _stage_status(
    *,
    inputs_present: bool,
    outputs_present: bool,
    blocking_missing: list[str],
    required_columns: tuple[str, ...],
    data_paths: list[Path],
    upstream: str,
    stats: dict[str, Any],
) -> tuple[str, str]:
    if upstream == "FAIL":
        return "SKIPPED", "upstream stage failed"
    if upstream == "MISSING":
        return "SKIPPED", "upstream stage missing"
    if not inputs_present:
        return "MISSING", "expected input missing"
    if not outputs_present:
        return "MISSING", "expected output missing"
    if required_columns and not data_paths:
        return "MISSING", "no tabular artifact found for required column validation"
    if blocking_missing:
        return "FAIL", "missing required columns: " + ",".join(blocking_missing)
    if stats.get("error"):
        return "FAIL", str(stats.get("error"))
    return "PASS", "ok"


def _repair_command(stage_index: int, target_col: str) -> str:
    if stage_index in {12, 13, 14, 15}:
        return "python run.py --from-stage baseline_feature_matrix --data-root data\\feature_matrices\\baseline"
    if stage_index in {9, 10, 11}:
        return "python run.py --from-stage causally_gated_normalized --data-root data\\causally_gated_normalized"
    if stage_index >= 20:
        return "no supported standalone final-stage regeneration command found; rerun the supported final pipeline when implemented"
    if target_col:
        return "regenerate upstream artifact for this stage"
    return ""


def _final_oos_reason(report: dict[str, Any]) -> str:
    if report.get("status") == "PASS":
        return "ok"
    return (
        f"{report.get('reason')}; artifact_path={report.get('artifact_path')}; "
        f"available_columns={report.get('available_columns')}; "
        f"required_columns={report.get('required_columns')}; "
        f"producing_stage={report.get('producing_stage')}; "
        f"regenerate_command={report.get('regenerate_command')}"
    )


def _expected_split_count() -> int | None:
    path = Path("reports/validation/wfa_contract_debug.csv")
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        splits = {str(r.get("split")) for r in rows if str(r.get("split") or "").isdigit()}
        return len(splits) or None
    except Exception:
        return None
