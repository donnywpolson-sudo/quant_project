from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from pipeline.common.io_safe import atomic_write_json, write_csv_rows


REPORT_JSON = Path("reports/validation/pipeline_coverage_report.json")
REPORT_CSV = Path("reports/validation/pipeline_coverage_summary.csv")


@dataclass(frozen=True)
class StageDef:
    number: int
    name: str
    module_or_script: str
    callable_or_command: str
    input_paths: tuple[str, ...]
    output_paths: tuple[str, ...]
    manifest_key: str
    test_hint: str
    external_contract: bool = False
    notes: str = ""


def stage_catalog() -> list[StageDef]:
    return [
        StageDef(1, "RAW DATA", "external", "data vendor/manual load", tuple(), ("data/raw/{market}/{year}.parquet",), "raw_data", "test_stage_raw_manifest", True, "External ingest contract; required OHLCV columns."),
        StageDef(2, "RAW DATA MANIFEST", "scripts/build_data_manifests.py", "python scripts/build_data_manifests.py --stages raw", ("data/raw/{market}/{year}.parquet",), ("data/raw/manifest.json", "data/raw/_manifest.csv"), "raw_manifest", "test_stage_raw_manifest"),
        StageDef(3, "RAW DATA VALIDATION", "scripts/validate_databento_continuous.py", "python scripts/validate_databento_continuous.py --audit-only", ("data/raw/{market}/{year}.parquet",), ("reports/validation/raw_validation_report.json", "reports/validation/raw_validation_summary.csv"), "raw_validation", "test_stage_validation_to_validated"),
        StageDef(4, "VALIDATED DATA", "scripts/validate_databento_continuous.py", "python scripts/validate_databento_continuous.py --write-validated --clean-policy drop-invalid", ("data/raw/{market}/{year}.parquet",), ("data/validated/{market}/{year}.parquet", "data/validated/manifest.json", "data/validated/_manifest.csv"), "validated_data", "test_stage_validation_to_validated"),
        StageDef(5, "SESSION NORMALIZATION", "pipeline/session/normalize.py", "session_normalize_root", ("data/validated/{market}/{year}.parquet", "configs/raw_data_validation.yaml"), ("reports/session_normalization/session_normalization_report.json", "reports/session_normalization/session_normalization_summary.csv"), "session_normalization", "test_stage_session_normalization"),
        StageDef(6, "SESSION-NORMALIZED DATA", "scripts/build_data_manifests.py", "python scripts/build_data_manifests.py --stages session_normalized", ("data/session_normalized/{market}/{year}.parquet",), ("data/session_normalized/manifest.json", "data/session_normalized/_manifest.csv"), "session_normalized_data", "test_stage_session_normalization"),
        StageDef(7, "CAUSAL GATING", "pipeline/causal/gate.py", "causal_gate_root", ("data/session_normalized/{market}/{year}.parquet",), ("reports/causal_gating/causal_gating_report.json", "reports/causal_gating/causal_gating_summary.csv"), "causal_gating", "test_stage_causal_gating"),
        StageDef(8, "CAUSALLY GATED NORMALIZED DATA", "scripts/build_data_manifests.py", "python scripts/build_data_manifests.py --stages causally_gated_normalized", ("data/causally_gated_normalized/{market}/{year}.parquet",), ("data/causally_gated_normalized/manifest.json", "data/causally_gated_normalized/_manifest.csv"), "causally_gated_data", "test_stage_causal_gating"),
        StageDef(9, "TARGET / LABEL GENERATION", "pipeline/labels/generate.py", "add_labels", ("data/causally_gated_normalized/{market}/{year}.parquet",), ("reports/validation/label_generation_report.json",), "label_generation", "test_stage_labels"),
        StageDef(10, "LABELED DATA", "pipeline/labels/generate.py", "label_root", ("data/causally_gated_normalized/{market}/{year}.parquet",), ("data/labeled/{market}/{year}.parquet", "data/labeled/manifest.json", "data/labeled/_manifest.csv"), "labeled_data", "test_stage_labels"),
        StageDef(11, "BASELINE FEATURE GENERATION", "pipeline/features/baseline.py", "build_baseline_features", ("data/labeled/{market}/{year}.parquet",), ("reports/metrics/baseline_feature_matrix_report.json",), "baseline_feature_generation", "test_stage_baseline_features"),
        StageDef(12, "BASELINE FEATURE MATRIX", "pipeline/features/baseline.py", "baseline_feature_root", ("data/labeled/{market}/{year}.parquet",), ("data/feature_matrices/baseline/{market}/{year}.parquet",), "baseline_feature_matrix", "test_stage_baseline_features"),
        StageDef(13, "FEATURE / TARGET / METADATA COLUMN REGISTRY", "pipeline/features/registry.py", "write_column_registry", ("data/feature_matrices/baseline/{market}/{year}.parquet",), ("data/feature_matrices/baseline/column_registry.json",), "column_registry", "test_stage_column_registry"),
        StageDef(14, "WFA SPLIT PLAN", "run.py", "generate_walkforward_splits", ("data/validated/{market}/{year}.parquet",), ("artifacts/run_manifests/{run_id}.json",), "wfa_split_plan", "test_walkforward_contract"),
        StageDef(15, "BASELINE WFA TRAIN / TEST", "pipeline/modeling/full_research.py", "run_full_research_modeling", ("data/feature_matrices/baseline/{market}/{year}.parquet",), ("backtest_results.parquet",), "baseline_wfa", "test_full_research_integration"),
        StageDef(16, "OOS PREDICTIONS", "pipeline/cli.py", "_write_oos_predictions", ("backtest_results.parquet",), ("oos_predictions.parquet",), "baseline_oos_predictions", "test_oos_predictions"),
        StageDef(17, "EXECUTION + COST MODEL", "pipeline/audit/execution_trace.py", "validate_execution_trace", ("oos_predictions.parquet",), ("execution_trace_report.json",), "execution_cost_model", "test_oos_predictions"),
        StageDef(18, "METRICS + DIAGNOSTICS", "pipeline/analytics/aggregate.py", "compute_backtest_metrics", ("backtest_results.parquet",), ("reports/metrics/*_metrics_report.json",), "baseline_metrics", "test_cli_integration"),
        StageDef(19, "BASELINE ACCEPT / REJECT GATE", "pipeline/gates/acceptance.py", "run_acceptance_gate", ("reports/metrics/*_metrics_report.json",), ("reports/acceptance/*_acceptance_gate.json",), "baseline_acceptance_gate", "test_acceptance_gate"),
        StageDef(20, "FEATURE EXPANSION", "pipeline/features/expansion.py", "expand_features", ("data/feature_matrices/baseline/{market}/{year}.parquet",), ("data/feature_matrices/expanded/{market}/{year}.parquet", "data/feature_matrices/expanded/column_registry.json"), "feature_expansion", "test_stage_feature_expansion"),
        StageDef(21, "FEATURE DISCOVERY", "pipeline/cli.py", "cmd_discover", ("data/feature_matrices/expanded/{market}/{year}.parquet",), ("output/manifest_*",), "feature_discovery", "test_cli_integration"),
        StageDef(22, "TRAIN-ONLY FEATURE RANKING / SELECTION", "pipeline/features/discovery.py", "select_features_train_only", ("train_df",), ("artifacts/selectors/{run_id}/{symbol}_split_{split_id}_features.json",), "train_only_feature_selection", "test_train_only_selection"),
        StageDef(23, "FROZEN FEATURE SET", "pipeline/features/discovery.py", "select_features_train_only", ("train_df",), ("artifacts/selectors/{run_id}/{symbol}_split_{split_id}_features.json",), "frozen_feature_set", "test_train_only_selection"),
        StageDef(24, "FINAL WFA WITH FROZEN FEATURES", "pipeline/modeling/full_research.py", "run_full_research_modeling", ("artifacts/selectors/{run_id}/{symbol}_split_{split_id}_features.json",), ("backtest_results.parquet",), "final_wfa", "test_full_research_integration"),
        StageDef(25, "FINAL OOS PREDICTIONS", "pipeline/cli.py", "_write_oos_predictions", ("backtest_results.parquet",), ("oos_predictions.parquet",), "final_oos_predictions", "test_full_research_integration"),
        StageDef(26, "FINAL METRICS + DIAGNOSTICS", "pipeline/analytics/aggregate.py", "compute_backtest_metrics", ("backtest_results.parquet",), ("reports/metrics/*_metrics_report.json", "reports/stress/*_stress_report.json"), "final_metrics", "test_full_research_integration"),
        StageDef(27, "STRATEGY ACCEPT / REJECT GATE", "pipeline/gates/acceptance.py", "run_acceptance_gate", ("reports/metrics/*_metrics_report.json",), ("reports/acceptance/*_acceptance_gate.json",), "strategy_acceptance_gate", "test_oos_predictions"),
    ]


def _module_exists(path: str) -> bool:
    if path == "external":
        return True
    if path.endswith(".py"):
        return Path(path).exists()
    return importlib.util.find_spec(path.replace("/", ".").removesuffix(".py")) is not None


def _test_exists(hint: str) -> bool:
    return any(hint in str(p) or hint in p.read_text(encoding="utf-8", errors="ignore") for p in Path("tests").glob("test_*.py"))


def _represented_in_manifest(key: str) -> bool:
    for p in Path("artifacts/run_manifests").glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(s.get("manifest_key") == key or s.get("stage_key") == key for s in raw.get("stages", [])):
            return True
    return False


def _exists_any(patterns: tuple[str, ...]) -> bool:
    concrete = [p for p in patterns if "{" not in p and "*" not in p]
    globs = [p for p in patterns if "*" in p and "{" not in p]
    if any(Path(p).exists() for p in concrete):
        return True
    return any(list(Path().glob(g)) for g in globs)


def build_coverage_report(config_path: str | Path = "configs/alpha_tiered.yaml") -> dict[str, Any]:
    rows = []
    for s in stage_catalog():
        impl = _module_exists(s.module_or_script)
        test = _test_exists(s.test_hint)
        manifest = _represented_in_manifest(s.manifest_key)
        output_exists = _exists_any(s.output_paths)
        failures = []
        if not impl:
            failures.append("missing implementation")
        if not test:
            failures.append("missing test coverage")
        status = "FAIL" if failures else ("WARN" if s.external_contract else "PASS")
        notes = s.notes or ""
        if not output_exists:
            notes = (notes + " " if notes else "") + "Expected outputs not present in current workspace until stage command/run executes."
        if not manifest:
            notes = (notes + " " if notes else "") + "No current run manifest with this stage key; next run will include it."
        row = {
            **asdict(s),
            "status": status,
            "implementation_exists": impl,
            "output_path_exists": output_exists,
            "represented_in_run_manifest": manifest,
            "test_covers_stage": test,
            "notes_remediation": "; ".join(failures) if failures else notes,
        }
        rows.append(row)
    report = {
        "status": "FAIL" if any(r["status"] == "FAIL" for r in rows) else ("WARN" if any(r["status"] == "WARN" for r in rows) else "PASS"),
        "config": str(config_path),
        "stages": rows,
    }
    atomic_write_json(REPORT_JSON, report)
    write_csv_rows(REPORT_CSV, rows)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/alpha_tiered.yaml")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    report = build_coverage_report(args.config)
    print(f"pipeline_coverage={report['status']} stages={len(report['stages'])} report={REPORT_JSON}")
    if args.strict:
        bad = [r for r in report["stages"] if r["status"] == "FAIL" or (r["status"] == "WARN" and not r["external_contract"])]
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
