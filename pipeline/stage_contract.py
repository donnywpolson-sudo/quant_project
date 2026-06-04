from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.audit.pipeline_coverage import stage_catalog
from pipeline.validation.target_integrity import validate_target_integrity_root


@dataclass(frozen=True)
class StageContract:
    stage_index: int
    stage_name: str
    expected_input_paths: tuple[str, ...]
    expected_output_paths: tuple[str, ...]
    required_columns: tuple[str, ...] = ()
    produced_columns: tuple[str, ...] = ()
    validation_gate: str = ""
    resumable: bool = False
    wfa_ready: bool = False
    upstream_dependencies: tuple[int, ...] = ()

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return self.expected_input_paths

    @property
    def produced_outputs(self) -> tuple[str, ...]:
        return self.expected_output_paths


START_STAGE_TO_STAGE_INDEX = {
    "raw": 1,
    "validated": 4,
    "session_normalized": 6,
    "causally_gated_normalized": 8,
    "labeled": 10,
    "baseline_feature_matrix": 12,
    "expanded_feature_matrix": 20,
    "final_wfa": 24,
}

WFA_READY_START_STAGES = {"baseline_feature_matrix", "expanded_feature_matrix"}


def stage_contracts(target_col: str = "target_15m_ret") -> list[StageContract]:
    by_num = {s.number: s for s in stage_catalog()}
    return [
        StageContract(1, by_num[1].name, (), ("data/raw",), ("ts_event",), ("ts_event",), "research_data_preflight", True, False, ()),
        StageContract(2, by_num[2].name, ("data/raw",), ("data/raw/manifest.json",), (), (), "raw_manifest", True, False, (1,)),
        StageContract(3, by_num[3].name, ("data/raw",), ("data/validated",), ("ts_event", "open", "high", "low", "close", "volume"), ("ts_event", "open", "high", "low", "close", "volume"), "raw_data_validation", True, False, (1, 2)),
        StageContract(4, by_num[4].name, ("data/validated",), ("data/validated",), ("ts_event", "open", "high", "low", "close", "volume"), ("ts_event", "open", "high", "low", "close", "volume"), "checkpoint_gate_validated", True, False, (3,)),
        StageContract(5, by_num[5].name, ("data/validated",), ("data/session_normalized",), ("session_id", "session_date"), ("session_id", "session_date"), "session_normalization", True, False, (4,)),
        StageContract(6, by_num[6].name, ("data/session_normalized",), ("data/session_normalized",), ("session_id", "session_date"), ("session_id", "session_date"), "checkpoint_gate_session_normalized", True, False, (5,)),
        StageContract(7, by_num[7].name, ("data/session_normalized",), ("data/causally_gated_normalized",), ("prediction_time", "earliest_execution_time"), ("prediction_time", "earliest_execution_time"), "causal_gate", True, False, (6,)),
        StageContract(8, by_num[8].name, ("data/causally_gated_normalized",), ("data/causally_gated_normalized",), ("prediction_time", "earliest_execution_time"), ("prediction_time", "earliest_execution_time"), "checkpoint_gate_causally_gated", True, False, (7,)),
        StageContract(9, by_num[9].name, ("data/causally_gated_normalized",), ("data/labeled",), (target_col, "target_valid"), (target_col, "target_valid"), "target_integrity", True, False, (8,)),
        StageContract(10, by_num[10].name, ("data/labeled",), ("data/labeled",), (target_col, "target_valid"), (target_col, "target_valid"), "checkpoint_gate_labeled", True, False, (9,)),
        StageContract(11, by_num[11].name, ("data/labeled",), ("data/feature_matrices/baseline",), (target_col, "target_valid"), ("ts_event", target_col, "target_valid"), "baseline_feature_generation", True, False, (10,)),
        StageContract(12, by_num[12].name, ("data/feature_matrices/baseline",), ("data/feature_matrices/baseline",), ("ts_event", target_col, "target_valid"), ("ts_event", target_col, "target_valid"), "target_integrity", True, True, (11,)),
        StageContract(13, by_num[13].name, ("data/feature_matrices/baseline",), ("data/feature_matrices/baseline/column_registry.json",), (target_col,), (), "column_registry", True, False, (12,)),
        StageContract(14, by_num[14].name, ("data/feature_matrices/baseline",), ("reports/validation/wfa_contract_debug.csv",), ("ts_event", target_col, "target_valid"), (), "wfa_split_plan", False, False, (12, 13)),
        StageContract(15, by_num[15].name, ("data/feature_matrices/baseline",), ("output",), ("ts_event", target_col, "target_valid"), (), "wfa_train_test", False, False, (14,)),
        StageContract(16, by_num[16].name, ("output",), ("output/*/*/backtest_results.parquet",), ("prediction",), ("prediction",), "oos_predictions", False, False, (15,)),
        StageContract(17, by_num[17].name, ("output/*/*/backtest_results.parquet",), ("output/*/*/backtest_results.parquet",), (), (), "execution_cost_model", False, False, (16,)),
        StageContract(18, by_num[18].name, ("output/*/*/backtest_results.parquet",), ("reports/validation/experiment_comparison.csv",), (), (), "final_diagnostics", False, False, (17,)),
        StageContract(19, by_num[19].name, ("reports/validation/experiment_comparison.csv",), ("reports/acceptance",), (), (), "acceptance_gate", False, False, (18,)),
        StageContract(20, by_num[20].name, ("data/feature_matrices/baseline",), ("data/feature_matrices/expanded",), ("ts_event", target_col, "target_valid"), ("ts_event", target_col, "target_valid"), "feature_expansion", True, True, (12,)),
        StageContract(21, by_num[21].name, ("data/feature_matrices/expanded",), ("reports/validation/stage_21_feature_discovery_audit_report.json",), (target_col,), (), "feature_discovery", False, False, (20,)),
        StageContract(22, by_num[22].name, ("data/feature_matrices/expanded",), ("reports/validation/stage_22_train_only_selection_audit_report.json",), (target_col,), (), "train_only_feature_selection", False, False, (21,)),
        StageContract(23, by_num[23].name, ("reports/validation/stage_22_train_only_selection_audit_report.json",), ("data/frozen_features/phase5_v1/feature_cols.json", "data/frozen_features/phase5_v1/selected_features.csv", "data/frozen_features/phase5_v1/rejected_features.csv", "data/frozen_features/phase5_v1/manifest.json"), (), (), "frozen_feature_set", True, False, (22,)),
        StageContract(24, by_num[24].name, ("data/frozen_features/phase5_v1/feature_cols.json", "data/feature_matrices/expanded"), ("reports/validation/stage_24_final_wfa_backtest_results.parquet",), ("ts_event", target_col, "target_valid"), (), "final_wfa", False, False, (23,)),
        StageContract(25, by_num[25].name, ("reports/validation/stage_24_final_wfa_backtest_results.parquet",), ("reports/validation/stage_25_final_oos_predictions.parquet",), ("run_id", "profile", "symbol", "split", "timestamp", "prediction", target_col), ("prediction",), "final_oos_predictions", False, False, (24,)),
        StageContract(26, by_num[26].name, ("reports/validation/stage_25_final_oos_predictions.parquet",), ("reports/validation/stage_26_final_metrics_diagnostics_audit_report.json",), (), (), "final_metrics", False, False, (25,)),
        StageContract(27, by_num[27].name, ("reports/validation/stage_26_final_metrics_diagnostics_audit_report.json",), ("reports/validation/stage_27_strategy_acceptance_audit_report.json",), (), (), "strategy_acceptance_gate", False, False, (26,)),
    ]


def validate_stage_order_contract(target_col: str = "target_15m_ret") -> dict[str, Any]:
    contracts = stage_contracts(target_col)
    produced_at: dict[str, int] = {}
    for contract in contracts:
        for output in contract.produced_outputs:
            produced_at.setdefault(output, contract.stage_index)

    failures: list[str] = []
    for contract in contracts:
        for required in contract.required_inputs:
            producer = produced_at.get(required)
            if producer is not None and producer >= contract.stage_index:
                failures.append(
                    f"stage {contract.stage_index} consumes {required} before producer stage {producer}"
                )
    if failures:
        raise RuntimeError("STAGE CONTRACT FAIL: " + "; ".join(failures))
    return {"status": "PASS", "stages": len(contracts)}


def assert_wfa_ready_root(
    *,
    start_stage: str,
    root: str | Path,
    config: Any,
    checkpoint_mode: bool,
    fail_prefix: str = "WFA INPUT CONTRACT FAIL",
) -> None:
    if start_stage not in WFA_READY_START_STAGES:
        raise RuntimeError(
            f"{fail_prefix}: stage={start_stage} root={root} checkpoint_mode={checkpoint_mode} "
            "is not WFA-ready; WFA requires baseline_feature_matrix with target_15m_ret and target_valid. "
            "Regenerate: python run.py --from-stage causally_gated_normalized --data-root data\\causally_gated_normalized. "
            "Resume: python run.py --from-stage baseline_feature_matrix --data-root data\\feature_matrices\\baseline"
        )
    try:
        validate_target_integrity_root(root, config, fail_prefix=fail_prefix)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{fail_prefix}: stage={start_stage} producer_stage=BASELINE FEATURE MATRIX "
            f"root={root} checkpoint_mode={checkpoint_mode}; {exc}. "
            "Regenerate baseline matrix: python run.py --from-stage causally_gated_normalized "
            "--data-root data\\causally_gated_normalized. "
            "Resume after fix: python run.py --from-stage baseline_feature_matrix "
            "--data-root data\\feature_matrices\\baseline"
        ) from exc
