from __future__ import annotations

from typing import Any

from pipeline.validation.experiment_comparison import print_threshold_outlier_summary, write_experiment_reports
from pipeline.validation.prediction_thresholds import print_threshold_diagnostic_summary, validate_current_run_diagnostics


def run_final_diagnostics(
    *,
    config: Any,
    run_id: str,
    profile: str,
    expected_rows: int,
    require_threshold_used: bool,
    verification_rows: list[dict],
    artifact_rows: list[dict],
) -> None:
    if getattr(config.pipeline, "modeling_mode", "minimal_compatible") != "full_research":
        return
    validate_current_run_diagnostics(
        expected_rows=expected_rows,
        require_threshold_used=require_threshold_used,
        expected_run_id=run_id,
        allow_env_fallback=False,
    )
    print_threshold_diagnostic_summary(
        expected_splits=expected_rows,
        expected_run_id=run_id,
        allow_env_fallback=False,
    )
    _, threshold_outliers = write_experiment_reports(
        run_id=run_id,
        profile=profile,
        expected_rows=expected_rows,
        verification_rows=verification_rows,
        artifact_rows=artifact_rows,
    )
    print_threshold_outlier_summary(threshold_outliers)
