from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase7_wfa.run_wfa import run_wfa  # noqa: E402
from scripts.phase8_model_selection.audit_return_model_scale import (  # noqa: E402
    build_return_model_scale_audit,
    main,
)


def _write_models_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
policy:
  random_splits_allowed: false
  final_holdout_tuning_allowed: false
  hyperparameter_tuning_allowed_initially: false
models:
  ridge_return_v1:
    stage: phase_7a_linear_controls
    family: ridge_regression
    task: regression
    target: target_ret_15m
    enabled: true
    requires_optional_dependency: false
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_feature_matrix(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    feature_cols = ["feature_bad", "feature_good"]
    (root / "feature_cols.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=12, freq="h")
    feature_bad = [float(idx) for idx in range(8)] + [1.0e9, 8.0, 9.0, 10.0]
    target = [value * 0.001 for value in range(8)] + [0.001, 0.002, 0.003, 0.004]
    path = root / "ES" / "2024.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ts": ts,
            "market": "ES",
            "year": 2024,
            "session_id": "session",
            "session_segment_id": "segment",
            "causal_valid": True,
            "target_valid": True,
            "feature_input_valid": True,
            "training_row_valid": True,
            "close": 100.0,
            "target_entry_ts": ts + pd.Timedelta(minutes=1),
            "target_exit_ts": ts + pd.Timedelta(minutes=16),
            "target_entry_price": 100.0,
            "target_exit_price": 100.1,
            "minutes_until_session_close": 60.0,
            "target_ret_15m": target,
            "feature_bad": feature_bad,
            "feature_good": [float(idx % 2) for idx in range(12)],
        }
    ).to_parquet(path, index=False)
    return root


def _write_split_plan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "profile": "fixture",
                "markets": ["ES"],
                "years": [2024],
                "folds": [
                    {
                        "market": "ES",
                        "fold_id": "ES_research_0001",
                        "split_group": "research",
                        "train_start": "2024-01-01T00:00:00+00:00",
                        "purged_train_end": "2024-01-01T07:00:00+00:00",
                        "test_start": "2024-01-01T08:00:00+00:00",
                        "test_end": "2024-01-01T11:00:00+00:00",
                        "is_final_holdout": False,
                        "final_holdout": False,
                        "selection_allowed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    input_root = _write_feature_matrix(tmp_path / "data" / "feature_matrices" / "baseline")
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(reports_root / "split_plan.json")
    run_wfa(
        profile="fixture",
        matrix="baseline",
        run="fixture",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )
    return (
        predictions_root / "fixture" / "oos_predictions.parquet",
        reports_root,
        input_root,
        split_plan,
        models_config,
    )


def test_return_model_scale_audit_writes_outlier_and_contribution_reports(tmp_path: Path) -> None:
    predictions, reports_root, feature_root, split_plan, models_config = _write_fixture(tmp_path)
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"

    report = build_return_model_scale_audit(
        predictions_path=predictions,
        reports_root=reports_root,
        feature_root=feature_root,
        split_plan=split_plan,
        models_config=models_config,
        output_root=output_root,
        run="fixture",
        model_id="ridge_return_v1",
        target_name="target_ret_15m",
        abs_outlier_threshold=0.01,
        ratio_warn_threshold=100.0,
        max_outliers=10,
        top_contributions=5,
    )

    assert (output_root / "fixture_return_model_scale_summary.json").exists()
    assert (output_root / "fixture_return_model_scale_by_scope.csv").exists()
    assert (output_root / "fixture_return_model_outliers.csv").exists()
    assert (output_root / "fixture_return_model_wfa_reconciliation.csv").exists()
    assert (output_root / "fixture_return_model_feature_contributions.csv").exists()
    assert report["raw_calibrated_check"]["raw_calibrated_identical"]
    assert report["overall_scale"]["prediction_to_target_std_ratio"] > 100.0
    assert report["decision"] == "extreme_feature_value_drives_unbounded_phase7_ridge_prediction"

    outliers = pd.read_csv(output_root / "fixture_return_model_outliers.csv")
    assert outliers.iloc[0]["abs_prediction"] > 1000.0

    contributions = pd.read_csv(output_root / "fixture_return_model_feature_contributions.csv")
    assert contributions.iloc[0]["feature"] == "feature_bad"

    reconciliation = pd.read_csv(output_root / "fixture_return_model_wfa_reconciliation.csv")
    assert bool(reconciliation.iloc[0]["phase7_report_matches_saved_predictions"])


def test_return_model_scale_main_runs_cleanly(tmp_path: Path, monkeypatch) -> None:
    predictions, reports_root, feature_root, split_plan, models_config = _write_fixture(tmp_path)
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_return_model_scale",
            "--predictions",
            predictions.as_posix(),
            "--reports-root",
            reports_root.as_posix(),
            "--feature-root",
            feature_root.as_posix(),
            "--split-plan",
            split_plan.as_posix(),
            "--models-config",
            models_config.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--run",
            "fixture",
        ],
    )

    assert main() == 0
    assert (output_root / "fixture_return_model_scale_summary.json").exists()
