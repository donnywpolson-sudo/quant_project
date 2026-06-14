from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.phase7_wfa.run_wfa as wfa
from scripts.phase7_wfa.run_wfa import PREDICTION_COLUMNS, run_wfa


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
  logistic_direction_v1:
    stage: phase_7a_linear_controls
    family: logistic_regression
    task: classification
    target: target_sign_with_deadzone
    enabled: true
    requires_optional_dependency: false
  logistic_fade_success_v1:
    stage: phase_7a_linear_controls
    family: logistic_regression
    task: classification
    target: target_fade_success_15m
    enabled: true
    requires_optional_dependency: false
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_feature_matrix(root: Path) -> Path:
    feature_cols = ["feature_train_only_marker", "feature_signal", "feature_fade_signal"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "feature_cols.json").write_text(json.dumps(feature_cols), encoding="utf-8")

    ts = pd.date_range("2024-01-01T00:00:00Z", periods=72, freq="h")
    train_marker = [10.0 if value < pd.Timestamp("2024-01-03T00:00:00Z") else 1000.0 for value in ts]
    signal = [float(idx % 3) for idx in range(len(ts))]
    fade_signal = [float(idx % 2) for idx in range(len(ts))]
    target_sign = [-1, 0, 1] * 24
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
            "target_entry_price": 100.25,
            "target_exit_price": 100.50,
            "minutes_until_session_close": 60.0,
            "target_ret_15m": [value * 0.001 for value in signal],
            "target_sign_with_deadzone": target_sign,
            "target_fade_success_15m": [idx % 2 == 0 for idx in range(len(ts))],
            "fade_long_success_15m": [idx % 2 == 0 for idx in range(len(ts))],
            "fade_short_success_15m": False,
            "feature_train_only_marker": train_marker,
            "feature_signal": signal,
            "feature_fade_signal": fade_signal,
        }
    ).to_parquet(path, index=False)
    return path


def _feature_root(tmp_path: Path) -> Path:
    return tmp_path / "data" / "feature_matrices" / "baseline"


def _write_split_plan(
    path: Path,
    *,
    split_group: str = "research",
    selection_allowed: bool | None = True,
) -> Path:
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
                        "fold_id": f"ES_{split_group}_0001",
                        "split_group": split_group,
                        "train_start": "2024-01-01T00:00:00+00:00",
                        "purged_train_end": "2024-01-02T23:00:00+00:00",
                        "test_start": "2024-01-03T00:00:00+00:00",
                        "test_end": "2024-01-03T23:00:00+00:00",
                        **(
                            {"selection_allowed": selection_allowed}
                            if selection_allowed is not None
                            else {}
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_wfa_writes_oos_predictions_and_manifest(tmp_path: Path) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(reports_root / "split_plan.json")
    _write_feature_matrix(input_root)

    manifest = run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    prediction_path = predictions_root / "baseline" / "oos_predictions.parquet"
    report_path = reports_root / "baseline_wfa_report.json"
    manifest_path = reports_root / "baseline_predictions_manifest.json"
    predictions = pd.read_parquet(prediction_path)

    assert manifest["failure_count"] == 0
    assert manifest["input_root"] == input_root.as_posix()
    assert manifest["output_root"] == predictions_root.as_posix()
    assert manifest["prediction_count"] == 72
    assert manifest["artifact_evidence_ready"] is True
    assert manifest["artifact_evidence_failures"] == []
    assert manifest["stale_output_path_exists"] is False
    assert set(PREDICTION_COLUMNS).issubset(predictions.columns)
    assert len(predictions) == 24 * 3
    assert predictions["timestamp"].min() >= pd.Timestamp("2024-01-03T00:00:00Z")
    assert predictions["timestamp"].max() <= pd.Timestamp("2024-01-03T23:00:00Z")
    assert not predictions.duplicated(
        subset=["market", "timestamp", "fold_id", "model_id", "target_name"]
    ).any()
    assert report_path.exists()
    assert manifest_path.exists()
    assert manifest["output_file_hashes"][prediction_path.as_posix()] != "MISSING"


def test_report_records_train_only_fit_window_and_feature_mean(tmp_path: Path) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(reports_root / "split_plan.json")
    _write_feature_matrix(input_root)

    run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    report = json.loads((reports_root / "baseline_wfa_report.json").read_text(encoding="utf-8"))
    first = report["diagnostics"][0]
    assert pd.Timestamp(first["fit_ts_max"]) < pd.Timestamp(first["score_ts_min"])
    assert first["train_feature_means_sample"]["feature_train_only_marker"] == 10.0
    classifier = next(item for item in report["diagnostics"] if item["model_id"] == "logistic_direction_v1")
    assert classifier["dummy_fallback_used"] is False
    assert classifier["y_train_unique"] == 3
    assert classifier["y_train_class_counts"] == {"-1": 16, "0": 16, "1": 16}
    assert classifier["probability_std_by_column"]["p_long"] > 0.0


def test_canonical_targets_are_consumed_without_alias_materialization(tmp_path: Path) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(reports_root / "split_plan.json")
    _write_feature_matrix(input_root)

    run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    predictions = pd.read_parquet(predictions_root / "baseline" / "oos_predictions.parquet")
    fade = predictions[predictions["target_name"] == "target_fade_success_15m"]
    assert not fade.empty
    assert set(fade["y_true"].unique()).issubset({0, 1})
    assert fade["p_fade_success"].notna().all()


@pytest.mark.parametrize("split_group", ["restricted", "forward", "final_holdout"])
def test_non_research_fold_is_not_fit_by_default(tmp_path: Path, split_group: str) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(
        reports_root / "split_plan.json",
        split_group=split_group,
        selection_allowed=False,
    )
    _write_feature_matrix(input_root)

    manifest = run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    assert manifest["failure_count"] > 0
    assert "no selectable research folds" in " ".join(manifest["failures"])
    assert manifest["skipped_fold_count"] == 1
    assert manifest["artifact_evidence_ready"] is False


def test_stale_split_plan_without_selection_allowed_fails(tmp_path: Path) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(reports_root / "split_plan.json", selection_allowed=None)
    _write_feature_matrix(input_root)

    manifest = run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    assert manifest["failure_count"] > 0
    assert "missing selection_allowed" in " ".join(manifest["failures"])


def test_existing_prediction_output_is_flagged_when_current_run_writes_none(
    tmp_path: Path,
) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(
        reports_root / "split_plan.json",
        split_group="restricted",
        selection_allowed=False,
    )
    _write_feature_matrix(input_root)
    stale_path = predictions_root / "baseline" / "oos_predictions.parquet"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "split_group": ["restricted"],
            "model_id": ["old_model"],
            "y_pred_raw": [0.25],
        }
    ).to_parquet(stale_path, index=False)
    stale_hash = wfa._file_sha256(stale_path)

    manifest = run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    assert wfa._file_sha256(stale_path) == stale_hash
    assert manifest["prediction_count"] == 0
    assert manifest["stale_output_path_exists"] is True
    assert manifest["stale_output_path"] == stale_path.as_posix()
    assert manifest["stale_output_file_hash"] == stale_hash
    assert manifest["stale_output_row_count"] == 1
    assert manifest["stale_output_split_groups"] == ["restricted"]
    assert manifest["output_file_hashes"][stale_path.as_posix()] == "NOT_WRITTEN"
    assert manifest["artifact_evidence_ready"] is False
    assert "stale prediction output exists" in " ".join(manifest["artifact_evidence_failures"])
    assert "stale prediction output exists from a previous run" in " ".join(
        manifest["failures"]
    )


@pytest.mark.parametrize("split_group", ["restricted", "forward", "final_holdout"])
def test_non_research_fold_marked_selectable_still_fails(tmp_path: Path, split_group: str) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(
        reports_root / "split_plan.json",
        split_group=split_group,
        selection_allowed=True,
    )
    _write_feature_matrix(input_root)

    manifest = run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    assert manifest["failure_count"] > 0
    failures = " ".join(manifest["failures"])
    assert "non-research split_group" in failures
    assert "no selectable research folds" in failures
    assert manifest["skipped_fold_count"] == 1


def test_convergence_warning_is_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class WarningEstimator:
        def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "WarningEstimator":
            import warnings

            warnings.warn("forced convergence", ConvergenceWarning)
            return self

    def forced_estimator(spec: object, y_train: pd.Series) -> tuple[WarningEstimator, str]:
        return WarningEstimator(), "logistic_regression"

    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(reports_root / "split_plan.json")
    _write_feature_matrix(input_root)
    monkeypatch.setattr(wfa, "_build_estimator", forced_estimator)

    manifest = run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    assert manifest["failure_count"] > 0
    assert "convergence warning" in " ".join(manifest["failures"])


def test_constant_classifier_probabilities_fail_without_dummy_fallback(tmp_path: Path) -> None:
    input_root = _feature_root(tmp_path)
    predictions_root = tmp_path / "data" / "predictions"
    reports_root = tmp_path / "reports" / "wfa"
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    split_plan = _write_split_plan(reports_root / "split_plan.json")
    path = _write_feature_matrix(input_root)
    frame = pd.read_parquet(path)
    frame["feature_signal"] = 1.0
    frame["feature_fade_signal"] = 1.0
    frame["feature_train_only_marker"] = 1.0
    frame.to_parquet(path, index=False)

    manifest = run_wfa(
        profile="fixture",
        matrix="baseline",
        run="baseline",
        input_root=input_root,
        split_plan=split_plan,
        predictions_root=predictions_root,
        reports_root=reports_root,
        models_config=models_config,
    )

    assert manifest["failure_count"] > 0
    assert "near-constant" in " ".join(manifest["failures"])
    report = json.loads((reports_root / "baseline_wfa_report.json").read_text(encoding="utf-8"))
    failed = [item for item in report["diagnostics"] if item["status"] == "FAIL"]
    assert failed
    assert failed[0]["dummy_fallback_used"] is False
    assert failed[0]["prediction_std"] <= wfa.CLASSIFIER_COLLAPSE_STD_EPS
