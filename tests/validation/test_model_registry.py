from __future__ import annotations

from pathlib import Path

import yaml

from scripts.phase4_features.build_baseline_features import validate_registry
from scripts.validation.model_registry import (
    REQUIRED_LINEAR_MODELS,
    REQUIRED_MODEL_FIELDS,
    REQUIRED_PREDICTION_COLUMNS,
    all_target_columns,
    load_yaml,
    resolve_purge_bars,
    validate_model_registry,
    validate_purge_policy,
)


ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = ROOT / "configs" / "models.yaml"
LAYOUT = ROOT / "project_layout.md"


def _config() -> dict:
    return load_yaml(MODELS_CONFIG)


def test_model_registry_validation() -> None:
    payload = yaml.safe_load(MODELS_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    config = _config()
    assert validate_model_registry(config) == []

    models = config["models"]
    assert REQUIRED_LINEAR_MODELS.issubset(models)

    for model_id, model in models.items():
        assert model_id
        for field in REQUIRED_MODEL_FIELDS:
            assert field in model
        assert isinstance(model["enabled"], bool)

    assert models["lightgbm_direction_v1"]["enabled"] is False
    assert models["xgboost_direction_v1"]["enabled"] is False
    assert models["lightgbm_direction_v1"]["requires_optional_dependency"] is True
    assert models["xgboost_direction_v1"]["requires_optional_dependency"] is True

    enabled_families = {model["family"] for model in models.values() if model["enabled"]}
    forbidden_tokens = ("neural", "transformer", "reinforcement")
    assert not any(token in family for family in enabled_families for token in forbidden_tokens)


def test_purge_auto_resolution() -> None:
    config = _config()
    purge = config["purge"]

    assert purge["entry_lag_bars"] == 1
    assert purge["target_horizon_bars"] == 15
    assert purge["purge_bars"] == "auto"
    assert resolve_purge_bars(purge) == 16
    assert purge["resolved_purge_bars"] == 16
    alpha = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    assert alpha["defaults"]["resolved_purge_bars"] == 16

    bad = dict(purge)
    bad["purge_bars"] = 15
    bad["resolved_purge_bars"] = 15
    errors = validate_purge_policy({"purge": bad})
    assert any("purge_bars must be auto" in error for error in errors)
    assert any("does not cover entry lag" in error for error in errors)


def test_target_group_exclusion() -> None:
    config = _config()
    groups = config["target_groups"]

    assert set(groups) == {
        "return_target",
        "direction_target",
        "fade_success_target",
        "trend_danger_target",
    }

    targets = all_target_columns(config)
    assert {
        "target_fade_success_15m",
        "target_trend_danger_30m",
    }.issubset(targets)

    forbidden_targets = set(config["feature_exclusion"]["forbidden_feature_targets"])
    assert set(groups["fade_success_target"]).issubset(forbidden_targets)
    assert set(groups["trend_danger_target"]).issubset(forbidden_targets)

    injected = validate_registry(["target_fade_success_15m", "target_trend_danger_30m"])
    assert any("forbidden columns" in failure for failure in injected)


def test_multi_model_prediction_schema() -> None:
    schema = _config()["prediction_schema"]
    columns = set(schema["required_columns"])

    assert REQUIRED_PREDICTION_COLUMNS.issubset(columns)
    assert {"model_id", "target_name"}.issubset(schema["primary_key"])
    assert schema["raw_and_calibrated_scores_separate"] is True
    assert schema["regression_probability_columns_nullable"] is True
    assert {"p_long", "p_short", "p_flat", "p_fade_success", "p_trend_danger"}.issubset(columns)


def test_calibration_train_only_discipline() -> None:
    calibration = _config()["calibration"]

    assert calibration["fit_scope"] == "train_fold_or_train_internal_only"
    assert calibration["test_fold_fit_allowed"] is False
    assert calibration["final_holdout_fit_allowed"] is False
    assert calibration["calibration_id_required"] is True
    assert calibration["no_calibration_marker"] == "no_calibration"
    assert calibration["preserve_raw_and_calibrated_scores"] is True


def test_model_selection_excludes_final_holdout() -> None:
    reports = _config()["model_selection_reports"]

    assert reports["final_holdout_excluded_from_selection"] is True
    assert "reports/model_selection/model_comparison.csv" in reports["artifacts"]
    assert "reports/model_selection/model_selection_report.json" in reports["artifacts"]
    assert "reports/model_selection/calibration_report.json" in reports["artifacts"]
    assert {"model_id", "model_family", "target_name", "model_config_hash"}.issubset(
        reports["group_by"]
    )


def test_frozen_model_immutability() -> None:
    frozen = _config()["frozen_set"]

    assert frozen["final_holdout_consumes_frozen_only"] is True
    assert frozen["final_holdout_model_selection_allowed"] is False
    assert frozen["final_holdout_threshold_tuning_allowed"] is False
    assert frozen["final_holdout_calibration_change_allowed"] is False
    assert frozen["final_holdout_feature_change_allowed"] is False
    assert frozen["config_hashes_required"] is True
    assert "data/frozen_models/phase5_v1/model_config.yaml" in frozen["model_artifacts"]
    assert "data/frozen_models/phase5_v1/calibration_config.yaml" in frozen["model_artifacts"]


def test_project_layout_downstream_ml_consistency() -> None:
    text = LAYOUT.read_text(encoding="utf-8")

    for phrase in (
        "Phase 7A linear controls",
        "Phase 7B HistGradientBoosting challengers",
        "Phase 7C optional LightGBM/XGBoost challengers",
        "Phase 8A calibration/model comparison",
        "frozen feature + model + calibration + policy set",
        "trend-danger / do-not-fade classifier",
    ):
        assert phrase in text
