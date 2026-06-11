#!/usr/bin/env python3
"""Validate the downstream model registry config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_MODEL_FAMILIES = {
    "ridge_regression",
    "logistic_regression",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
}
BLOCKED_FAMILY_TOKENS = ("neural", "transformer", "reinforcement", "rl")
REQUIRED_MODEL_FIELDS = ("stage", "family", "task", "target", "enabled")
REQUIRED_LINEAR_MODELS = {
    "ridge_return_v1",
    "logistic_direction_v1",
    "logistic_fade_success_v1",
    "logistic_trend_danger_v1",
}
OPTIONAL_EXTERNAL_MODELS = {"lightgbm_direction_v1", "xgboost_direction_v1"}
REQUIRED_STAGES = {
    "phase_7a_linear_controls",
    "phase_7b_sklearn_nonlinear",
    "phase_7c_optional_boosted_trees",
}
REQUIRED_PREDICTION_COLUMNS = {
    "market",
    "year",
    "fold_id",
    "timestamp",
    "session_segment_id",
    "model_id",
    "model_family",
    "target_name",
    "prediction_type",
    "y_true",
    "y_pred_raw",
    "y_pred_calibrated",
    "p_long",
    "p_short",
    "p_flat",
    "p_fade_success",
    "p_trend_danger",
    "calibration_id",
    "model_config_hash",
    "feature_config_hash",
    "execution_open",
    "execution_close",
    "target_valid",
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def all_target_columns(config: dict[str, Any]) -> set[str]:
    groups = config.get("target_groups", {})
    if not isinstance(groups, dict):
        return set()
    targets: set[str] = set()
    for values in groups.values():
        if isinstance(values, list):
            targets.update(str(value) for value in values)
    return targets


def resolve_purge_bars(purge: dict[str, Any]) -> int:
    entry_lag = int(purge["entry_lag_bars"])
    target_horizon = int(purge["target_horizon_bars"])
    configured = purge.get("purge_bars")
    if configured == "auto":
        return entry_lag + target_horizon
    return int(configured)


def model_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_purge_policy(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    purge = config.get("purge", {})
    if not isinstance(purge, dict):
        return ["purge mapping missing"]

    try:
        entry_lag = int(purge["entry_lag_bars"])
        target_horizon = int(purge["target_horizon_bars"])
        resolved = resolve_purge_bars(purge)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"invalid purge policy: {exc}"]

    expected = entry_lag + target_horizon
    if purge.get("purge_bars") != "auto":
        errors.append("purge_bars must be auto for target-aligned WFA")
    if resolved != expected:
        errors.append(f"resolved purge must be {expected}, got {resolved}")
    if int(purge.get("resolved_purge_bars", -1)) != expected:
        errors.append(f"resolved_purge_bars must be {expected}")
    if resolved == target_horizon:
        errors.append("hardcoded target_horizon purge does not cover entry lag")
    return errors


def validate_model_registry(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    policy = config.get("policy", {})
    if not isinstance(policy, dict):
        errors.append("policy mapping missing")
        policy = {}
    for key in (
        "random_splits_allowed",
        "final_holdout_tuning_allowed",
        "neural_nets_allowed",
        "transformers_allowed",
        "reinforcement_learning_allowed",
        "hyperparameter_tuning_allowed_initially",
        "optional_external_model_dependencies_required",
    ):
        if policy.get(key) is not False:
            errors.append(f"policy {key} must be false")

    errors.extend(validate_purge_policy(config))

    stages = config.get("model_stages", {})
    if not isinstance(stages, dict):
        errors.append("model_stages mapping missing")
        stages = {}
    missing_stages = sorted(REQUIRED_STAGES - set(stages))
    if missing_stages:
        errors.append(f"missing stages: {missing_stages}")

    targets = all_target_columns(config)
    if not targets:
        errors.append("target_groups must define target columns")
    non_target_names = sorted(target for target in targets if not target.startswith("target_"))
    if non_target_names:
        errors.append(f"target groups contain non-target names: {non_target_names}")

    models = config.get("models", {})
    if not isinstance(models, dict):
        errors.append("models mapping missing")
        models = {}

    missing_linear = sorted(REQUIRED_LINEAR_MODELS - set(models))
    if missing_linear:
        errors.append(f"missing required linear controls: {missing_linear}")

    for model_id, model in models.items():
        if not str(model_id).strip():
            errors.append("model id must be non-empty")
        if not isinstance(model, dict):
            errors.append(f"model {model_id} must be a mapping")
            continue
        missing = [field for field in REQUIRED_MODEL_FIELDS if field not in model]
        if missing:
            errors.append(f"model {model_id} missing fields: {missing}")
        family = str(model.get("family", ""))
        if family not in ALLOWED_MODEL_FAMILIES:
            errors.append(f"model {model_id} has unapproved family: {family}")
        if any(token in family for token in BLOCKED_FAMILY_TOKENS):
            errors.append(f"model {model_id} has blocked family: {family}")
        if model.get("task") not in {"regression", "classification"}:
            errors.append(f"model {model_id} has invalid task: {model.get('task')}")
        if model.get("stage") not in stages:
            errors.append(f"model {model_id} references unknown stage: {model.get('stage')}")
        if model.get("target") not in targets:
            errors.append(f"model {model_id} target not in target_groups: {model.get('target')}")
        if not isinstance(model.get("enabled"), bool):
            errors.append(f"model {model_id} enabled must be boolean")

    for model_id in REQUIRED_LINEAR_MODELS:
        model = models.get(model_id, {})
        if isinstance(model, dict) and model.get("enabled") is not True:
            errors.append(f"required linear control disabled: {model_id}")

    for model_id in OPTIONAL_EXTERNAL_MODELS:
        model = models.get(model_id, {})
        if not isinstance(model, dict):
            errors.append(f"missing optional external model entry: {model_id}")
            continue
        if model.get("enabled") is not False:
            errors.append(f"optional external model must be disabled by default: {model_id}")
        if model.get("requires_optional_dependency") is not True:
            errors.append(f"optional external model must declare optional dependency: {model_id}")
        if model.get("cpu_first") is not True:
            errors.append(f"optional external model must be cpu_first: {model_id}")

    prediction_schema = config.get("prediction_schema", {})
    if not isinstance(prediction_schema, dict):
        errors.append("prediction_schema mapping missing")
        prediction_schema = {}
    columns = set(prediction_schema.get("required_columns", []))
    missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - columns)
    if missing_columns:
        errors.append(f"prediction_schema missing columns: {missing_columns}")
    primary_key = set(prediction_schema.get("primary_key", []))
    if not {"model_id", "target_name"}.issubset(primary_key):
        errors.append("prediction_schema primary_key must include model_id and target_name")
    if prediction_schema.get("raw_and_calibrated_scores_separate") is not True:
        errors.append("raw and calibrated predictions must be separate")
    if prediction_schema.get("regression_probability_columns_nullable") is not True:
        errors.append("regression probability columns must be nullable")

    calibration = config.get("calibration", {})
    if not isinstance(calibration, dict):
        errors.append("calibration mapping missing")
        calibration = {}
    if calibration.get("test_fold_fit_allowed") is not False:
        errors.append("calibration cannot fit on test fold")
    if calibration.get("final_holdout_fit_allowed") is not False:
        errors.append("calibration cannot fit on final holdout")
    if calibration.get("calibration_id_required") is not True:
        errors.append("calibration_id is required")
    if not calibration.get("no_calibration_marker"):
        errors.append("no_calibration marker is required")

    reports = config.get("model_selection_reports", {})
    if not isinstance(reports, dict):
        errors.append("model_selection_reports mapping missing")
        reports = {}
    if reports.get("final_holdout_excluded_from_selection") is not True:
        errors.append("model selection must exclude final holdout")

    frozen = config.get("frozen_set", {})
    if not isinstance(frozen, dict):
        errors.append("frozen_set mapping missing")
        frozen = {}
    for key in (
        "final_holdout_consumes_frozen_only",
        "config_hashes_required",
    ):
        if frozen.get(key) is not True:
            errors.append(f"frozen_set {key} must be true")
    for key in (
        "final_holdout_model_selection_allowed",
        "final_holdout_threshold_tuning_allowed",
        "final_holdout_calibration_change_allowed",
        "final_holdout_feature_change_allowed",
    ):
        if frozen.get(key) is not False:
            errors.append(f"frozen_set {key} must be false")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml")
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    errors = validate_model_registry(config)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(model_config_hash(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
