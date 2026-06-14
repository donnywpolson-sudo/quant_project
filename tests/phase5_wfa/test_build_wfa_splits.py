from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase5_wfa.build_wfa_splits import build_split_plan


def _write_profile_config(path: Path, *, profile: str = "research") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
defaults:
  final_holdout_years: [2025]
profile_defaults:
  tiny:
    train_days: 2
    test_days: 1
    step_days: 1
profiles:
  research:
    intent: test_research
    settings_profile: tiny
    markets: ["ES"]
    years: [2024]
  holdout:
    intent: test_final_holdout
    settings_profile: tiny
    markets: ["ES"]
    years: [2025]
    forbid_research_use: true
  restricted:
    intent: smoke_test
    settings_profile: tiny
    markets: ["ES"]
    years: [2024]
    forbid_research_use: true
  mixed:
    intent: test_mixed
    settings_profile: tiny
    markets: ["ES"]
    years: [2024, 2025]
aliases:
  selected: {profile}
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_models_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
policy:
  random_splits_allowed: false
  final_holdout_tuning_allowed: false
purge:
  entry_lag_bars: 1
  target_horizon_bars: 2
  purge_bars: auto
  resolved_purge_bars: 3
model_selection_reports:
  final_holdout_excluded_from_selection: true
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_matrix(root: Path, *, year: int, start: str) -> Path:
    path = root / "ES" / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 7 * 24 * 60
    ts = pd.date_range(start, periods=rows, freq="min", tz="UTC")
    pd.DataFrame(
        {
            "ts": ts,
            "market": "ES",
            "year": year,
            "training_row_valid": True,
            "target_valid": True,
            "feature_input_valid": True,
        }
    ).to_parquet(path, index=False)
    return path


def test_build_split_plan_enforces_purge_and_writes_manifest(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "feature_matrices"
    reports_root = tmp_path / "reports" / "wfa"
    profile_config = _write_profile_config(tmp_path / "configs" / "alpha_tiered.yaml")
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    _write_matrix(input_root, year=2024, start="2024-01-01T00:00:00Z")

    manifest = build_split_plan(
        profile="selected",
        input_root=input_root,
        reports_root=reports_root,
        profile_config=profile_config,
        models_config=models_config,
    )

    assert manifest["failure_count"] == 0
    assert manifest["fold_count"] > 0
    assert (reports_root / "split_plan.csv").exists()
    assert (reports_root / "split_plan.json").exists()

    first = manifest["folds"][0]
    assert first["market"] == "ES"
    assert first["split_group"] == "research"
    assert first["train_rows_before_purge"] > first["train_rows_after_purge"] > 0
    assert first["purged_train_rows"] == 3
    assert first["test_rows"] > 0
    assert first["resolved_purge_bars"] == 3
    assert first["embargo_bars"] == 3
    assert pd.Timestamp(first["purged_train_end"]) < pd.Timestamp(first["test_start"])
    assert first["selection_allowed"] is True

    saved = json.loads((reports_root / "split_plan.json").read_text(encoding="utf-8"))
    assert saved["fold_count_by_market"] == {"ES": manifest["fold_count"]}
    assert saved["final_holdout_policy"]["final_holdout_excluded_from_selection"] is True


def test_final_holdout_profile_is_tagged_and_excluded_from_selection(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "feature_matrices"
    reports_root = tmp_path / "reports" / "wfa"
    profile_config = _write_profile_config(
        tmp_path / "configs" / "alpha_tiered.yaml",
        profile="holdout",
    )
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    _write_matrix(input_root, year=2025, start="2025-01-01T00:00:00Z")

    manifest = build_split_plan(
        profile="selected",
        input_root=input_root,
        reports_root=reports_root,
        profile_config=profile_config,
        models_config=models_config,
    )

    assert manifest["failure_count"] == 0
    fold = manifest["folds"][0]
    assert fold["split_group"] == "final_holdout"
    assert fold["is_final_holdout"] is True
    assert fold["selection_allowed"] is False
    assert manifest["final_holdout_policy"]["final_holdout_tuning_allowed"] is False


def test_restricted_non_holdout_profile_is_not_tagged_as_forward(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "feature_matrices"
    reports_root = tmp_path / "reports" / "wfa"
    profile_config = _write_profile_config(
        tmp_path / "configs" / "alpha_tiered.yaml",
        profile="restricted",
    )
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    _write_matrix(input_root, year=2024, start="2024-01-01T00:00:00Z")

    manifest = build_split_plan(
        profile="selected",
        input_root=input_root,
        reports_root=reports_root,
        profile_config=profile_config,
        models_config=models_config,
    )

    assert manifest["failure_count"] == 0
    fold = manifest["folds"][0]
    assert fold["split_group"] == "restricted"
    assert fold["selection_allowed"] is False


def test_mixed_research_and_final_holdout_years_fail(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "feature_matrices"
    reports_root = tmp_path / "reports" / "wfa"
    profile_config = _write_profile_config(
        tmp_path / "configs" / "alpha_tiered.yaml",
        profile="mixed",
    )
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    _write_matrix(input_root, year=2024, start="2024-01-01T00:00:00Z")
    _write_matrix(input_root, year=2025, start="2025-01-01T00:00:00Z")

    manifest = build_split_plan(
        profile="selected",
        input_root=input_root,
        reports_root=reports_root,
        profile_config=profile_config,
        models_config=models_config,
    )

    assert manifest["failure_count"] == 1
    assert "mixes research and final-holdout years" in manifest["failures"][0]


def test_random_split_policy_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "feature_matrices"
    reports_root = tmp_path / "reports" / "wfa"
    profile_config = _write_profile_config(tmp_path / "configs" / "alpha_tiered.yaml")
    models_config = _write_models_config(tmp_path / "configs" / "models.yaml")
    text = models_config.read_text(encoding="utf-8")
    models_config.write_text(text.replace("random_splits_allowed: false", "random_splits_allowed: true"), encoding="utf-8")

    with pytest.raises(SystemExit, match="random splits"):
        build_split_plan(
            profile="selected",
            input_root=input_root,
            reports_root=reports_root,
            profile_config=profile_config,
            models_config=models_config,
        )
