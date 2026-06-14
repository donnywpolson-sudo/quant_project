from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import yaml

from scripts.validation.check_tier_2_coverage import (
    EXCLUDED,
    REQUIRED_COST_KEYS,
    TIER_2_UNIVERSE,
    build_report,
    load_yaml,
    resolve_profile_name,
)


ROOT = Path(__file__).resolve().parents[2]
FULL_UNIVERSE_PROFILE = "tier_3_research"
REMOVED_PROFILE_NAMES = (
    "tier_2_" + "liquid_recent",
    "tier_2_" + "liquid_long",
    "tier_3_" + "full_long",
    "tier_forward_2026",
    "target" + "_28_recent",
    "target" + "_28_long",
    "target" + "_28_forward_2026",
    "primary_" + "universe_recent",
    "primary_" + "universe_long",
    "primary_" + "universe_forward_2026",
)
STALE_REFERENCE_PATTERNS = (
    "tier_2_" + "liquid",
    "tier_3_" + "full",
    "target" + "_28",
    "primary_" + "universe",
)
REFERENCE_ROOTS = ("README.md", "build/project_layout.md", "configs", "tests", "scripts")


def _namespace(tmp_path: Path, *, config: Path, stage: str = "all") -> Namespace:
    return Namespace(
        profile=FULL_UNIVERSE_PROFILE,
        stage=stage,
        config=str(config),
        session_config=str(ROOT / "configs" / "market_sessions.yaml"),
        costs_config=str(ROOT / "configs" / "costs.yaml"),
        raw_root=str(tmp_path / "data" / "raw"),
        causal_root=str(tmp_path / "data" / "causally_gated_normalized"),
        labeled_root=str(tmp_path / "data" / "labeled"),
        feature_root=str(tmp_path / "data" / "feature_matrices"),
        canonical_feature_root=str(tmp_path / "data" / "feature_matrices" / "baseline"),
        wfa_reports_root=str(tmp_path / "reports" / "wfa"),
        report_out=str(tmp_path / "reports" / "validation" / "full_universe_coverage.json"),
    )


def _touch_complete_tree(tmp_path: Path, years: list[int]) -> None:
    for root_name in ("raw", "causally_gated_normalized", "labeled"):
        root = tmp_path / "data" / root_name
        for market in TIER_2_UNIVERSE:
            for year in years:
                path = root / market / f"{year}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder", encoding="utf-8")


def test_default_profile_exists_aliases_resolve_and_retired_profiles_absent() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    profiles = config["profiles"]
    aliases = config["aliases"]

    assert config["default_profile"] in profiles
    for alias, target in aliases.items():
        assert resolve_profile_name(alias, aliases) in profiles
        assert target in aliases or target in profiles

    for name in REMOVED_PROFILE_NAMES:
        assert name not in profiles
        assert name not in aliases


def test_tier_3_profile_matches_exact_universe_and_years() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    profiles = config["profiles"]
    aliases = config["aliases"]

    assert resolve_profile_name("tier_1", aliases) == "tier_1_research"
    assert resolve_profile_name("tier_2", aliases) == "tier_2_research"
    assert resolve_profile_name("tier_3", aliases) == "tier_3_research"
    assert resolve_profile_name("tier_2_long", aliases) == "tier_2_research"
    assert resolve_profile_name("tier_2_forward", aliases) == "tier_2_forward"

    markets = profiles[FULL_UNIVERSE_PROFILE]["markets"]
    assert markets == TIER_2_UNIVERSE
    assert len(markets) == 31
    assert len(set(markets)) == 31
    assert EXCLUDED == ["E7", "J7", "PA", "QI", "QO", "ZQ"]
    assert set(markets).isdisjoint(EXCLUDED)
    assert profiles[FULL_UNIVERSE_PROFILE]["years"] == list(range(2010, 2025))
    assert profiles["tier_3_holdout"]["years"] == [2025]
    assert profiles["tier_3_forward"]["years"] == [2026]


def test_inventory_and_test_only_profiles_are_blocked_from_research_use() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    profiles = config["profiles"]

    assert profiles["metadata_optional_test"]["forbid_research_use"] is True
    assert profiles["all_raw"]["discovery"] is True
    assert profiles["all_raw"]["forbid_research_use"] is True


def test_retired_profile_references_absent_from_primary_docs_config_tests_scripts() -> None:
    files: list[Path] = []
    for item in REFERENCE_ROOTS:
        path = ROOT / item
        if path.is_file():
            files.append(path)
        else:
            files.extend(p for p in path.rglob("*") if p.is_file())

    offenders: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in STALE_REFERENCE_PATTERNS:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{pattern}")

    assert offenders == []


def test_every_tier_2_market_has_family_session_cost_and_tick_coverage() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    sessions = load_yaml(ROOT / "configs" / "market_sessions.yaml")
    costs = load_yaml(ROOT / "configs" / "costs.yaml")

    families = config["profiles"][FULL_UNIVERSE_PROFILE]["market_families"]
    session_markets = sessions["markets"]
    templates = sessions["session_templates"]
    cost_markets = costs["markets"]

    for market in TIER_2_UNIVERSE:
        assert families.get(market)
        assert market in session_markets
        assert session_markets[market]["session_template"] in templates
        assert market in cost_markets
        for key in REQUIRED_COST_KEYS:
            assert key in cost_markets[market]
        assert cost_markets[market]["tick_size"] > 0
        assert cost_markets[market]["tick_value"] > 0
        assert cost_markets[market]["point_value"] > 0


def test_coverage_gate_passes_on_tmp_complete_tree(tmp_path: Path) -> None:
    config = ROOT / "configs" / "alpha_tiered.yaml"
    _touch_complete_tree(tmp_path, list(range(2010, 2025)))

    report = build_report(_namespace(tmp_path, config=config, stage="all"))

    assert report["status"] == "PASS"
    assert report["coverage_errors"] == []
    assert report["production_alpha_evidence_ready"] is True
    assert report["artifact_evidence_ready"] is True
    assert report["artifact_evidence_failures"] == []
    assert report["research_pipeline_ready"] is True
    assert report["live_trading_ready"] is False
    assert report["canonical_feature_root"] == (
        tmp_path / "data" / "feature_matrices" / "baseline"
    ).as_posix()
    assert report["non_canonical_feature_artifact_count"] == 0
    assert report["hard_gates"]["production_alpha_cost_gate"]["status"] == "PASS"
    artifact_gate = report["hard_gates"]["artifact_evidence_gate"]
    assert artifact_gate["status"] == "PASS"
    assert artifact_gate["non_canonical_feature_artifact_count"] == 0
    assert artifact_gate["invalid_prediction_manifest_count"] == 0
    live_gate = report["hard_gates"]["live_trading_readiness_gate"]
    assert live_gate["status"] == "FAIL"
    assert live_gate["contract_execution_mapping_ready"] is False
    assert live_gate["calendar_refresh_current"] is False
    assert live_gate["live_fill_model_available"] is False
    assert "contract_specific_execution_mapping_missing" in live_gate["blocking_reasons"]
    assert "current_exchange_calendar_refresh_missing" in live_gate["blocking_reasons"]
    assert "live_fill_or_slippage_model_missing" in live_gate["blocking_reasons"]


def test_coverage_gate_skips_product_unavailable_years(tmp_path: Path) -> None:
    config = ROOT / "configs" / "alpha_tiered.yaml"
    _touch_complete_tree(tmp_path, list(range(2010, 2025)))
    for market, years in {"RTY": range(2010, 2017), "SR3": range(2010, 2018)}.items():
        for year in years:
            (tmp_path / "data" / "raw" / market / f"{year}.parquet").unlink()

    report = build_report(_namespace(tmp_path, config=config, stage="raw"))

    assert report["status"] == "PASS"
    assert report["artifact_checks"]["raw"]["missing"] == []
    assert report["artifact_checks"]["raw"]["unavailable_by_market"] == {
        "RTY": list(range(2010, 2017)),
        "SR3": list(range(2010, 2018)),
    }


def test_non_canonical_feature_artifacts_are_reported_without_failing_research(
    tmp_path: Path,
) -> None:
    config = ROOT / "configs" / "alpha_tiered.yaml"
    _touch_complete_tree(tmp_path, list(range(2010, 2025)))
    feature_root = tmp_path / "data" / "feature_matrices"
    stale_path = feature_root / "ES" / "2024.parquet"
    canonical_path = feature_root / "baseline" / "ES" / "2024.parquet"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("old", encoding="utf-8")
    canonical_path.write_text("new", encoding="utf-8")

    report = build_report(_namespace(tmp_path, config=config, stage="all"))

    assert report["status"] == "PASS"
    assert report["research_pipeline_ready"] is True
    assert report["artifact_evidence_ready"] is False
    assert report["hard_gates"]["artifact_evidence_gate"]["status"] == "FAIL"
    assert report["non_canonical_feature_artifact_count"] == 1
    assert "non-canonical feature artifacts exist" in report["artifact_evidence_failures"][0]
    assert report["non_canonical_feature_artifacts"] == [
        {
            "artifact_path": stale_path.as_posix(),
            "canonical_path": canonical_path.as_posix(),
        }
    ]


def test_invalid_prediction_manifest_fails_artifact_evidence_only(tmp_path: Path) -> None:
    config = ROOT / "configs" / "alpha_tiered.yaml"
    _touch_complete_tree(tmp_path, list(range(2010, 2025)))
    reports_root = tmp_path / "reports" / "wfa"
    reports_root.mkdir(parents=True, exist_ok=True)
    manifest_path = reports_root / "baseline_predictions_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "failure_count": 1,
                "prediction_count": 0,
                "output_file_hashes": {
                    "data/predictions/baseline/oos_predictions.parquet": "NOT_WRITTEN"
                },
                "stale_output_path_exists": True,
                "artifact_evidence_ready": False,
            },
        ),
        encoding="utf-8",
    )

    report = build_report(_namespace(tmp_path, config=config, stage="all"))

    assert report["status"] == "PASS"
    assert report["research_pipeline_ready"] is True
    assert report["artifact_evidence_ready"] is False
    artifact_gate = report["hard_gates"]["artifact_evidence_gate"]
    assert artifact_gate["status"] == "FAIL"
    assert artifact_gate["invalid_prediction_manifest_count"] == 1
    assert manifest_path.as_posix() in report["artifact_evidence_failures"][0]
    prediction_warnings = report["artifact_warnings"]["prediction_manifests"]
    assert prediction_warnings["invalid_manifest_count"] == 1
    assert prediction_warnings["manifests"][0]["artifact_evidence_ready"] is False


def test_coverage_gate_fails_when_one_raw_file_is_missing(tmp_path: Path) -> None:
    config = ROOT / "configs" / "alpha_tiered.yaml"
    _touch_complete_tree(tmp_path, list(range(2010, 2025)))
    (tmp_path / "data" / "raw" / "ES" / "2010.parquet").unlink()

    report = build_report(_namespace(tmp_path, config=config, stage="raw"))

    assert report["status"] == "FAIL"
    assert "data/raw/ES/2010.parquet" in report["artifact_checks"]["raw"]["missing"][0]


def test_coverage_gate_fails_if_excluded_market_is_inserted(tmp_path: Path) -> None:
    payload = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    payload["profiles"][FULL_UNIVERSE_PROFILE]["markets"] = TIER_2_UNIVERSE + ["E7"]
    config = tmp_path / "alpha_tiered.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _touch_complete_tree(tmp_path, list(range(2010, 2025)))

    report = build_report(_namespace(tmp_path, config=config, stage="raw"))

    assert report["status"] == "FAIL"
    assert any("excluded markets present" in item for item in report["coverage_errors"])
