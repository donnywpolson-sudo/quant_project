from __future__ import annotations

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
TIER_2_PROFILES = (
    "tier_2_universe_recent",
    "tier_2_universe_long",
    "tier_2_forward_2026",
)
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
        profile="tier_2_universe_recent",
        stage=stage,
        config=str(config),
        session_config=str(ROOT / "configs" / "market_sessions.yaml"),
        costs_config=str(ROOT / "configs" / "costs.yaml"),
        raw_root=str(tmp_path / "data" / "raw"),
        causal_root=str(tmp_path / "data" / "causally_gated_normalized"),
        labeled_root=str(tmp_path / "data" / "labeled"),
        report_out=str(tmp_path / "reports" / "validation" / "tier_2_coverage.json"),
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


def test_tier_2_profiles_match_exact_universe_and_years() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    profiles = config["profiles"]
    aliases = config["aliases"]

    assert resolve_profile_name("tier_2", aliases) == "tier_2_universe_recent"
    assert resolve_profile_name("tier_2_recent", aliases) == "tier_2_universe_recent"
    assert resolve_profile_name("tier_2_long", aliases) == "tier_2_universe_long"
    assert resolve_profile_name("tier_2_forward", aliases) == "tier_2_forward_2026"

    for profile_name in TIER_2_PROFILES:
        markets = profiles[profile_name]["markets"]
        assert markets == TIER_2_UNIVERSE
        assert len(markets) == 28
        assert len(set(markets)) == 28
        assert set(markets).isdisjoint(EXCLUDED)

    assert profiles["tier_2_universe_recent"]["years"] == [2023, 2024, 2025]
    assert profiles["tier_2_universe_long"]["years"] == list(range(2010, 2026))
    assert profiles["tier_2_forward_2026"]["years"] == [2026]
    assert profiles["tier_2_forward_2026"]["forbid_feature_selection"] is True
    assert profiles["tier_2_forward_2026"]["forbid_policy_selection"] is True


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

    families = config["profiles"]["tier_2_universe_recent"]["market_families"]
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
    _touch_complete_tree(tmp_path, [2023, 2024, 2025])

    report = build_report(_namespace(tmp_path, config=config, stage="all"))

    assert report["status"] == "PASS"
    assert report["coverage_errors"] == []
    assert report["production_alpha_evidence_ready"] is False
    assert report["hard_gates"]["production_alpha_cost_gate"]["status"] == "FAIL"


def test_coverage_gate_fails_when_one_raw_file_is_missing(tmp_path: Path) -> None:
    config = ROOT / "configs" / "alpha_tiered.yaml"
    _touch_complete_tree(tmp_path, [2023, 2024, 2025])
    (tmp_path / "data" / "raw" / "ES" / "2023.parquet").unlink()

    report = build_report(_namespace(tmp_path, config=config, stage="raw"))

    assert report["status"] == "FAIL"
    assert "data/raw/ES/2023.parquet" in report["artifact_checks"]["raw"]["missing"][0]


def test_coverage_gate_fails_if_excluded_market_is_inserted(tmp_path: Path) -> None:
    payload = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    payload["profiles"]["tier_2_universe_recent"]["markets"] = TIER_2_UNIVERSE + ["MES"]
    config = tmp_path / "alpha_tiered.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _touch_complete_tree(tmp_path, [2023, 2024, 2025])

    report = build_report(_namespace(tmp_path, config=config, stage="raw"))

    assert report["status"] == "FAIL"
    assert any("excluded markets present" in item for item in report["coverage_errors"])
