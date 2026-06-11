from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

from scripts.validation.check_target_28_coverage import (
    EXCLUDED,
    REQUIRED_COST_KEYS,
    TARGET_28,
    build_report,
    load_yaml,
    resolve_profile_name,
)


ROOT = Path(__file__).resolve().parents[2]


def _namespace(tmp_path: Path, *, config: Path, stage: str = "all") -> Namespace:
    return Namespace(
        profile="target_28_recent",
        stage=stage,
        config=str(config),
        session_config=str(ROOT / "configs" / "market_sessions.yaml"),
        costs_config=str(ROOT / "configs" / "costs.yaml"),
        raw_root=str(tmp_path / "data" / "raw"),
        causal_root=str(tmp_path / "data" / "causally_gated_normalized"),
        labeled_root=str(tmp_path / "data" / "labeled"),
        report_out=str(tmp_path / "reports" / "validation" / "target_28_coverage.json"),
    )


def _touch_complete_tree(tmp_path: Path, years: list[int]) -> None:
    for root_name in ("raw", "causally_gated_normalized", "labeled"):
        root = tmp_path / "data" / root_name
        for market in TARGET_28:
            for year in years:
                path = root / market / f"{year}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder", encoding="utf-8")


def test_target_28_profiles_exist_and_match_exact_universe() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    profiles = config["profiles"]
    aliases = config["aliases"]

    assert "target_28_recent" in profiles
    assert "target_28_long" in profiles
    assert aliases["target_28"] == "target_28_recent"
    assert aliases["target_28_recent"] == "target_28_recent"
    assert aliases["target_28_long"] == "target_28_long"
    assert resolve_profile_name("target_28", aliases) == "target_28_recent"

    assert profiles["target_28_recent"]["markets"] == TARGET_28
    assert profiles["target_28_recent"]["years"] == [2023, 2024, 2025]
    assert profiles["target_28_long"]["markets"] == TARGET_28
    assert profiles["target_28_long"]["years"] == list(range(2010, 2026))


def test_target_28_profiles_exclude_forbidden_markets() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    for profile_name in ("target_28_recent", "target_28_long"):
        markets = set(config["profiles"][profile_name]["markets"])
        assert markets.isdisjoint(EXCLUDED)


def test_every_target_28_market_has_family_session_cost_and_tick_coverage() -> None:
    config = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    sessions = load_yaml(ROOT / "configs" / "market_sessions.yaml")
    costs = load_yaml(ROOT / "configs" / "costs.yaml")

    families = config["profiles"]["target_28_recent"]["market_families"]
    session_markets = sessions["markets"]
    templates = sessions["session_templates"]
    cost_markets = costs["markets"]

    for market in TARGET_28:
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
    (tmp_path / "data" / "raw" / "6A" / "2023.parquet").unlink()

    report = build_report(_namespace(tmp_path, config=config, stage="raw"))

    assert report["status"] == "FAIL"
    assert "data/raw/6A/2023.parquet" in report["artifact_checks"]["raw"]["missing"][0]


def test_coverage_gate_fails_if_excluded_market_is_inserted(tmp_path: Path) -> None:
    payload = load_yaml(ROOT / "configs" / "alpha_tiered.yaml")
    payload["profiles"]["target_28_recent"]["markets"] = TARGET_28 + ["MES"]
    config = tmp_path / "alpha_tiered.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _touch_complete_tree(tmp_path, [2023, 2024, 2025])

    report = build_report(_namespace(tmp_path, config=config, stage="raw"))

    assert report["status"] == "FAIL"
    assert any("excluded markets present" in item for item in report["coverage_errors"])
