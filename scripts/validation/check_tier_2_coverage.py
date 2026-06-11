#!/usr/bin/env python3
"""Validate tier-2 universe config and artifact coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


TIER_2_UNIVERSE = [
    "ES",
    "NQ",
    "RTY",
    "YM",
    "CL",
    "NG",
    "RB",
    "HO",
    "GC",
    "SI",
    "HG",
    "SR3",
    "ZN",
    "ZB",
    "6A",
    "6B",
    "6C",
    "6E",
    "6J",
    "6M",
    "6N",
    "6S",
    "ZC",
    "ZS",
    "ZW",
    "LE",
    "HE",
    "VX",
]

EXCLUDED = [
    "MES",
    "MNQ",
    "MYM",
    "M2K",
    "MCL",
    "MGC",
    "E7",
    "J7",
    "ZT",
    "ZF",
    "UB",
    "ZQ",
    "QO",
    "QI",
    "PA",
    "PL",
    "VXM",
]

REQUIRED_COST_KEYS = [
    "tick_size",
    "tick_value",
    "point_value",
    "min_profit_ticks",
    "min_stop_ticks",
    "commission_per_contract_dollars",
    "slippage_ticks_per_side",
    "round_turn_cost_ticks",
    "round_turn_cost_dollars",
    "cost_source",
    "provisional",
]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def resolve_profile_name(profile: str, aliases: dict[str, str]) -> str:
    seen: set[str] = set()
    resolved = profile
    while resolved in aliases and resolved not in seen:
        seen.add(resolved)
        resolved = aliases[resolved]
    return resolved


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value >= 0


def check_profile(config: dict[str, Any], requested_profile: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    profiles = config.get("profiles", {})
    aliases = config.get("aliases", {})
    if not isinstance(profiles, dict):
        return {}, ["alpha_tiered profiles mapping missing"]
    if not isinstance(aliases, dict):
        aliases = {}

    resolved = resolve_profile_name(requested_profile, {str(k): str(v) for k, v in aliases.items()})
    profile = profiles.get(resolved)
    if not isinstance(profile, dict):
        return {}, [f"profile {requested_profile!r} resolved to {resolved!r} but was not found"]

    markets = [str(item) for item in profile.get("markets", [])]
    if markets != TIER_2_UNIVERSE:
        errors.append("profile markets do not exactly match tier-2 universe order and membership")
    if len(markets) != 28 or len(set(markets)) != 28:
        errors.append("profile markets must contain exactly 28 unique markets")

    excluded_present = sorted(set(markets) & set(EXCLUDED))
    if excluded_present:
        errors.append(f"excluded markets present in profile: {','.join(excluded_present)}")

    years = [int(item) for item in profile.get("years", [])]
    if not years:
        errors.append("profile years missing")

    families = profile.get("market_families", {})
    if not isinstance(families, dict):
        errors.append("profile market_families mapping missing")
        families = {}
    missing_family = [market for market in TIER_2_UNIVERSE if not families.get(market)]
    if missing_family:
        errors.append(f"missing market_families: {','.join(missing_family)}")

    return (
        {
            "requested_profile": requested_profile,
            "resolved_profile": resolved,
            "markets": markets,
            "years": years,
            "excluded_present": excluded_present,
            "missing_family": missing_family,
        },
        errors,
    )


def check_sessions(session_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    templates = session_config.get("session_templates", {})
    markets = session_config.get("markets", {})
    if not isinstance(templates, dict):
        templates = {}
        errors.append("session_templates mapping missing")
    if not isinstance(markets, dict):
        markets = {}
        errors.append("session markets mapping missing")

    missing = [market for market in TIER_2_UNIVERSE if market not in markets]
    bad_template: list[str] = []
    for market in TIER_2_UNIVERSE:
        entry = markets.get(market, {})
        template = entry.get("session_template") if isinstance(entry, dict) else None
        if market in markets and template not in templates:
            bad_template.append(market)

    if missing:
        errors.append(f"missing session markets: {','.join(missing)}")
    if bad_template:
        errors.append(f"session template missing for markets: {','.join(bad_template)}")

    return {"missing_session": missing, "bad_session_template": bad_template}, errors


def check_costs(cost_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    markets = cost_config.get("markets", {})
    if not isinstance(markets, dict):
        markets = {}
        errors.append("cost markets mapping missing")

    missing = [market for market in TIER_2_UNIVERSE if market not in markets]
    missing_keys: dict[str, list[str]] = {}
    invalid_values: dict[str, list[str]] = {}
    provisional = []

    for market in TIER_2_UNIVERSE:
        entry = markets.get(market)
        if not isinstance(entry, dict):
            continue
        missing_for_market = [key for key in REQUIRED_COST_KEYS if key not in entry]
        if missing_for_market:
            missing_keys[market] = missing_for_market

        invalid_for_market = []
        for key in ("tick_size", "tick_value", "point_value", "round_turn_cost_ticks", "round_turn_cost_dollars"):
            if key in entry and not _positive_number(entry[key]):
                invalid_for_market.append(key)
        for key in ("min_profit_ticks", "min_stop_ticks", "commission_per_contract_dollars", "slippage_ticks_per_side"):
            if key in entry and not _non_negative_number(entry[key]):
                invalid_for_market.append(key)
        if "cost_source" in entry and not str(entry["cost_source"]).strip():
            invalid_for_market.append("cost_source")
        if "provisional" in entry and not isinstance(entry["provisional"], bool):
            invalid_for_market.append("provisional")
        if invalid_for_market:
            invalid_values[market] = invalid_for_market
        if entry.get("provisional") is True:
            provisional.append(market)

    if missing:
        errors.append(f"missing cost markets: {','.join(missing)}")
    if missing_keys:
        errors.append("missing cost keys")
    if invalid_values:
        errors.append("invalid cost values")

    return (
        {
            "missing_cost": missing,
            "missing_cost_keys": missing_keys,
            "invalid_cost_values": invalid_values,
            "provisional_cost_markets": provisional,
        },
        errors,
    )


def check_files(root: Path, years: list[int]) -> dict[str, Any]:
    missing: list[str] = []
    present: list[str] = []
    by_market: dict[str, dict[str, list[int]]] = {}
    for market in TIER_2_UNIVERSE:
        market_present: list[int] = []
        market_missing: list[int] = []
        for year in years:
            path = root / market / f"{year}.parquet"
            if path.exists():
                present.append(path.as_posix())
                market_present.append(year)
            else:
                missing.append(path.as_posix())
                market_missing.append(year)
        by_market[market] = {"present_years": market_present, "missing_years": market_missing}
    return {"root": root.as_posix(), "present": present, "missing": missing, "by_market": by_market}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(Path(args.config))
    session_config = load_yaml(Path(args.session_config))
    cost_config = load_yaml(Path(args.costs_config))

    profile_info, profile_errors = check_profile(config, args.profile)
    session_info, session_errors = check_sessions(session_config)
    cost_info, cost_errors = check_costs(cost_config)

    years = profile_info.get("years", [])
    artifact_checks: dict[str, Any] = {}
    if args.stage in {"raw", "all"}:
        artifact_checks["raw"] = check_files(Path(args.raw_root), years)
    if args.stage in {"causal", "all"}:
        artifact_checks["causal"] = check_files(Path(args.causal_root), years)
    if args.stage in {"labels", "all"}:
        artifact_checks["labels"] = check_files(Path(args.labeled_root), years)

    artifact_errors = []
    for name, check in artifact_checks.items():
        if check["missing"]:
            artifact_errors.append(f"missing {name} files: {len(check['missing'])}")

    config_errors = profile_errors + session_errors + cost_errors
    coverage_errors = config_errors + artifact_errors
    provisional_costs = cost_info.get("provisional_cost_markets", [])
    production_alpha_cost_gate = {
        "name": "non_provisional_costs_required_for_production_alpha_evidence",
        "status": "FAIL" if provisional_costs else "PASS",
        "provisional_cost_markets": provisional_costs,
    }

    return {
        "profile": profile_info,
        "stage": args.stage,
        "tier_2_universe": TIER_2_UNIVERSE,
        "excluded": EXCLUDED,
        "config_checks": {
            "profile": {"errors": profile_errors, **profile_info},
            "sessions": {"errors": session_errors, **session_info},
            "costs": {"errors": cost_errors, **cost_info},
        },
        "artifact_checks": artifact_checks,
        "hard_gates": {
            "production_alpha_cost_gate": production_alpha_cost_gate,
        },
        "production_alpha_evidence_ready": production_alpha_cost_gate["status"] == "PASS",
        "status": "FAIL" if coverage_errors else "PASS",
        "coverage_errors": coverage_errors,
        "config_error_count": len(config_errors),
        "artifact_error_count": len(artifact_errors),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="tier_2_universe_recent")
    parser.add_argument("--stage", choices=["raw", "causal", "labels", "all"], default="all")
    parser.add_argument("--config", default="configs/alpha_tiered.yaml")
    parser.add_argument("--session-config", default="configs/market_sessions.yaml")
    parser.add_argument("--costs-config", default="configs/costs.yaml")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--causal-root", default="data/causally_gated_normalized")
    parser.add_argument("--labeled-root", default="data/labeled")
    parser.add_argument("--report-out", default="reports/validation/tier_2_coverage.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = build_report(args)
    write_report(Path(args.report_out), report)

    missing = {
        name: len(check["missing"])
        for name, check in report["artifact_checks"].items()
        if check["missing"]
    }
    print(
        f"{report['status']} profile={args.profile} stage={args.stage} "
        f"config_errors={report['config_error_count']} artifact_errors={report['artifact_error_count']} "
        f"missing={missing} production_alpha_cost_gate="
        f"{report['hard_gates']['production_alpha_cost_gate']['status']} "
        f"report={Path(args.report_out).as_posix()}"
    )
    return 1 if report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
