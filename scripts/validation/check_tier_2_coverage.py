#!/usr/bin/env python3
"""Validate full-universe tier config and artifact coverage."""

from __future__ import annotations

import argparse
import hashlib
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
    "PL",
    "SR3",
    "ZT",
    "ZF",
    "ZN",
    "ZB",
    "UB",
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
]

EXCLUDED = [
    "E7",
    "J7",
    "PA",
    "QI",
    "QO",
    "ZQ",
]

PRODUCT_AVAILABLE_START_YEAR = {
    "RTY": 2017,
    "SR3": 2018,
}

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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    unknown_markets = sorted(set(markets) - set(TIER_2_UNIVERSE))
    duplicate_markets = sorted({market for market in markets if markets.count(market) > 1})
    if not markets:
        errors.append("profile markets missing")
    if duplicate_markets:
        errors.append(f"profile markets must be unique: {','.join(duplicate_markets)}")
    if unknown_markets:
        errors.append(f"profile markets outside supported universe: {','.join(unknown_markets)}")
    if resolved.startswith("tier_3") and markets != TIER_2_UNIVERSE:
        errors.append("profile markets do not exactly match full-universe tier order and membership")
    if resolved.startswith("tier_3") and (len(markets) != 31 or len(set(markets)) != 31):
        errors.append("profile markets must contain exactly 31 unique markets")

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
    missing_family = [market for market in markets if not families.get(market)]
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


def check_live_readiness(
    session_config: dict[str, Any],
    cost_config: dict[str, Any],
) -> dict[str, Any]:
    cost_model = cost_config.get("cost_model", {})
    if not isinstance(cost_model, dict):
        cost_model = {}
    calendar_refresh = session_config.get("live_calendar_refresh", {})
    if not isinstance(calendar_refresh, dict):
        calendar_refresh = {}

    contract_execution_mapping_ready = False
    calendar_refresh_current = calendar_refresh.get("current") is True
    live_fill_model_available = cost_model.get("live_fill_model_available") is True
    fixed_slippage = str(cost_model.get("slippage_source", "")).startswith(
        "approved_internal_model_assumption"
    )
    blocking_reasons: list[str] = []
    if not contract_execution_mapping_ready:
        blocking_reasons.append("contract_specific_execution_mapping_missing")
    if not calendar_refresh_current:
        blocking_reasons.append("current_exchange_calendar_refresh_missing")
    if not live_fill_model_available or fixed_slippage:
        blocking_reasons.append("live_fill_or_slippage_model_missing")

    return {
        "live_trading_ready": not blocking_reasons,
        "contract_execution_mapping_ready": contract_execution_mapping_ready,
        "calendar_refresh_current": calendar_refresh_current,
        "live_fill_model_available": live_fill_model_available,
        "fixed_slippage_research_assumption": fixed_slippage,
        "blocking_reasons": blocking_reasons,
        "status": "PASS" if not blocking_reasons else "FAIL",
    }


def check_files(root: Path, markets: list[str], years: list[int]) -> dict[str, Any]:
    missing: list[str] = []
    present: list[str] = []
    by_market: dict[str, dict[str, list[int]]] = {}
    unavailable_by_market: dict[str, list[int]] = {}
    for market in markets:
        market_present: list[int] = []
        market_missing: list[int] = []
        available_start_year = PRODUCT_AVAILABLE_START_YEAR.get(market)
        unavailable_years = [
            year for year in years if available_start_year is not None and year < available_start_year
        ]
        if unavailable_years:
            unavailable_by_market[market] = unavailable_years
        expected_years = [
            year for year in years if available_start_year is None or year >= available_start_year
        ]
        for year in expected_years:
            path = root / market / f"{year}.parquet"
            if path.exists():
                present.append(path.as_posix())
                market_present.append(year)
            else:
                missing.append(path.as_posix())
                market_missing.append(year)
        by_market[market] = {
            "present_years": market_present,
            "missing_years": market_missing,
            "unavailable_years": unavailable_years,
        }
    return {
        "root": root.as_posix(),
        "present": present,
        "missing": missing,
        "unavailable_by_market": unavailable_by_market,
        "by_market": by_market,
    }


def check_non_canonical_feature_artifacts(
    feature_root: Path,
    canonical_feature_root: Path,
) -> dict[str, Any]:
    artifacts: list[dict[str, str]] = []
    if feature_root.exists():
        for market_dir in sorted(path for path in feature_root.iterdir() if path.is_dir()):
            if market_dir.resolve() == canonical_feature_root.resolve():
                continue
            for path in sorted(market_dir.glob("*.parquet")):
                canonical_path = canonical_feature_root / market_dir.name / path.name
                if canonical_path.exists():
                    artifacts.append(
                        {
                            "artifact_path": path.as_posix(),
                            "canonical_path": canonical_path.as_posix(),
                            "artifact_sha256": file_sha256(path),
                        }
                    )
    return {
        "canonical_feature_root": canonical_feature_root.as_posix(),
        "feature_root": feature_root.as_posix(),
        "non_canonical_feature_artifact_count": len(artifacts),
        "non_canonical_feature_artifacts": artifacts,
    }


def quarantined_feature_artifacts(
    artifacts: list[dict[str, str]],
    quarantine: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw_entries = quarantine.get("non_canonical_feature_artifacts", [])
    entries = raw_entries if isinstance(raw_entries, list) else []
    quarantined: list[dict[str, str]] = []
    unquarantined: list[dict[str, str]] = []
    for artifact in artifacts:
        matched = False
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                continue
            if (
                raw_entry.get("artifact_path") == artifact["artifact_path"]
                and raw_entry.get("canonical_path") == artifact["canonical_path"]
                and raw_entry.get("artifact_sha256") == artifact["artifact_sha256"]
            ):
                matched = True
                break
        if matched:
            quarantined.append(artifact)
        else:
            unquarantined.append(artifact)
    return quarantined, unquarantined


def check_prediction_evidence_manifests(wfa_reports_root: Path) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    failures: list[str] = []
    if not wfa_reports_root.exists():
        return {
            "wfa_reports_root": wfa_reports_root.as_posix(),
            "manifest_count": 0,
            "invalid_manifest_count": 0,
            "manifests": manifests,
            "failures": failures,
        }

    for path in sorted(wfa_reports_root.glob("*_predictions_manifest.json")):
        item_failures: list[str] = []
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            item_failures.append(f"invalid json: {exc}")
            manifest = {}
        if int(manifest.get("failure_count") or 0) > 0:
            item_failures.append("failure_count is nonzero")
        if int(manifest.get("prediction_count") or 0) <= 0:
            item_failures.append("prediction_count is zero")
        output_hashes = manifest.get("output_file_hashes", {})
        if isinstance(output_hashes, dict) and any(
            value == "NOT_WRITTEN" for value in output_hashes.values()
        ):
            item_failures.append("output hash is NOT_WRITTEN")
        if manifest.get("stale_output_path_exists") is True:
            item_failures.append("stale prediction output exists")
        if manifest.get("artifact_evidence_ready") is False:
            item_failures.append("artifact_evidence_ready is false")
        if item_failures:
            failures.append(f"{path.as_posix()}: {'; '.join(item_failures)}")
        manifests.append(
            {
                "manifest_path": path.as_posix(),
                "artifact_evidence_ready": not item_failures,
                "failure_count": int(manifest.get("failure_count") or 0),
                "prediction_count": int(manifest.get("prediction_count") or 0),
                "stale_output_path_exists": manifest.get("stale_output_path_exists") is True,
                "failures": item_failures,
            }
        )

    return {
        "wfa_reports_root": wfa_reports_root.as_posix(),
        "manifest_count": len(manifests),
        "invalid_manifest_count": sum(not item["artifact_evidence_ready"] for item in manifests),
        "manifests": manifests,
        "failures": failures,
    }


def check_research_alpha_promotion(
    metrics_root: Path,
    model_selection_root: Path,
) -> dict[str, Any]:
    metrics_path = metrics_root / "baseline_metrics.json"
    selection_path = model_selection_root / "model_selection_report.json"
    failures: list[str] = []
    promotion_blockers: list[str] = []
    metrics: dict[str, Any] = {}
    selection: dict[str, Any] = {}

    if not metrics_path.exists():
        failures.append(f"missing metrics report: {metrics_path.as_posix()}")
    else:
        try:
            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError as exc:
            failures.append(f"invalid metrics report json: {exc}")
    if not selection_path.exists():
        failures.append(f"missing model selection report: {selection_path.as_posix()}")
    else:
        try:
            loaded = json.loads(selection_path.read_text(encoding="utf-8"))
            selection = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError as exc:
            failures.append(f"invalid model selection report json: {exc}")

    for name, payload in (("metrics", metrics), ("model_selection", selection)):
        if not payload:
            continue
        if int(payload.get("failure_count") or 0) > 0:
            failures.append(f"{name} report failure_count is nonzero")
        if payload.get("research_alpha_ready") is not True:
            failures.append(f"{name} report research_alpha_ready is not true")
        if payload.get("model_promotion_allowed") is not True:
            failures.append(f"{name} report model_promotion_allowed is not true")
        gate = payload.get("promotion_gate", {})
        if isinstance(gate, dict):
            raw_blockers = gate.get("promotion_blockers", [])
            if isinstance(raw_blockers, list):
                promotion_blockers.extend(str(item) for item in raw_blockers)

    return {
        "name": "research_alpha_requires_positive_net_stable_costed_policy",
        "status": "FAIL" if failures else "PASS",
        "metrics_report_path": metrics_path.as_posix(),
        "model_selection_report_path": selection_path.as_posix(),
        "research_alpha_ready": not failures,
        "model_promotion_allowed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "promotion_blockers": sorted(set(promotion_blockers)),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(Path(args.config))
    session_config = load_yaml(Path(args.session_config))
    cost_config = load_yaml(Path(args.costs_config))
    artifact_quarantine_path = getattr(
        args, "artifact_quarantine", "reports/validation/artifact_quarantine.json"
    )
    artifact_quarantine = load_json(Path(artifact_quarantine_path))

    profile_info, profile_errors = check_profile(config, args.profile)
    session_info, session_errors = check_sessions(session_config)
    cost_info, cost_errors = check_costs(cost_config)

    markets = profile_info.get("markets", [])
    years = profile_info.get("years", [])
    artifact_checks: dict[str, Any] = {}
    if args.stage in {"raw", "all"}:
        artifact_checks["raw"] = check_files(Path(args.raw_root), markets, years)
    if args.stage in {"causal", "all"}:
        artifact_checks["causal"] = check_files(Path(args.causal_root), markets, years)
    if args.stage in {"labels", "all"}:
        artifact_checks["labels"] = check_files(Path(args.labeled_root), markets, years)
    if args.stage in {"features", "all"}:
        artifact_checks["features"] = check_files(
            Path(args.canonical_feature_root),
            markets,
            years,
        )
    feature_artifact_warnings = check_non_canonical_feature_artifacts(
        Path(getattr(args, "feature_root", "data/feature_matrices")),
        Path(getattr(args, "canonical_feature_root", "data/feature_matrices/baseline")),
    )
    quarantined_features, unquarantined_features = quarantined_feature_artifacts(
        feature_artifact_warnings["non_canonical_feature_artifacts"],
        artifact_quarantine,
    )
    prediction_evidence = check_prediction_evidence_manifests(
        Path(getattr(args, "wfa_reports_root", "reports/wfa"))
    )
    research_alpha = check_research_alpha_promotion(
        Path(getattr(args, "metrics_root", "reports/metrics")),
        Path(getattr(args, "model_selection_root", "reports/model_selection")),
    )

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
    artifact_evidence_failures = []
    if unquarantined_features:
        artifact_evidence_failures.append(
            "unquarantined non-canonical feature artifacts exist alongside canonical baseline outputs"
        )
    artifact_evidence_failures.extend(prediction_evidence["failures"])
    artifact_evidence_gate = {
        "name": "canonical_artifact_lineage_required_for_research_evidence",
        "status": "FAIL" if artifact_evidence_failures else "PASS",
        "failures": artifact_evidence_failures,
        "canonical_feature_root": feature_artifact_warnings["canonical_feature_root"],
        "non_canonical_feature_artifact_count": feature_artifact_warnings[
            "non_canonical_feature_artifact_count"
        ],
        "quarantined_non_canonical_feature_artifact_count": len(quarantined_features),
        "unquarantined_non_canonical_feature_artifact_count": len(unquarantined_features),
        "invalid_prediction_manifest_count": prediction_evidence["invalid_manifest_count"],
    }
    live_readiness = check_live_readiness(session_config, cost_config)

    return {
        "profile": profile_info,
        "stage": args.stage,
        "full_universe": TIER_2_UNIVERSE,
        "excluded": EXCLUDED,
        "config_checks": {
            "profile": {"errors": profile_errors, **profile_info},
            "sessions": {"errors": session_errors, **session_info},
            "costs": {"errors": cost_errors, **cost_info},
        },
        "artifact_checks": artifact_checks,
        "artifact_warnings": {
            "features": feature_artifact_warnings,
            "prediction_manifests": prediction_evidence,
        },
        "canonical_feature_root": feature_artifact_warnings["canonical_feature_root"],
        "non_canonical_feature_artifact_count": feature_artifact_warnings[
            "non_canonical_feature_artifact_count"
        ],
        "non_canonical_feature_artifacts": feature_artifact_warnings[
            "non_canonical_feature_artifacts"
        ],
        "quarantined_non_canonical_feature_artifacts": quarantined_features,
        "unquarantined_non_canonical_feature_artifacts": unquarantined_features,
        "artifact_quarantine_path": artifact_quarantine_path,
        "hard_gates": {
            "production_alpha_cost_gate": production_alpha_cost_gate,
            "artifact_evidence_gate": artifact_evidence_gate,
            "research_alpha_promotion_gate": research_alpha,
            "live_trading_readiness_gate": {
                "name": "live_trading_requires_contract_mapping_current_calendar_and_live_fill_model",
                **live_readiness,
            },
        },
        "production_alpha_evidence_ready": production_alpha_cost_gate["status"] == "PASS",
        "artifact_evidence_ready": artifact_evidence_gate["status"] == "PASS",
        "artifact_evidence_failures": artifact_evidence_failures,
        "research_alpha_ready": research_alpha["status"] == "PASS",
        "model_promotion_allowed": research_alpha["model_promotion_allowed"],
        "research_pipeline_ready": not coverage_errors
        and production_alpha_cost_gate["status"] == "PASS",
        "live_trading_ready": live_readiness["live_trading_ready"],
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
    parser.add_argument("--profile", default="tier_3")
    parser.add_argument(
        "--stage",
        choices=["raw", "causal", "labels", "features", "all"],
        default="all",
    )
    parser.add_argument("--config", default="configs/alpha_tiered.yaml")
    parser.add_argument("--session-config", default="configs/market_sessions.yaml")
    parser.add_argument("--costs-config", default="configs/costs.yaml")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--causal-root", default="data/causally_gated_normalized")
    parser.add_argument("--labeled-root", default="data/labeled")
    parser.add_argument("--feature-root", default="data/feature_matrices")
    parser.add_argument("--canonical-feature-root", default="data/feature_matrices/baseline")
    parser.add_argument("--wfa-reports-root", default="reports/wfa")
    parser.add_argument("--metrics-root", default="reports/metrics")
    parser.add_argument("--model-selection-root", default="reports/model_selection")
    parser.add_argument("--artifact-quarantine", default="reports/validation/artifact_quarantine.json")
    parser.add_argument("--report-out", default="reports/validation/full_universe_coverage.json")
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
        f"artifact_evidence_ready={report['artifact_evidence_ready']} "
        f"live_trading_ready={report['live_trading_ready']} "
        f"report={Path(args.report_out).as_posix()}"
    )
    return 1 if report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
