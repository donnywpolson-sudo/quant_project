#!/usr/bin/env python3
"""Evaluate anti-overfit robustness checks from existing report JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metrics_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = report.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _overall(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = _metrics_payload(report)
    overall = metrics.get("overall", report.get("costed_oos", {}))
    return overall if isinstance(overall, Mapping) else {}


def _summaries(report: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    metrics = _metrics_payload(report)
    summaries = metrics.get("summaries", [])
    if isinstance(summaries, list):
        return [
            dict(item)
            for item in summaries
            if isinstance(item, Mapping) and item.get("scope") == scope
        ]
    fallback = report.get(f"{scope}s", [])
    return [dict(item) for item in fallback if isinstance(item, Mapping)] if isinstance(fallback, list) else []


def _has_failure_markers(reports: Iterable[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for report in reports:
        failure_count = _safe_int(report.get("failure_count")) or 0
        if failure_count > 0:
            reasons.append("failure_count is nonzero")
        failures = report.get("failures", [])
        if isinstance(failures, list) and failures:
            reasons.extend(str(item) for item in failures)
        if report.get("artifact_evidence_ready") is False:
            reasons.append("artifact_evidence_ready is false")
    return reasons


def _cost_stress(overall: Mapping[str, Any], multiplier: float) -> tuple[float | None, str | None]:
    gross = _safe_float(overall.get("gross_return_dollars"))
    cost = _safe_float(overall.get("cost_dollars"))
    net = _safe_float(overall.get("net_return_dollars"))
    if gross is None or cost is None or net is None:
        return None, "cost_stress_unavailable"
    if abs((gross - cost) - net) > max(1e-6, abs(net) * 1e-9):
        return None, "cost_stress_unavailable"
    return gross - multiplier * cost, None


def _optional_breakdowns(failure_breakdowns: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    side_available = False
    hour_available = False
    regime_available = False
    unavailable: list[str] = []
    for breakdown in failure_breakdowns:
        side = breakdown.get("side_damage", [])
        side_available = side_available or (isinstance(side, list) and bool(side))
        outputs = breakdown.get("outputs", {})
        diagnostics = breakdown.get("unavailable_diagnostics", [])
        if isinstance(diagnostics, list):
            unavailable.extend(str(item) for item in diagnostics)
        if isinstance(outputs, Mapping):
            hour_available = hour_available or (
                "market_side_hour" in outputs
                and not any("market_side_hour" in item for item in unavailable)
            )
            regime_available = regime_available or (
                "regime" in outputs
                and not any("regime" in item for item in unavailable)
            )
    return {
        "side": "available" if side_available else "unavailable",
        "hour": "available" if hour_available else "unavailable",
        "session": "unavailable",
        "regime": "available" if regime_available else "unavailable",
    }


def evaluate_robustness_gate(
    *,
    metrics_report: Mapping[str, Any],
    wfa_report: Mapping[str, Any] | None = None,
    split_plan: Mapping[str, Any] | None = None,
    failure_breakdowns: Iterable[Mapping[str, Any]] = (),
    max_turnover_per_bar: float = 0.10,
    min_fold_pass_rate: float = 1.0,
    max_single_market_profit_contribution: float = 0.50,
) -> dict[str, Any]:
    wfa_report = wfa_report or {}
    split_plan = split_plan or {}
    breakdowns = [dict(item) for item in failure_breakdowns]
    overall = _overall(metrics_report)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    net = _safe_float(overall.get("net_return_dollars"))
    checks["base_cost_net"] = net
    if net is None or net <= 0.0:
        failures.append("base_net_nonpositive")

    for label, multiplier in (("cost_stress_1_5x", 1.5), ("cost_stress_2x", 2.0)):
        stressed_net, reason = _cost_stress(overall, multiplier)
        checks[label] = stressed_net
        if reason is not None:
            failures.append(reason)
        elif stressed_net is None or stressed_net <= 0.0:
            failures.append(f"{label}_nonpositive")

    turnover = _safe_float(overall.get("turnover_per_bar"))
    checks["turnover_per_bar"] = turnover
    if turnover is None:
        failures.append("turnover_unavailable")
    elif turnover > max_turnover_per_bar:
        failures.append("turnover_above_ceiling")

    market_breakdown = _summaries(metrics_report, "market")
    checks["market_count"] = len(market_breakdown)
    if not market_breakdown:
        failures.append("market_breakdown_unavailable")
    else:
        positive = [
            value
            for value in (_safe_float(item.get("net_return_dollars")) for item in market_breakdown)
            if value is not None and value > 0.0
        ]
        positive_total = sum(positive)
        contribution = max(positive) / positive_total if positive_total > 0.0 else None
        checks["max_single_market_profit_contribution"] = contribution
        if contribution is None or contribution > max_single_market_profit_contribution:
            failures.append("single_market_profit_contribution_above_cap")

    fold_breakdown = _summaries(metrics_report, "fold")
    checks["fold_count"] = len(fold_breakdown)
    if not fold_breakdown:
        failures.append("fold_pass_rate_unavailable")
    else:
        fold_nets = [_safe_float(item.get("net_return_dollars")) for item in fold_breakdown]
        if any(value is None for value in fold_nets):
            failures.append("fold_pass_rate_unavailable")
        else:
            pass_rate = sum(1 for value in fold_nets if value is not None and value > 0.0) / len(fold_nets)
            checks["fold_pass_rate"] = pass_rate
            if pass_rate < min_fold_pass_rate:
                failures.append("fold_pass_rate_below_minimum")

    marker_failures = _has_failure_markers([metrics_report, wfa_report, split_plan, *breakdowns])
    checks["validation_failure_markers"] = marker_failures
    if marker_failures:
        failures.append("validation_failure_markers_present")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "breakdowns": {
            "market": market_breakdown,
            "fold": fold_breakdown,
            "optional": _optional_breakdowns(breakdowns),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--wfa-report-json", default=None)
    parser.add_argument("--split-plan-json", default=None)
    parser.add_argument("--failure-breakdown-json", action="append", default=[])
    parser.add_argument("--max-turnover-per-bar", type=float, default=0.10)
    parser.add_argument("--min-fold-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-single-market-profit-contribution", type=float, default=0.50)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = evaluate_robustness_gate(
        metrics_report=_read_json(Path(args.metrics_json)),
        wfa_report=_read_json(Path(args.wfa_report_json)) if args.wfa_report_json else {},
        split_plan=_read_json(Path(args.split_plan_json)) if args.split_plan_json else {},
        failure_breakdowns=[
            _read_json(Path(path)) for path in args.failure_breakdown_json
        ],
        max_turnover_per_bar=args.max_turnover_per_bar,
        min_fold_pass_rate=args.min_fold_pass_rate,
        max_single_market_profit_contribution=args.max_single_market_profit_contribution,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
