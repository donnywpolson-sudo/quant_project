#!/usr/bin/env python3
"""Append experiment summaries to an audit ledger."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_LEDGER_PATH = Path("reports/experiments/ledger.jsonl")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_present(*values: object) -> object:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _metric_summaries(metrics: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    nested = metrics.get("metrics", {})
    summaries = nested.get("summaries", []) if isinstance(nested, Mapping) else []
    if not isinstance(summaries, list):
        return []
    return [
        dict(item)
        for item in summaries
        if isinstance(item, Mapping) and item.get("scope") == scope
    ]


def _failure_reasons(*reports: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for report in reports:
        failure_count = _safe_int(report.get("failure_count")) or 0
        failures = report.get("failures", [])
        if failure_count > 0 and not failures:
            reasons.append("failure_count is nonzero")
        if isinstance(failures, list):
            reasons.extend(str(item) for item in failures)
        promotion = report.get("promotion_gate", {})
        if isinstance(promotion, Mapping) and promotion.get("model_promotion_allowed") is False:
            blockers = promotion.get("promotion_blockers", [])
            if isinstance(blockers, list):
                reasons.extend(str(item) for item in blockers)
    return reasons


def build_ledger_record(
    *,
    metrics: Mapping[str, Any] | None = None,
    wfa_report: Mapping[str, Any] | None = None,
    split_plan: Mapping[str, Any] | None = None,
    failure_breakdowns: Iterable[Mapping[str, Any]] = (),
    command: str | None = None,
    profile: str | None = None,
    configs: Iterable[str] = (),
    timestamp: str | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    wfa_report = wfa_report or {}
    split_plan = split_plan or {}
    breakdowns = [dict(item) for item in failure_breakdowns]
    overall = metrics.get("metrics", {}).get("overall", {}) if isinstance(metrics.get("metrics"), Mapping) else {}
    if not isinstance(overall, Mapping):
        overall = {}
    reasons = _failure_reasons(metrics, wfa_report, split_plan, *breakdowns)
    side_breakdown: list[dict[str, Any]] = []
    for breakdown in breakdowns:
        side_damage = breakdown.get("side_damage", [])
        if isinstance(side_damage, list):
            side_breakdown.extend(dict(item) for item in side_damage if isinstance(item, Mapping))

    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "git_hash": _first_present(
            metrics.get("git_commit"),
            wfa_report.get("git_commit"),
            split_plan.get("git_commit"),
            _git_commit(),
        ),
        "command": command,
        "profile": _first_present(profile, metrics.get("profile"), wfa_report.get("profile"), split_plan.get("profile")),
        "configs": list(configs)
        or [
            value
            for value in (metrics.get("costs_config"), metrics.get("models_config"))
            if value is not None
        ],
        "markets": _first_present(split_plan.get("markets"), wfa_report.get("markets")),
        "years": split_plan.get("years"),
        "cost_assumptions": {
            "costs_config": metrics.get("costs_config"),
            "execution_realism": metrics.get("execution_realism"),
            "policy_config": metrics.get("policy_config"),
        },
        "gross_return_dollars": overall.get("gross_return_dollars"),
        "net_return_dollars": overall.get("net_return_dollars"),
        "cost_dollars": overall.get("cost_dollars"),
        "cost_drag_to_abs_gross": overall.get("cost_drag_to_abs_gross"),
        "slippage_cost_dollars": overall.get("slippage_cost_dollars"),
        "commission_cost_dollars": overall.get("commission_cost_dollars"),
        "turnover_per_bar": overall.get("turnover_per_bar"),
        "trades": overall.get("trade_count"),
        "folds": _first_present(wfa_report.get("fold_count"), split_plan.get("fold_count")),
        "market_breakdown": _metric_summaries(metrics, "market") or metrics.get("markets"),
        "side_breakdown": side_breakdown,
        "passed": not reasons,
        "pass_fail_reason": reasons,
    }


def append_ledger_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str))
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH.as_posix())
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--wfa-report-json", default=None)
    parser.add_argument("--split-plan-json", default=None)
    parser.add_argument("--failure-breakdown-json", action="append", default=[])
    parser.add_argument("--command", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--config", action="append", default=[])
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    record = build_ledger_record(
        metrics=_read_json(Path(args.metrics_json)) if args.metrics_json else {},
        wfa_report=_read_json(Path(args.wfa_report_json)) if args.wfa_report_json else {},
        split_plan=_read_json(Path(args.split_plan_json)) if args.split_plan_json else {},
        failure_breakdowns=[
            _read_json(Path(path)) for path in args.failure_breakdown_json
        ],
        command=args.command,
        profile=args.profile,
        configs=args.config,
    )
    append_ledger_record(Path(args.ledger_path), record)
    print(f"ledger_appended path={args.ledger_path} passed={record['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
