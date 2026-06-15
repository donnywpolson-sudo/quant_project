#!/usr/bin/env python3
"""Run post-WFA anti-overfit audit, ledger append, and report write."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.ledger import (
    DEFAULT_LEDGER_PATH,
    append_ledger_record,
    build_ledger_record,
)
from scripts.experiments.robustness_gate import evaluate_robustness_gate


DEFAULT_AUDIT_REPORT = Path("reports/experiments/anti_overfit_audit.json")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_anti_overfit_audit(
    *,
    metrics_json: Path,
    wfa_report_json: Path | None = None,
    split_plan_json: Path | None = None,
    failure_breakdown_json: list[Path] | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    audit_report_json: Path = DEFAULT_AUDIT_REPORT,
    profile: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    failure_breakdown_json = failure_breakdown_json or []
    metrics_report = _read_json(metrics_json)
    wfa_report = _read_json(wfa_report_json)
    split_plan = _read_json(split_plan_json)
    failure_breakdowns = [_read_json(path) for path in failure_breakdown_json]

    robustness = evaluate_robustness_gate(
        metrics_report=metrics_report,
        wfa_report=wfa_report,
        split_plan=split_plan,
        failure_breakdowns=failure_breakdowns,
    )
    source_report_paths = {
        "metrics_json": metrics_json.as_posix(),
        "wfa_report_json": wfa_report_json.as_posix() if wfa_report_json else None,
        "split_plan_json": split_plan_json.as_posix() if split_plan_json else None,
        "failure_breakdown_json": [path.as_posix() for path in failure_breakdown_json],
    }
    audit_report = {
        "timestamp": timestamp,
        "profile": profile,
        "command": command,
        "robustness_status": robustness["status"],
        "failures": robustness["failures"],
        "checks": robustness["checks"],
        "available_breakdowns": robustness["breakdowns"],
        "ledger_path": ledger_path.as_posix(),
        "source_report_paths": source_report_paths,
    }
    _write_json(audit_report_json, audit_report)

    ledger_record = build_ledger_record(
        metrics=metrics_report,
        wfa_report=wfa_report,
        split_plan=split_plan,
        failure_breakdowns=failure_breakdowns,
        command=command,
        profile=profile,
        timestamp=timestamp,
    )
    ledger_record.update(
        {
            "passed": robustness["status"] == "PASS",
            "pass_fail_reason": list(robustness["failures"]),
            "robustness_status": robustness["status"],
            "robustness_checks": robustness["checks"],
            "audit_report_path": audit_report_json.as_posix(),
        }
    )
    append_ledger_record(ledger_path, ledger_record)

    return {
        "audit_report": audit_report,
        "audit_report_path": audit_report_json,
        "ledger_path": ledger_path,
        "ledger_record": ledger_record,
        "robustness": robustness,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--wfa-report-json", default=None)
    parser.add_argument("--split-plan-json", default=None)
    parser.add_argument("--failure-breakdown-json", action="append", default=[])
    parser.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH.as_posix())
    parser.add_argument("--audit-report-json", default=DEFAULT_AUDIT_REPORT.as_posix())
    parser.add_argument("--profile", default=None)
    parser.add_argument("--command", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_anti_overfit_audit(
        metrics_json=Path(args.metrics_json),
        wfa_report_json=Path(args.wfa_report_json) if args.wfa_report_json else None,
        split_plan_json=Path(args.split_plan_json) if args.split_plan_json else None,
        failure_breakdown_json=[Path(path) for path in args.failure_breakdown_json],
        ledger_path=Path(args.ledger_path),
        audit_report_json=Path(args.audit_report_json),
        profile=args.profile,
        command=args.command,
    )
    robustness = result["robustness"]
    print(
        f"{robustness['status']} anti-overfit audit: "
        f"failures={robustness['failure_count']} "
        f"report={result['audit_report_path'].as_posix()} "
        f"ledger={result['ledger_path'].as_posix()}"
    )
    return 0 if robustness["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
