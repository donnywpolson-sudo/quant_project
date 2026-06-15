#!/usr/bin/env python3
"""Validate final-holdout evaluation uses only frozen artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FREEZE_ROOT = Path("artifacts/frozen")
DEFAULT_REPORTS_ROOT = Path("reports/final_holdout")


def is_final_holdout_year_set(years: Iterable[int], final_holdout_years: Iterable[int]) -> bool:
    profile_years = {int(year) for year in years}
    final_years = {int(year) for year in final_holdout_years}
    return bool(profile_years) and bool(final_years) and profile_years <= final_years


def final_holdout_permission_failure(
    *,
    is_final_holdout: bool,
    allow_final_holdout: bool,
    action: str,
) -> str | None:
    if is_final_holdout and not allow_final_holdout:
        return f"{action} requires --allow-final-holdout"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_final_holdout_guard(
    *,
    frozen_artifact_id: str,
    freeze_root: Path = DEFAULT_FREEZE_ROOT,
    run_id: str = "final_holdout",
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    allow_tuning: bool = False,
    allow_feature_selection: bool = False,
    allow_calibration_change: bool = False,
    allow_policy_change: bool = False,
) -> dict[str, Any]:
    manifest_path = freeze_root / frozen_artifact_id / "manifest.json"
    freeze_manifest = _read_json(manifest_path)
    failures: list[str] = []
    if not freeze_manifest:
        failures.append(f"missing frozen manifest: {manifest_path.as_posix()}")
    else:
        if freeze_manifest.get("frozen") is not True:
            failures.append("frozen manifest is not marked frozen")
        if int(freeze_manifest.get("failure_count") or 0) > 0:
            failures.append("frozen manifest failure_count is nonzero")
        if freeze_manifest.get("final_holdout_consumes_frozen_only") is not True:
            failures.append("frozen manifest does not require frozen-only final holdout")

    if allow_tuning:
        failures.append("final holdout tuning requested")
    if allow_feature_selection:
        failures.append("final holdout feature selection requested")
    if allow_calibration_change:
        failures.append("final holdout calibration change requested")
    if allow_policy_change:
        failures.append("final holdout policy change requested")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "frozen_artifact_id": frozen_artifact_id,
        "frozen_manifest_path": manifest_path.as_posix(),
        "validity": "PASS" if not failures else "FAIL",
        "used_final_holdout_for_tuning": False,
        "loaded_frozen_artifacts_only": not any(
            [allow_tuning, allow_feature_selection, allow_calibration_change, allow_policy_change]
        ),
        "failure_count": len(failures),
        "failures": failures,
    }
    _write_json(reports_root / "final_metrics.json", metrics)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-artifact-id", required=True)
    parser.add_argument("--freeze-root", default=DEFAULT_FREEZE_ROOT.as_posix())
    parser.add_argument("--run-id", default="final_holdout")
    parser.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT.as_posix())
    parser.add_argument("--allow-tuning", action="store_true")
    parser.add_argument("--allow-feature-selection", action="store_true")
    parser.add_argument("--allow-calibration-change", action="store_true")
    parser.add_argument("--allow-policy-change", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    metrics = validate_final_holdout_guard(
        frozen_artifact_id=args.frozen_artifact_id,
        freeze_root=Path(args.freeze_root),
        run_id=args.run_id,
        reports_root=Path(args.reports_root),
        allow_tuning=args.allow_tuning,
        allow_feature_selection=args.allow_feature_selection,
        allow_calibration_change=args.allow_calibration_change,
        allow_policy_change=args.allow_policy_change,
    )
    print(
        f"{metrics['validity']} final holdout guard:"
        f" frozen_artifact_id={args.frozen_artifact_id}"
        f" failures={metrics['failure_count']}"
    )
    return 1 if metrics["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
