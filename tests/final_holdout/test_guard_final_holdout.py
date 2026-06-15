from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.final_holdout.guard_final_holdout import (
    final_holdout_permission_failure,
    is_final_holdout_year_set,
    validate_final_holdout_guard,
)


def _write_freeze_manifest(root: Path, freeze_id: str) -> Path:
    path = root / freeze_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "frozen": True,
                "failure_count": 0,
                "final_holdout_consumes_frozen_only": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_final_holdout_guard_writes_valid_metrics_from_frozen_manifest(tmp_path: Path) -> None:
    freeze_root = tmp_path / "artifacts" / "frozen"
    _write_freeze_manifest(freeze_root, "freeze-1")

    metrics = validate_final_holdout_guard(
        frozen_artifact_id="freeze-1",
        freeze_root=freeze_root,
        reports_root=tmp_path / "reports" / "final_holdout",
        run_id="final-smoke",
    )

    assert metrics["validity"] == "PASS"
    assert metrics["used_final_holdout_for_tuning"] is False
    assert metrics["frozen_artifact_id"] == "freeze-1"
    saved = json.loads(
        (tmp_path / "reports" / "final_holdout" / "final_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["run_id"] == "final-smoke"


def test_final_holdout_guard_refuses_tuning_and_policy_changes(tmp_path: Path) -> None:
    freeze_root = tmp_path / "artifacts" / "frozen"
    _write_freeze_manifest(freeze_root, "freeze-1")

    metrics = validate_final_holdout_guard(
        frozen_artifact_id="freeze-1",
        freeze_root=freeze_root,
        reports_root=tmp_path / "reports" / "final_holdout",
        allow_tuning=True,
        allow_policy_change=True,
    )

    assert metrics["validity"] == "FAIL"
    assert metrics["used_final_holdout_for_tuning"] is False
    assert "final holdout tuning requested" in metrics["failures"]
    assert "final holdout policy change requested" in metrics["failures"]


def test_final_holdout_permission_helper_requires_explicit_allow() -> None:
    assert is_final_holdout_year_set([2025], [2025]) is True
    assert is_final_holdout_year_set([2024, 2025], [2025]) is False

    failure = final_holdout_permission_failure(
        is_final_holdout=True,
        allow_final_holdout=False,
        action="final-holdout split-plan generation",
    )

    assert failure == "final-holdout split-plan generation requires --allow-final-holdout"
    assert (
        final_holdout_permission_failure(
            is_final_holdout=True,
            allow_final_holdout=True,
            action="final-holdout split-plan generation",
        )
        is None
    )
