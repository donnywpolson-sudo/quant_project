from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.experiments.run_anti_overfit_audit as audit
from scripts.experiments.run_anti_overfit_audit import run_anti_overfit_audit


def _metrics(
    *,
    gross: float = 300.0,
    cost: float = 100.0,
    net: float = 200.0,
    turnover: float | None = 0.01,
    market_breakdown: bool = True,
    fold_breakdown: bool = True,
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    if market_breakdown:
        summaries.extend(
            [
                {"scope": "market", "market": "ES", "net_return_dollars": 100.0},
                {"scope": "market", "market": "CL", "net_return_dollars": 100.0},
            ]
        )
    if fold_breakdown:
        summaries.extend(
            [
                {"scope": "fold", "fold_id": "f1", "net_return_dollars": 100.0},
                {"scope": "fold", "fold_id": "f2", "net_return_dollars": 100.0},
            ]
        )
    overall = {
        "gross_return_dollars": gross,
        "cost_dollars": cost,
        "net_return_dollars": net,
    }
    if turnover is not None:
        overall["turnover_per_bar"] = turnover
    return {
        "git_commit": "abc123",
        "failure_count": 0,
        "failures": [],
        "metrics": {"overall": overall, "summaries": summaries},
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_ledger(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_passing_audit_writes_report_and_appends_ledger(tmp_path: Path) -> None:
    metrics_path = _write_json(tmp_path / "reports" / "metrics.json", _metrics())
    ledger_path = tmp_path / "reports" / "experiments" / "ledger.jsonl"
    audit_path = tmp_path / "reports" / "experiments" / "audit.json"

    result = run_anti_overfit_audit(
        metrics_json=metrics_path,
        ledger_path=ledger_path,
        audit_report_json=audit_path,
        profile="fixture",
        command="python fixture",
    )

    report = json.loads(audit_path.read_text(encoding="utf-8"))
    ledger = _read_ledger(ledger_path)
    assert result["robustness"]["status"] == "PASS"
    assert report["robustness_status"] == "PASS"
    assert report["source_report_paths"]["metrics_json"] == metrics_path.as_posix()
    assert ledger[0]["passed"] is True
    assert ledger[0]["pass_fail_reason"] == []
    assert ledger[0]["robustness_status"] == "PASS"
    assert ledger[0]["audit_report_path"] == audit_path.as_posix()


def test_failing_audit_writes_report_appends_ledger_and_main_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path = _write_json(
        tmp_path / "reports" / "metrics.json",
        _metrics(gross=190.0, cost=100.0, net=90.0),
    )
    ledger_path = tmp_path / "ledger.jsonl"
    audit_path = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_anti_overfit_audit.py",
            "--metrics-json",
            metrics_path.as_posix(),
            "--ledger-path",
            ledger_path.as_posix(),
            "--audit-report-json",
            audit_path.as_posix(),
        ],
    )

    exit_code = audit.main()

    report = json.loads(audit_path.read_text(encoding="utf-8"))
    ledger = _read_ledger(ledger_path)
    assert exit_code == 1
    assert report["robustness_status"] == "FAIL"
    assert "cost_stress_2x_nonpositive" in report["failures"]
    assert ledger[0]["passed"] is False
    assert "cost_stress_2x_nonpositive" in ledger[0]["pass_fail_reason"]
    assert ledger[0]["robustness_checks"]["cost_stress_2x"] == -10.0


def test_missing_required_evidence_fails_closed_after_report_and_ledger(tmp_path: Path) -> None:
    metrics_path = _write_json(
        tmp_path / "metrics.json",
        _metrics(market_breakdown=False, fold_breakdown=False, turnover=None),
    )
    ledger_path = tmp_path / "ledger.jsonl"
    audit_path = tmp_path / "audit.json"

    result = run_anti_overfit_audit(
        metrics_json=metrics_path,
        ledger_path=ledger_path,
        audit_report_json=audit_path,
    )

    ledger = _read_ledger(ledger_path)
    assert result["robustness"]["status"] == "FAIL"
    assert audit_path.exists()
    assert ledger_path.exists()
    assert "market_breakdown_unavailable" in ledger[0]["pass_fail_reason"]
    assert "fold_pass_rate_unavailable" in ledger[0]["pass_fail_reason"]
    assert "turnover_unavailable" in ledger[0]["pass_fail_reason"]


def test_source_paths_are_recorded(tmp_path: Path) -> None:
    metrics_path = _write_json(tmp_path / "metrics.json", _metrics())
    wfa_path = _write_json(tmp_path / "wfa.json", {"failure_count": 0, "failures": []})
    split_path = _write_json(tmp_path / "split.json", {"failure_count": 0, "failures": []})
    breakdown_path = _write_json(tmp_path / "breakdown.json", {"side_damage": []})
    audit_path = tmp_path / "audit.json"

    run_anti_overfit_audit(
        metrics_json=metrics_path,
        wfa_report_json=wfa_path,
        split_plan_json=split_path,
        failure_breakdown_json=[breakdown_path],
        ledger_path=tmp_path / "ledger.jsonl",
        audit_report_json=audit_path,
    )

    paths = json.loads(audit_path.read_text(encoding="utf-8"))["source_report_paths"]
    assert paths == {
        "metrics_json": metrics_path.as_posix(),
        "wfa_report_json": wfa_path.as_posix(),
        "split_plan_json": split_path.as_posix(),
        "failure_breakdown_json": [breakdown_path.as_posix()],
    }
