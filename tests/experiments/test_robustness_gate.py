from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.robustness_gate import evaluate_robustness_gate


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
    return {"failure_count": 0, "failures": [], "metrics": {"overall": overall, "summaries": summaries}}


def test_robustness_gate_passes_synthetic_valid_metrics() -> None:
    result = evaluate_robustness_gate(metrics_report=_metrics())

    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert result["checks"]["cost_stress_2x"] == 100.0


def test_robustness_gate_fails_synthetic_overfit_looking_metrics() -> None:
    metrics = _metrics(gross=220.0, cost=100.0, net=120.0)
    summaries = metrics["metrics"]["summaries"]  # type: ignore[index]
    summaries[0]["net_return_dollars"] = 119.0  # type: ignore[index]
    summaries[1]["net_return_dollars"] = 1.0  # type: ignore[index]
    summaries[3]["net_return_dollars"] = -1.0  # type: ignore[index]

    result = evaluate_robustness_gate(metrics_report=metrics)

    assert result["status"] == "FAIL"
    assert "single_market_profit_contribution_above_cap" in result["failures"]
    assert "fold_pass_rate_below_minimum" in result["failures"]


def test_robustness_gate_fails_closed_on_ambiguous_cost_stress_units() -> None:
    metrics = _metrics()
    overall = metrics["metrics"]["overall"]  # type: ignore[index]
    del overall["gross_return_dollars"]  # type: ignore[index]
    del overall["cost_dollars"]  # type: ignore[index]
    overall["cost_drag_to_abs_gross"] = 0.25  # type: ignore[index]

    result = evaluate_robustness_gate(metrics_report=metrics)

    assert result["status"] == "FAIL"
    assert "cost_stress_unavailable" in result["failures"]


def test_robustness_gate_fails_closed_on_missing_market_breakdown() -> None:
    result = evaluate_robustness_gate(metrics_report=_metrics(market_breakdown=False))

    assert result["status"] == "FAIL"
    assert "market_breakdown_unavailable" in result["failures"]


def test_robustness_gate_fails_closed_on_missing_fold_evidence() -> None:
    result = evaluate_robustness_gate(metrics_report=_metrics(fold_breakdown=False))

    assert result["status"] == "FAIL"
    assert "fold_pass_rate_unavailable" in result["failures"]


def test_robustness_gate_fails_closed_on_missing_turnover() -> None:
    result = evaluate_robustness_gate(metrics_report=_metrics(turnover=None))

    assert result["status"] == "FAIL"
    assert "turnover_unavailable" in result["failures"]


def test_optional_breakdowns_are_marked_unavailable_without_invention() -> None:
    result = evaluate_robustness_gate(metrics_report=_metrics(), failure_breakdowns=[{}])

    assert result["status"] == "PASS"
    optional = result["breakdowns"]["optional"]
    assert optional == {
        "side": "unavailable",
        "hour": "unavailable",
        "session": "unavailable",
        "regime": "unavailable",
    }
