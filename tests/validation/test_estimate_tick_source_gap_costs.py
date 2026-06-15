from __future__ import annotations

import json
from pathlib import Path

from scripts.validation.estimate_tick_source_gap_costs import (
    build_cost_request_plan,
    estimate_costs,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tick_source_plan() -> dict[str, object]:
    return {
        "status": "PASS",
        "dry_run_only": True,
        "tasks": [
            {
                "market": "ZN",
                "year": 2024,
                "dataset": "GLBX.MDP3",
                "schema": "trades",
                "stype_in": "instrument_id",
                "instrument_id": 123,
                "start": "2024-01-02T23:01:00Z",
                "end": "2024-01-02T23:03:00Z",
                "reason": "validate_whether_ohlcv_gap_has_trade_or_book_activity",
                "source_gap_timestamps": {
                    "first_synthetic_ts": "2024-01-02T23:01:00Z",
                    "last_synthetic_ts": "2024-01-02T23:02:00Z",
                },
                "raw_ohlcv_source_file": "data/dbn/ohlcv_1m/ZN/2024/source.dbn.zst",
                "raw_ohlcv_source_hash": "abc123",
            }
        ],
    }


class FakeMetadata:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_cost(self, **kwargs: object) -> float:
        self.calls.append(kwargs)
        return 0.25

    def get_billable_size(self, **kwargs: object) -> int:
        self.calls.append(kwargs)
        return 42


class FakeClient:
    def __init__(self) -> None:
        self.metadata = FakeMetadata()


def test_builds_cost_request_plan_from_tick_source_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "reports" / "tick_source_gap_plan.json"
    _write_json(plan_path, _tick_source_plan())

    plan = build_cost_request_plan([plan_path])
    request = plan["requests"][0]

    assert plan["status"] == "PASS"
    assert plan["estimate_only"] is True
    assert plan["download_allowed"] is False
    assert plan["api_called"] is False
    assert request["dataset"] == "GLBX.MDP3"
    assert request["symbols"] == "123"
    assert request["schema"] == "trades"
    assert request["stype_in"] == "instrument_id"


def test_estimate_costs_uses_metadata_only() -> None:
    request_plan = {
        "status": "PASS",
        "failures": [],
        "requests": [
            {
                "market": "ZN",
                "year": 2024,
                "dataset": "GLBX.MDP3",
                "symbols": "123",
                "schema": "trades",
                "stype_in": "instrument_id",
                "start": "2024-01-02T23:01:00Z",
                "end": "2024-01-02T23:03:00Z",
            }
        ],
    }
    client = FakeClient()

    result = estimate_costs(request_plan, client)

    assert result["status"] == "PASS"
    assert result["api_called"] is True
    assert result["total_estimated_cost_usd"] == 0.25
    assert result["estimates"][0]["billable_size"] == 42
    assert client.metadata.calls[0]["schema"] == "trades"


def test_failed_source_plan_fails_closed(tmp_path: Path) -> None:
    plan_path = tmp_path / "reports" / "tick_source_gap_plan.json"
    _write_json(plan_path, {"status": "FAIL", "tasks": []})

    plan = build_cost_request_plan([plan_path])

    assert plan["status"] == "FAIL"
    assert any("source plan is not PASS" in failure for failure in plan["failures"])
    assert any("source plan has no tasks" in failure for failure in plan["failures"])


def test_main_requires_allow_network_for_estimate_cost(tmp_path: Path) -> None:
    plan_path = tmp_path / "reports" / "tick_source_gap_plan.json"
    out_path = tmp_path / "reports" / "cost_plan.json"
    _write_json(plan_path, _tick_source_plan())

    code = main(
        [
            "--plan-json",
            str(plan_path),
            "--estimate-out",
            str(out_path),
            "--estimate-cost",
        ]
    )

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 1
    assert result["status"] == "FAIL"
    assert "--estimate-cost requires --allow-network" in result["failures"]
