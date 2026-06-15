from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.ledger import append_ledger_record, build_ledger_record


def _metrics() -> dict[str, object]:
    return {
        "git_commit": "abc123",
        "costs_config": "configs/costs.yaml",
        "models_config": "configs/models.yaml",
        "failure_count": 0,
        "metrics": {
            "overall": {
                "gross_return_dollars": 300.0,
                "cost_dollars": 100.0,
                "net_return_dollars": 200.0,
                "cost_drag_to_abs_gross": 0.33,
                "slippage_cost_dollars": 60.0,
                "commission_cost_dollars": 40.0,
                "turnover_per_bar": 0.01,
                "trade_count": 12,
            },
            "summaries": [
                {"scope": "market", "market": "ES", "net_return_dollars": 100.0},
            ],
        },
    }


def test_ledger_append_writes_valid_jsonl(tmp_path: Path) -> None:
    ledger_path = tmp_path / "reports" / "experiments" / "ledger.jsonl"
    record = build_ledger_record(
        metrics=_metrics(),
        split_plan={"profile": "fixture", "markets": ["ES"], "years": [2024], "fold_count": 2},
        command="python fixture",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    append_ledger_record(ledger_path, record)

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["git_hash"] == "abc123"
    assert parsed["command"] == "python fixture"
    assert parsed["net_return_dollars"] == 200.0
    assert parsed["market_breakdown"][0]["market"] == "ES"


def test_ledger_append_only_preserves_existing_records(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    first = {"timestamp": "1", "passed": False}
    second = {"timestamp": "2", "passed": True}

    append_ledger_record(ledger_path, first)
    first_line = ledger_path.read_text(encoding="utf-8").splitlines()[0]
    append_ledger_record(ledger_path, second)

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == first_line
    assert [json.loads(line)["timestamp"] for line in lines] == ["1", "2"]
