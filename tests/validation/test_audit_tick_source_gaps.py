from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.phase1_raw_contract import REQUIRED_DATASET
from scripts.validation.audit_tick_source_gaps import build_plan, main


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _gap_audit_payload() -> dict[str, object]:
    return {
        "entries": [
            {
                "market": "ZN",
                "year": 2024,
                "largest_gaps": [
                    {
                        "synthetic_gap_id": "gap-1",
                        "gap_size_minutes": 3,
                        "synthetic_rows": 2,
                        "first_synthetic_ts": "2024-01-02T23:01:00+00:00",
                        "last_synthetic_ts": "2024-01-02T23:02:00+00:00",
                    }
                ],
            }
        ]
    }


def _args(tmp_path: Path, **overrides: object) -> object:
    values = {
        "gap_audit_json": str(tmp_path / "reports" / "raw_session_gap_audit.json"),
        "markets": ["ZN"],
        "years": [2024],
        "raw_root": str(tmp_path / "data" / "raw"),
        "schemas": ["trades"],
        "max_windows": 3,
        "max_window_minutes": 90,
        "buffer_minutes": 0,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_builds_plan_from_gap_audit_and_raw_parquet(tmp_path: Path) -> None:
    _write_json(tmp_path / "reports" / "raw_session_gap_audit.json", _gap_audit_payload())
    _write_raw(
        tmp_path / "data" / "raw" / "ZN" / "2024.parquet",
        [
            {
                "ts_event": "2024-01-02T23:00:00Z",
                "instrument_id": 123,
                "source_file": "data/dbn/ohlcv_1m/ZN/2024/source.dbn.zst",
                "source_sha256": "abc123",
            },
            {
                "ts_event": "2024-01-02T23:03:00Z",
                "instrument_id": 123,
                "source_file": "data/dbn/ohlcv_1m/ZN/2024/source.dbn.zst",
                "source_sha256": "abc123",
            },
        ],
    )

    plan = build_plan(_args(tmp_path))
    task = plan["tasks"][0]

    assert plan["status"] == "PASS"
    assert task["dataset"] == REQUIRED_DATASET
    assert task["schema"] == "trades"
    assert task["stype_in"] == "instrument_id"
    assert task["instrument_id"] == 123
    assert task["start"] == "2024-01-02T23:01:00Z"
    assert task["end"] == "2024-01-02T23:03:00Z"
    assert task["raw_ohlcv_source_file"].endswith("source.dbn.zst")
    assert task["raw_ohlcv_source_hash"] == "abc123"


def test_resolves_adjacent_instrument_id_with_buffered_window(tmp_path: Path) -> None:
    _write_json(tmp_path / "reports" / "raw_session_gap_audit.json", _gap_audit_payload())
    _write_raw(
        tmp_path / "data" / "raw" / "ZN" / "2024.parquet",
        [
            {"ts_event": "2024-01-02T22:59:00Z", "instrument_id": 456},
            {"ts_event": "2024-01-02T23:04:00Z", "instrument_id": 456},
        ],
    )

    plan = build_plan(_args(tmp_path, buffer_minutes=5))
    task = plan["tasks"][0]

    assert plan["status"] == "PASS"
    assert task["instrument_id"] == 456
    assert task["start"] == "2024-01-02T22:56:00Z"
    assert task["end"] == "2024-01-02T23:08:00Z"
    assert task["pre_buffer_minutes"] == 5
    assert task["post_buffer_minutes"] == 5


def test_rejects_oversized_window(tmp_path: Path) -> None:
    _write_json(tmp_path / "reports" / "raw_session_gap_audit.json", _gap_audit_payload())
    _write_raw(
        tmp_path / "data" / "raw" / "ZN" / "2024.parquet",
        [
            {"ts_event": "2024-01-02T23:00:00Z", "instrument_id": 123},
            {"ts_event": "2024-01-02T23:03:00Z", "instrument_id": 123},
        ],
    )

    plan = build_plan(_args(tmp_path, max_window_minutes=1))

    assert plan["status"] == "FAIL"
    assert "exceeds max" in plan["failures"][0]


def test_rejects_unsupported_schema(tmp_path: Path) -> None:
    _write_json(tmp_path / "reports" / "raw_session_gap_audit.json", _gap_audit_payload())

    plan = build_plan(_args(tmp_path, schemas=["ohlcv-1m"]))

    assert plan["status"] == "FAIL"
    assert "unsupported audit schemas" in plan["failures"][0]
    assert plan["tasks"] == []


def test_fails_closed_on_missing_raw(tmp_path: Path) -> None:
    _write_json(tmp_path / "reports" / "raw_session_gap_audit.json", _gap_audit_payload())

    plan = build_plan(_args(tmp_path))

    assert plan["status"] == "FAIL"
    assert "missing raw parquet" in plan["failures"][0]


def test_fails_closed_on_missing_gap_audit_after_writing_plan(tmp_path: Path) -> None:
    plan_out = tmp_path / "reports" / "tick_source_gap_plan.json"

    code = main(
        [
            "--gap-audit-json",
            str(tmp_path / "reports" / "missing.json"),
            "--markets",
            "ZN",
            "--years",
            "2024",
            "--raw-root",
            str(tmp_path / "data" / "raw"),
            "--plan-out",
            str(plan_out),
        ]
    )

    plan = json.loads(plan_out.read_text(encoding="utf-8"))
    assert code == 1
    assert plan["status"] == "FAIL"
    assert "missing gap audit" in plan["failures"][0]
