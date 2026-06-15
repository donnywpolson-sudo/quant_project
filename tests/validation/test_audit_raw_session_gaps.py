from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.validation.audit_raw_session_gaps import build_report, main


def _write_session_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "session_templates:",
                "  cme_globex_17_16_ct:",
                "    timezone: America/Chicago",
                '    regular_open: "17:00"',
                '    regular_close: "16:00"',
                "    holidays: []",
                "    closed_dates: []",
                "    early_closes: {}",
                "markets:",
                "  default:",
                "    session_template: cme_globex_17_16_ct",
                "  ZN:",
                "    session_template: cme_globex_17_16_ct",
            ]
        ),
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _args(tmp_path: Path) -> object:
    return type(
        "Args",
        (),
        {
            "markets": ["ZN"],
            "years": [2024],
            "raw_root": str(tmp_path / "data" / "raw"),
            "causal_root": str(tmp_path / "data" / "causally_gated_normalized"),
            "session_config": str(tmp_path / "configs" / "market_sessions.yaml"),
        },
    )()


def test_synthetic_timestamps_absent_from_raw_are_reported(tmp_path: Path) -> None:
    _write_session_config(tmp_path / "configs" / "market_sessions.yaml")
    _write_parquet(
        tmp_path / "data" / "raw" / "ZN" / "2024.parquet",
        [
            {"ts_event": "2024-01-02T23:00:00Z", "close": 100.0},
            {"ts_event": "2024-01-02T23:03:00Z", "close": 100.25},
        ],
    )
    _write_parquet(
        tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet",
        [
            {"ts": "2024-01-02T23:00:00Z", "is_synthetic": False},
            {
                "ts": "2024-01-02T23:01:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 1,
                "synthetic_gap_size_minutes": 3,
            },
            {
                "ts": "2024-01-02T23:02:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 1,
                "synthetic_gap_size_minutes": 3,
            },
            {"ts": "2024-01-02T23:03:00Z", "is_synthetic": False},
        ],
    )

    report = build_report(_args(tmp_path))
    entry = report["entries"][0]

    assert report["status"] == "PASS"
    assert entry["raw_gap_call"] == "confirmed_absent_from_raw_parquet"
    assert entry["synthetic_rows"] == 2
    assert entry["synthetic_timestamps_missing_from_raw"] == 2
    assert entry["synthetic_timestamps_present_in_raw"] == 0


def test_gap_bucket_counts_use_unique_gap_ids(tmp_path: Path) -> None:
    _write_session_config(tmp_path / "configs" / "market_sessions.yaml")
    _write_parquet(
        tmp_path / "data" / "raw" / "ZN" / "2024.parquet",
        [
            {"ts_event": "2024-01-02T23:00:00Z"},
            {"ts_event": "2024-01-02T23:03:00Z"},
            {"ts_event": "2024-01-03T01:00:00Z"},
            {"ts_event": "2024-01-03T01:05:00Z"},
        ],
    )
    _write_parquet(
        tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet",
        [
            {
                "ts": "2024-01-02T23:01:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 1,
                "synthetic_gap_size_minutes": 3,
            },
            {
                "ts": "2024-01-02T23:02:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 1,
                "synthetic_gap_size_minutes": 3,
            },
            {
                "ts": "2024-01-03T01:01:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 2,
                "synthetic_gap_size_minutes": 5,
            },
            {
                "ts": "2024-01-03T01:02:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 2,
                "synthetic_gap_size_minutes": 5,
            },
        ],
    )

    entry = build_report(_args(tmp_path))["entries"][0]

    assert entry["gap_size_buckets"] == [
        {"gap_size_minutes": 3, "gaps": 1},
        {"gap_size_minutes": 5, "gaps": 1},
    ]


def test_session_edge_bucket_classification(tmp_path: Path) -> None:
    _write_session_config(tmp_path / "configs" / "market_sessions.yaml")
    _write_parquet(
        tmp_path / "data" / "raw" / "ZN" / "2024.parquet",
        [{"ts_event": "2024-01-02T23:00:00Z"}, {"ts_event": "2024-01-03T03:01:00Z"}],
    )
    _write_parquet(
        tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet",
        [
            {
                "ts": "2024-01-02T23:10:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 1,
                "synthetic_gap_size_minutes": 2,
            },
            {
                "ts": "2024-01-03T00:30:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 2,
                "synthetic_gap_size_minutes": 2,
            },
            {
                "ts": "2024-01-03T03:00:00Z",
                "is_synthetic": True,
                "synthetic_gap_id": 3,
                "synthetic_gap_size_minutes": 2,
            },
        ],
    )

    entry = build_report(_args(tmp_path))["entries"][0]
    buckets = {item["bucket"]: item["rows"] for item in entry["session_buckets"]}

    assert buckets["first_60m_after_configured_open"] == 1
    assert buckets["configured_evening_17_18_ct"] == 1
    assert buckets["overnight_19_05_ct"] == 1
    assert entry["session_template_call"] == "not_primarily_session_edge_under_local_config"


def test_missing_input_fails_closed_after_writing_reports(tmp_path: Path) -> None:
    _write_session_config(tmp_path / "configs" / "market_sessions.yaml")
    json_out = tmp_path / "reports" / "gap_audit.json"
    md_out = tmp_path / "reports" / "gap_audit.md"

    code = main(
        [
            "--markets",
            "ZN",
            "--years",
            "2024",
            "--raw-root",
            str(tmp_path / "data" / "raw"),
            "--causal-root",
            str(tmp_path / "data" / "causally_gated_normalized"),
            "--session-config",
            str(tmp_path / "configs" / "market_sessions.yaml"),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ]
    )

    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert code == 1
    assert report["status"] == "FAIL"
    assert "missing raw input" in report["failures"][0]
    assert md_out.exists()
