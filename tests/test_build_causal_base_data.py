from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_causal_base_data import (
    OUTPUT_COLUMNS,
    discover_raw_inputs,
    process_file,
    resolve_profile_inputs,
    write_reports,
)


def _write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_raw_with_datetime_index(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(df.pop("ts"), name="ts")
    df.to_parquet(path, index=True)


def test_causal_base_schema_synthetic_and_source_lineage(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    rows = [
        {
            "rtype": 33,
            "publisher_id": 1,
            "instrument_id": 100,
            "symbol": "ESH4",
            "ts_event": "2024-01-02T15:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10,
        },
        {
            "rtype": 33,
            "publisher_id": 1,
            "instrument_id": 100,
            "symbol": "ESH4",
            "ts_event": "2024-01-02T15:02:00Z",
            "open": 100.5,
            "high": 101.5,
            "low": 100.0,
            "close": 101.0,
            "volume": 12,
        },
    ]
    _write_raw(raw_path, rows)

    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    assert result.status == "WARN"
    assert result.synthetic_rows == 1
    output = pd.read_parquet(out_path)
    assert list(output.columns) == OUTPUT_COLUMNS
    assert output["ts"].is_monotonic_increasing

    synthetic = output.loc[output["is_synthetic"]].iloc[0]
    assert synthetic["raw_row_present"] == False
    assert synthetic["causal_valid"] == False
    assert pd.isna(synthetic["source_row_number"])
    assert synthetic["open"] == 100.5
    assert synthetic["high"] == 100.5
    assert synthetic["low"] == 100.5
    assert synthetic["close"] == 100.5
    assert synthetic["volume"] == 0

    raw_rows = output.loc[~output["is_synthetic"]]
    assert raw_rows["source_row_number"].tolist() == [0, 1]
    assert raw_rows["source_file_hash"].nunique() == 1
    assert raw_rows["inside_session"].all()
    assert raw_rows["causal_valid"].all()


def test_roll_boundary_sets_window_and_blocks_causal_valid(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "CL" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "CL" / "2024.parquet"
    rows = []
    for i, symbol in enumerate(["CLH4", "CLH4", "CLK4", "CLK4"]):
        rows.append(
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 10 if symbol == "CLH4" else 11,
                "symbol": symbol,
                "ts_event": pd.Timestamp("2024-01-02T15:00:00Z") + pd.Timedelta(minutes=i),
                "open": 70.0 + i,
                "high": 71.0 + i,
                "low": 69.0 + i,
                "close": 70.5 + i,
                "volume": 10 + i,
            }
        )
    _write_raw(raw_path, rows)

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_CL_ES_ZN",
        roll_window_bars=1,
    )

    assert result.status == "WARN"
    output = pd.read_parquet(out_path)
    assert output["roll_boundary_flag"].sum() == 1
    assert output["symbol_change_flag"].sum() == 1
    assert output["instrument_id_change_flag"].sum() == 1

    boundary_idx = int(output.index[output["roll_boundary_flag"]][0])
    window = output.loc[[boundary_idx - 1, boundary_idx, boundary_idx + 1]]
    assert window["roll_window_flag"].all()
    assert not window["causal_valid"].any()


def test_missing_audit_columns_warn_but_output_required_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ZN" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "volume": 10,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    assert result.status == "WARN"
    assert result.failures == []
    assert result.raw_schema_variant == "ohlcv_only"
    assert result.timestamp_source == "ts_event_column"
    assert result.metadata_available is False
    assert result.roll_detection_available is False
    assert result.roll_detection_source == "unavailable"
    assert result.roll_policy_status == "unavailable_metadata"
    assert set(result.missing_audit_cols) == {
        "rtype",
        "publisher_id",
        "instrument_id",
        "symbol",
    }
    output = pd.read_parquet(out_path)
    assert list(output.columns) == OUTPUT_COLUMNS
    assert output.loc[0, "causal_valid"]
    assert output.loc[0, "raw_schema_variant"] == "ohlcv_only"
    assert output.loc[0, "timestamp_source"] == "ts_event_column"
    assert output.loc[0, "roll_detection_available"] == False


def test_reports_are_written(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    reports_root = tmp_path / "reports" / "causal_base"
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-02T15:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            }
        ],
    )
    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    write_reports([result], reports_root, "tier_1_CL_ES_ZN")

    manifest = json.loads((reports_root / "causal_base_manifest.json").read_text())
    validation = json.loads((reports_root / "causal_base_validation.json").read_text())
    assert (reports_root / "causal_base_validation.csv").exists()
    assert manifest["stage"] == "causal_base"
    assert manifest["outputs"][0]["raw_schema_variant"] == "databento_full"
    assert manifest["outputs"][0]["timestamp_source"] == "ts_event_column"
    assert manifest["outputs"][0]["metadata_available"] is True
    assert manifest["outputs"][0]["roll_detection_available"] is True
    assert manifest["outputs"][0]["roll_detection_source"] == "instrument_id"
    assert manifest["outputs"][0]["roll_policy_status"] == "active"
    assert manifest["outputs"][0]["symbol_nonnull_count"] == 1
    assert manifest["outputs"][0]["instrument_id_nonnull_count"] == 1
    assert manifest["outputs"][0]["instrument_id_nunique"] == 1
    assert "warnings" in manifest["outputs"][0]
    assert validation["files"][0]["output_path"].endswith("2024.parquet")


def test_all_raw_discovery_uses_top_level_market_year_files_only(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    _write_raw(
        raw_root / "ES" / "2024.parquet",
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            }
        ],
    )
    _write_raw(
        raw_root / "CL" / "2023.parquet",
        [
            {
                "ts_event": "2023-01-03T15:00:00Z",
                "open": 70.0,
                "high": 71.0,
                "low": 69.0,
                "close": 70.5,
                "volume": 10,
            }
        ],
    )
    _write_raw(
        raw_root / "GC" / "LE" / "2024.parquet",
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1,
            }
        ],
    )

    discovered = discover_raw_inputs(raw_root)

    assert [(market, year) for market, year, _ in discovered] == [
        ("CL", 2023),
        ("ES", 2024),
    ]


def test_profile_resolution_supports_all_raw_and_tier_profile(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    _write_raw(
        raw_root / "ZN" / "2025.parquet",
        [
            {
                "ts_event": "2025-01-02T15:00:00Z",
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "volume": 10,
            }
        ],
    )

    all_raw = resolve_profile_inputs("all_raw", raw_root)
    tier = resolve_profile_inputs("tier_1_CL_ES_ZN", raw_root)

    assert [(market, year) for market, year, _ in all_raw] == [("ZN", 2025)]
    assert len(tier) == 9
    assert ("CL", 2023) in [(market, year) for market, year, _ in tier]


def test_metadata_with_timestamp_index_file_passes(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw_with_datetime_index(
        raw_path,
        [
            {
                "ts": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    assert result.status == "PASS"
    assert result.raw_schema_variant == "metadata_no_ts_event"
    assert result.timestamp_source == "dataframe_index"
    assert result.metadata_available is True
    assert result.roll_detection_available is True
    output = pd.read_parquet(out_path)
    assert output.loc[0, "raw_schema_variant"] == "metadata_no_ts_event"
    assert output.loc[0, "timestamp_source"] == "dataframe_index"


def test_full_databento_schema_file_passes(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "CL" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "CL" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 10,
                "symbol": "CLH4",
                "open": 70.0,
                "high": 71.0,
                "low": 69.0,
                "close": 70.5,
                "volume": 10,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    assert result.status == "PASS"
    assert result.raw_schema_variant == "databento_full"
    assert result.timestamp_source == "ts_event_column"
    assert result.roll_policy_status == "active"


def test_missing_timestamp_fails(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ZN" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "volume": 10,
            }
        ]
    ).to_parquet(raw_path, index=False)

    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    assert result.status == "FAIL"
    assert "missing timestamp source" in result.failures
    assert not out_path.exists()


def test_missing_ohlcv_column_fails(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ZN" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "volume": 10,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["close"]
    assert "missing required OHLCV columns" in result.failures
    assert not out_path.exists()


def test_symbol_change_without_instrument_id_does_not_activate_roll_window(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "symbol": "ESH4",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            },
            {
                "ts_event": "2024-01-02T15:01:00Z",
                "symbol": "ESM4",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 11,
            },
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_CL_ES_ZN")

    assert result.roll_detection_available is False
    output = pd.read_parquet(out_path)
    assert output["symbol_change_flag"].sum() == 1
    assert output["instrument_id_change_flag"].sum() == 0
    assert output["roll_boundary_flag"].sum() == 0
    assert output["roll_window_flag"].sum() == 0
    assert output["roll_detection_available"].eq(False).all()
