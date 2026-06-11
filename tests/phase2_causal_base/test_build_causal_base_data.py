from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase2_causal_base.build_causal_base_data import (
    OUTPUT_COLUMNS,
    discover_raw_inputs,
    load_causal_base_config,
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


def _write_profile_config(path: Path, *, synthetic_pct: float = 2.0, degraded_pct: float = 1.0, roll_pct: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "defaults:",
                "  years: [2024]",
                "  max_synthetic_gap_minutes: 120",
                f"  max_synthetic_rows_pct: {synthetic_pct}",
                f"  max_degraded_rows_pct: {degraded_pct}",
                f"  max_roll_window_rows_pct: {roll_pct}",
                "  require_roll_metadata_for_profiles: [tier_1_core, tier_2_universe_recent, tier_2_universe_long]",
                "profiles:",
                "  tier_0_smoke:",
                "    markets: [ES]",
                "    years: [2024]",
                "  metadata_optional_test:",
                "    markets: [ES]",
                "    years: [2024]",
                "  tier_1_core:",
                "    markets: [CL, ES, ZN]",
                "    years: [2024]",
                "aliases:",
                "  tier_1: tier_1_core",
            ]
        ),
        encoding="utf-8",
    )


def _write_profile_defaults_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "defaults:",
                "  years: [2024]",
                "  max_synthetic_gap_minutes: 77",
                "  require_roll_metadata_for_profiles: [tier_1_core_recent, tier_2_forward_2026]",
                "profile_defaults:",
                "  smoke:",
                "    max_synthetic_rows_pct: 5.0",
                "    max_degraded_rows_pct: 5.0",
                "    max_roll_window_rows_pct: 2.0",
                "  recent_research:",
                "    max_synthetic_rows_pct: 2.0",
                "    max_degraded_rows_pct: 1.0",
                "    max_roll_window_rows_pct: 1.0",
                "  production_like:",
                "    max_synthetic_rows_pct: 1.0",
                "    max_degraded_rows_pct: 0.5",
                "    max_roll_window_rows_pct: 1.0",
                "profiles:",
                "  tier_0_smoke:",
                "    settings_profile: smoke",
                "    markets: [ES]",
                "    years: [2024]",
                "  tier_1_core_recent:",
                "    settings_profile: recent_research",
                "    markets: [CL, ES, ZN]",
                "    years: [2024]",
                "  tier_2_forward_2026:",
                "    settings_profile: production_like",
                "    markets: [ES]",
                "    years: [2026]",
                "aliases:",
                "  tier_1: tier_1_core_recent",
                "  tier_2_forward: tier_2_forward_2026",
            ]
        ),
        encoding="utf-8",
    )


def _write_session_config(
    path: Path,
    *,
    early_closes: dict[str, str] | None = None,
    closed_dates: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    early_closes = early_closes or {}
    closed_dates = closed_dates or []
    early_lines = [f'      {day}: "{time}"' for day, time in early_closes.items()]
    closed_text = ", ".join(closed_dates)
    path.write_text(
        "\n".join(
            [
                "session_templates:",
                "  cme_globex_17_16_ct:",
                "    timezone: America/Chicago",
                '    regular_open: "17:00"',
                '    regular_close: "16:00"',
                "    holidays: []",
                f"    closed_dates: [{closed_text}]",
                "    early_closes:",
                *(early_lines or []),
                "markets:",
                "  default:",
                "    session_template: cme_globex_17_16_ct",
                "  CL:",
                "    session_template: cme_globex_17_16_ct",
                "  ES:",
                "    session_template: cme_globex_17_16_ct",
                "  ZN:",
                "    session_template: cme_globex_17_16_ct",
            ]
        ),
        encoding="utf-8",
    )


def test_causal_base_config_uses_smoke_profile_thresholds(tmp_path: Path) -> None:
    profile_config = tmp_path / "configs" / "alpha_tiered.yaml"
    _write_profile_defaults_config(profile_config)

    config = load_causal_base_config(profile_config, "tier_0_smoke")

    assert config.max_synthetic_rows_pct == 5.0
    assert config.max_degraded_rows_pct == 5.0
    assert config.max_roll_window_rows_pct == 2.0
    assert config.max_synthetic_gap_minutes == 77


def test_causal_base_config_resolves_alias_before_threshold_lookup(tmp_path: Path) -> None:
    profile_config = tmp_path / "configs" / "alpha_tiered.yaml"
    _write_profile_defaults_config(profile_config)

    config = load_causal_base_config(profile_config, "tier_1")
    direct_config = load_causal_base_config(profile_config, "tier_1_core_recent")

    assert config.max_synthetic_rows_pct == 2.0
    assert config.max_degraded_rows_pct == 1.0
    assert config.max_roll_window_rows_pct == 1.0
    assert direct_config == config


def test_causal_base_config_uses_forward_production_like_thresholds(tmp_path: Path) -> None:
    profile_config = tmp_path / "configs" / "alpha_tiered.yaml"
    _write_profile_defaults_config(profile_config)

    config = load_causal_base_config(profile_config, "tier_2_forward")

    assert config.max_synthetic_rows_pct == 1.0
    assert config.max_degraded_rows_pct == 0.5
    assert config.max_roll_window_rows_pct == 1.0


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

    result = process_file(raw_path, out_path, profile="metadata_optional_test")

    assert result.status == "WARN"
    assert result.synthetic_rows == 1
    output = pd.read_parquet(out_path)
    assert list(output.columns) == OUTPUT_COLUMNS
    assert output["ts"].is_monotonic_increasing
    assert {
        "causal_invalid_reason",
        "session_calendar_status",
        "holiday_calendar_available",
        "early_close_calendar_available",
        "calendar_coverage_status",
    }.issubset(output.columns)

    synthetic = output.loc[output["is_synthetic"]].iloc[0]
    assert synthetic["raw_row_present"] == False
    assert synthetic["causal_valid"] == False
    assert "synthetic" in synthetic["causal_invalid_reason"]
    assert synthetic["boundary_session_flag"] == True
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
    assert raw_rows["boundary_session_flag"].all()
    assert not raw_rows["causal_valid"].any()
    assert raw_rows["causal_invalid_reason"].str.contains("boundary_session").all()
    assert output["calendar_coverage_status"].eq("regular_session_only").all()


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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        )
    _write_raw(raw_path, rows)

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        roll_window_bars=1,
    )

    assert result.status == "WARN"
    assert result.failures == []
    output = pd.read_parquet(out_path)
    assert output["roll_boundary_flag"].sum() == 1
    assert output["symbol_change_flag"].sum() == 1
    assert output["instrument_id_change_flag"].sum() == 1

    boundary_idx = int(output.index[output["roll_boundary_flag"]][0])
    window = output.loc[[boundary_idx - 1, boundary_idx, boundary_idx + 1]]
    assert window["roll_window_flag"].all()
    assert not window["causal_valid"].any()
    assert window["causal_invalid_reason"].str.contains("roll_window").all()


def test_roll_exclusion_is_not_warn_under_threshold(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "CL" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "CL" / "2024.parquet"
    profile_config = tmp_path / "configs" / "alpha_tiered.yaml"
    _write_profile_config(profile_config, roll_pct=100.0)
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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        )
    _write_raw(raw_path, rows)

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        roll_window_bars=1,
        profile_config_path=profile_config,
    )

    output = pd.read_parquet(out_path)
    assert output.loc[output["roll_window_flag"], "causal_valid"].eq(False).all()
    assert result.roll_window_threshold_breached is False
    assert not any("roll exclusion threshold breached" in item for item in result.warnings)


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

    result = process_file(raw_path, out_path, profile="metadata_optional_test")

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
    assert output.loc[0, "boundary_session_flag"]
    assert output.loc[0, "causal_valid"] == False
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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )
    result = process_file(raw_path, out_path, profile="tier_0_smoke")

    write_reports([result], reports_root, "metadata_optional_test")

    manifest = json.loads((reports_root / "causal_base_manifest.json").read_text())
    validation = json.loads((reports_root / "causal_base_validation.json").read_text())
    assert (reports_root / "causal_base_validation.csv").exists()
    provenance_keys = {
        "generated_at",
        "git_commit",
        "script_path",
        "script_hash",
        "config_hash",
        "input_file_hashes",
        "output_file_hashes",
        "profile",
        "markets",
        "years",
        "warning_count",
        "failure_count",
        "failures",
    }
    assert provenance_keys <= set(manifest)
    assert provenance_keys <= set(validation)
    assert manifest["input_file_hashes"][result.input_path] == result.source_file_hash
    output_hash = manifest["output_file_hashes"][result.output_path]
    assert isinstance(output_hash, str)
    assert len(output_hash) == 64
    assert manifest["warning_count"] == result.to_dict()["warning_count"]
    assert manifest["failure_count"] == 0
    assert manifest["failures"] == []
    assert manifest["markets"] == ["ES"]
    assert manifest["years"] == [2024]
    assert manifest["stage"] == "causal_base"
    assert manifest["outputs"][0]["raw_schema_variant"] == "databento_full"
    assert manifest["outputs"][0]["timestamp_source"] == "ts_event_column"
    assert manifest["outputs"][0]["metadata_available"] is True
    assert manifest["outputs"][0]["roll_detection_available"] is True
    assert manifest["outputs"][0]["roll_detection_source"] == "instrument_id"
    assert manifest["outputs"][0]["roll_policy_status"] == "active"
    assert manifest["outputs"][0]["raw_schema_policy"] == "strict"
    assert manifest["outputs"][0]["required_raw_schema_cols"] == [
        "ts_event",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rtype",
        "publisher_id",
        "instrument_id",
        "symbol",
        "data_quality_status",
        "data_quality_degraded",
    ]
    assert manifest["outputs"][0]["raw_schema_missing_cols"] == []
    assert manifest["outputs"][0]["missing_required_raw_cols"] == []
    assert manifest["outputs"][0]["symbol_nonnull_count"] == 1
    assert manifest["outputs"][0]["instrument_id_nonnull_count"] == 1
    assert manifest["outputs"][0]["instrument_id_nunique"] == 1
    assert manifest["outputs"][0]["warning_count"] == result.to_dict()["warning_count"]
    assert manifest["outputs"][0]["failure_count"] == 0
    assert manifest["outputs"][0]["failures"] == []
    assert manifest["outputs"][0]["boundary_session_rows"] == 1
    assert manifest["outputs"][0]["causal_valid_rows"] == 0
    assert manifest["outputs"][0]["causal_invalid_rows"] == 1
    assert manifest["outputs"][0]["session_calendar_status"] == "config_backed_regular_session"
    assert manifest["outputs"][0]["holiday_calendar_available"] is False
    assert manifest["outputs"][0]["early_close_calendar_available"] is False
    assert manifest["outputs"][0]["calendar_coverage_status"] == "regular_session_only"
    assert "warnings" in manifest["outputs"][0]
    assert "holiday calendar unavailable: using hardcoded regular session" not in manifest["outputs"][0]["warnings"]
    assert "early-close calendar unavailable: using hardcoded regular session" not in manifest["outputs"][0]["warnings"]
    validation_file = validation["files"][0]
    assert validation_file["synthetic_gap_count"] == 0
    assert validation_file["synthetic_rows_pct"] == 0.0
    assert validation_file["synthetic_gap_threshold_breached"] is False
    assert validation_file["roll_window_rows_pct"] == 0.0
    assert validation_file["roll_window_threshold_breached"] is False
    assert validation_file["degraded_rows_pct"] == 0.0
    assert validation_file["degraded_threshold_breached"] is False
    assert validation_file["raw_schema_policy"] == "strict"
    assert validation_file["raw_schema_missing_cols"] == []
    assert validation_file["calendar_coverage_status"] == "regular_session_only"
    assert validation["summary"]["synthetic_gap_threshold_breached_files"] == 0
    assert validation["summary"]["roll_window_threshold_breached_files"] == 0
    assert validation["summary"]["degraded_threshold_breached_files"] == 0
    assert validation["files"][0]["output_path"].endswith("2024.parquet")


def test_calendar_config_removes_hardcoded_calendar_warning(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    session_config = tmp_path / "configs" / "market_sessions.yaml"
    _write_session_config(session_config)
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-03T15:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        session_config_path=session_config,
    )

    assert result.session_calendar_status == "config_backed_regular_session"
    assert result.calendar_coverage_status == "regular_session_only"
    assert result.holiday_calendar_available is False
    assert result.early_close_calendar_available is False
    assert "hardcoded session calendar used" not in result.warnings
    assert (
        "holiday/early-close calendar coverage unavailable: regular session only"
        in result.warnings
    )


def test_early_close_changes_minutes_until_session_close(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    session_config = tmp_path / "configs" / "market_sessions.yaml"
    _write_session_config(session_config, early_closes={"2024-01-03": "12:00"})
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-03T17:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        session_config_path=session_config,
    )

    assert result.session_calendar_status == "config_backed"
    assert result.calendar_coverage_status == "config_backed"
    assert result.holiday_calendar_available is False
    assert result.early_close_calendar_available is True
    output = pd.read_parquet(out_path)
    assert output.loc[0, "inside_session"] == True
    assert output.loc[0, "minutes_until_session_close"] == 60.0


def test_closed_date_is_excluded(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    session_config = tmp_path / "configs" / "market_sessions.yaml"
    _write_session_config(session_config, closed_dates=["2024-01-03"])
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-03T15:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        session_config_path=session_config,
    )

    assert result.session_calendar_status == "config_backed"
    assert result.calendar_coverage_status == "config_backed"
    assert result.holiday_calendar_available is True
    assert result.early_close_calendar_available is False
    output = pd.read_parquet(out_path)
    assert output.loc[0, "inside_session"] == False
    assert output.loc[0, "causal_valid"] == False


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
                "data_quality_status": "available",
                "data_quality_degraded": False,
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
                "data_quality_status": "available",
                "data_quality_degraded": False,
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
    tier = resolve_profile_inputs("tier_1_core", raw_root)

    assert [(market, year) for market, year, _ in all_raw] == [("ZN", 2025)]
    assert len(tier) == 9
    assert ("CL", 2023) in [(market, year) for market, year, _ in tier]


def test_boundary_sessions_are_not_causal_valid(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    rows = []
    for day in [2, 3, 4]:
        rows.append(
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": f"2024-01-{day:02d}T15:00:00Z",
                "open": 100.0 + day,
                "high": 101.0 + day,
                "low": 99.0 + day,
                "close": 100.5 + day,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        )
    _write_raw(raw_path, rows)

    result = process_file(raw_path, out_path, profile="metadata_optional_test")

    assert result.boundary_session_rows == 2
    assert result.causal_valid_rows == 1
    output = pd.read_parquet(out_path)
    assert output["boundary_session_flag"].tolist() == [True, False, True]
    assert output["causal_valid"].tolist() == [False, True, False]
    assert output.loc[output["causal_valid"], "causal_invalid_reason"].eq("").all()
    assert output.loc[output["boundary_session_flag"], "causal_invalid_reason"].str.contains(
        "boundary_session"
    ).all()


def test_year_boundary_bleed_prevents_boundary_flag_when_adjacent_data_exists(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "data" / "raw" / "ES"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    common = {
        "rtype": 33,
        "publisher_id": 1,
        "instrument_id": 100,
        "symbol": "ESH4",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 10,
        "data_quality_status": "available",
        "data_quality_degraded": False,
    }
    _write_raw(raw_root / "2023.parquet", [{**common, "ts_event": "2023-12-31T23:30:00Z"}])
    _write_raw(
        raw_root / "2024.parquet",
        [
            {**common, "ts_event": "2024-01-01T06:30:00Z"},
            {**common, "ts_event": "2024-12-31T23:30:00Z"},
        ],
    )
    _write_raw(raw_root / "2025.parquet", [{**common, "ts_event": "2025-01-01T06:30:00Z"}])

    process_file(raw_root / "2024.parquet", out_path, profile="tier_1_core")

    output = pd.read_parquet(out_path)
    assert output["boundary_session_flag"].tolist() == [False, False]


def test_boundary_session_flag_remains_true_when_adjacent_data_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-01T06:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    process_file(raw_path, out_path, profile="tier_1_core")

    output = pd.read_parquet(out_path)
    assert output.loc[0, "boundary_session_flag"] == True
    assert output.loc[0, "causal_valid"] == False


def test_causal_valid_formula_includes_boundary_session_flag(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "CL" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "CL" / "2024.parquet"
    rows = []
    for day in [2, 3, 4]:
        rows.append(
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 10,
                "symbol": "CLH4",
                "ts_event": f"2024-01-{day:02d}T15:00:00Z",
                "open": 70.0 + day,
                "high": 71.0 + day,
                "low": 69.0 + day,
                "close": 70.5 + day,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        )
    _write_raw(raw_path, rows)

    process_file(raw_path, out_path, profile="tier_1_core")

    output = pd.read_parquet(out_path)
    expected = (
        output["raw_row_present"]
        & ~output["is_synthetic"]
        & output["valid_ohlcv"]
        & output["inside_session"]
        & output["trainable_data_quality"]
        & ~output["roll_window_flag"]
        & ~output["boundary_session_flag"]
    )
    assert output["causal_valid"].equals(expected)
    assert output.loc[0, "boundary_session_flag"]
    assert output.loc[0, "causal_valid"] == False


def test_missing_raw_file_manifest_reports_failure_count_and_failures(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ZN" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet"
    reports_root = tmp_path / "reports" / "causal_base"

    result = process_file(raw_path, out_path, profile="tier_0_smoke")
    write_reports([result], reports_root, "tier_1_core")

    manifest = json.loads((reports_root / "causal_base_manifest.json").read_text())
    item = manifest["outputs"][0]
    assert item["status"] == "FAIL"
    assert item["failure_count"] == 1
    assert item["failures"] == ["input file missing"]


def test_synthetic_gap_at_max_limit_is_filled(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-02T15:03:00Z",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 12,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
        ],
    )

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        max_synthetic_gap_minutes=3,
    )

    assert result.synthetic_rows == 2
    output = pd.read_parquet(out_path)
    synthetic_ts = output.loc[output["is_synthetic"], "ts"].tolist()
    assert synthetic_ts == [
        pd.Timestamp("2024-01-02T15:01:00Z"),
        pd.Timestamp("2024-01-02T15:02:00Z"),
    ]
    assert output.loc[output["is_synthetic"], "causal_valid"].eq(False).all()
    assert output.loc[output["is_synthetic"], "synthetic_gap_id"].notna().all()
    assert output.loc[output["is_synthetic"], "synthetic_gap_size_minutes"].eq(3).all()
    assert output.loc[output["is_synthetic"], "synthetic_gap_reason"].eq(
        "missing_in_session_minute"
    ).all()


def test_synthetic_gap_above_max_limit_is_not_filled(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-02T15:04:00Z",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 12,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
        ],
    )

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        max_synthetic_gap_minutes=3,
    )

    assert result.synthetic_rows == 0
    output = pd.read_parquet(out_path)
    assert not output["is_synthetic"].any()


def test_no_synthetic_fill_across_instrument_change_when_metadata_available(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "data" / "raw" / "CL" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "CL" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 10,
                "symbol": "CLH4",
                "ts_event": "2024-01-02T15:00:00Z",
                "open": 70.0,
                "high": 71.0,
                "low": 69.0,
                "close": 70.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 11,
                "symbol": "CLK4",
                "ts_event": "2024-01-02T15:03:00Z",
                "open": 70.5,
                "high": 71.5,
                "low": 70.0,
                "close": 71.0,
                "volume": 12,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
        ],
    )

    result = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        max_synthetic_gap_minutes=3,
    )

    assert result.synthetic_rows == 0
    output = pd.read_parquet(out_path)
    assert not output["is_synthetic"].any()
    assert output["instrument_id_change_flag"].sum() == 1
    assert output["roll_boundary_flag"].sum() == 1


def test_synthetic_warning_only_triggers_above_threshold(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    high_threshold_config = tmp_path / "configs" / "alpha_tiered.yaml"
    _write_profile_config(high_threshold_config, synthetic_pct=90.0)
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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-02T15:03:00Z",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 12,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
        ],
    )

    high = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        max_synthetic_gap_minutes=3,
        profile_config_path=high_threshold_config,
    )
    low = process_file(
        raw_path,
        tmp_path / "data" / "second" / "ES" / "2024.parquet",
        profile="tier_1_core",
        max_synthetic_gap_minutes=3,
    )

    assert high.synthetic_gap_threshold_breached is False
    assert not any("synthetic threshold breached" in item for item in high.warnings)
    assert low.synthetic_gap_threshold_breached is True
    assert any("synthetic threshold breached" in item for item in low.warnings)


def test_no_synthetic_fill_across_session_boundary(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-02T21:59:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "ts_event": "2024-01-02T23:00:00Z",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 12,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.synthetic_rows == 0
    output = pd.read_parquet(out_path)
    assert output["session_id"].nunique() == 2
    assert not output["is_synthetic"].any()


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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="metadata_optional_test")

    assert result.status == "WARN"
    assert result.raw_schema_variant == "metadata_no_ts_event"
    assert result.raw_schema_policy == "relaxed"
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
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "WARN"
    assert result.raw_schema_variant == "databento_full"
    assert result.raw_schema_policy == "strict"
    assert result.timestamp_source == "ts_event_column"
    assert result.roll_policy_status == "active"


def test_missing_timestamp_fails(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ZN" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ZNH4",
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ]
    ).to_parquet(raw_path, index=False)

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["ts_event"]
    assert result.raw_schema_missing_cols == ["ts_event"]
    assert "missing required raw schema columns: ts_event" in result.failures
    assert not out_path.exists()


def test_missing_ohlcv_column_fails(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ZN" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ZN" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ZNH4",
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["close"]
    assert result.raw_schema_missing_cols == ["close"]
    assert "missing required raw schema columns: close" in result.failures
    assert not out_path.exists()


def test_production_profile_fails_if_data_quality_status_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["data_quality_status"]
    assert result.raw_schema_missing_cols == ["data_quality_status"]
    assert "missing required raw schema columns: data_quality_status" in result.failures
    assert not out_path.exists()


def test_production_profile_fails_if_data_quality_degraded_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["data_quality_degraded"]
    assert result.raw_schema_missing_cols == ["data_quality_degraded"]
    assert "missing required raw schema columns: data_quality_degraded" in result.failures
    assert not out_path.exists()


def test_strict_profile_fails_if_required_metadata_values_are_null(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": None,
                "publisher_id": None,
                "instrument_id": None,
                "symbol": None,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == [
        "rtype",
        "publisher_id",
        "instrument_id",
        "symbol",
    ]
    assert (
        "null or blank required raw schema columns: rtype, publisher_id, instrument_id, symbol"
        in result.failures
    )
    assert not out_path.exists()


def test_strict_profile_fails_if_symbol_is_blank(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": " ",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["symbol"]
    assert "null or blank required raw schema columns: symbol" in result.failures
    assert not out_path.exists()


def test_strict_profile_fails_if_data_quality_status_is_null(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": None,
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["data_quality_status"]
    assert "null or blank required raw schema columns: data_quality_status" in result.failures
    assert not out_path.exists()


def test_strict_profile_fails_if_data_quality_degraded_is_null(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ESH4",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": None,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["data_quality_degraded"]
    assert "null or blank required raw schema columns: data_quality_degraded" in result.failures
    assert not out_path.exists()


def test_metadata_optional_test_remains_relaxed_for_null_optional_fields(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": None,
                "publisher_id": None,
                "instrument_id": None,
                "symbol": " ",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": None,
                "data_quality_degraded": None,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="metadata_optional_test")

    assert result.status == "WARN"
    assert result.raw_schema_policy == "relaxed"
    assert result.failures == []
    assert out_path.exists()
    output = pd.read_parquet(out_path)
    assert output.loc[0, "data_quality_status"] == "unknown"
    assert output.loc[0, "data_quality_degraded"] == False


def test_symbol_change_without_instrument_id_does_not_activate_roll_window(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    reports_root = tmp_path / "reports" / "causal_base"
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
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

    result = process_file(raw_path, out_path, profile="metadata_optional_test")

    assert result.status == "WARN"
    assert result.roll_detection_available is False
    output = pd.read_parquet(out_path)
    assert output["symbol_change_flag"].sum() == 1
    assert output["instrument_id_change_flag"].sum() == 0
    assert output["roll_boundary_flag"].sum() == 0
    assert output["roll_window_flag"].sum() == 0
    assert output["roll_detection_available"].eq(False).all()

    write_reports([result], reports_root, "metadata_optional_test")
    manifest = json.loads((reports_root / "causal_base_manifest.json").read_text())
    item = manifest["outputs"][0]
    assert item["roll_detection_available"] is False
    assert item["roll_detection_source"] == "unavailable"
    assert item["roll_policy_status"] == "unavailable_metadata"
    assert "roll detection unavailable: missing populated instrument_id" in item["warnings"]


def test_production_profile_fails_if_roll_metadata_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    alias_out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024_alias.parquet"
    _write_raw(
        raw_path,
        [
            {
                "ts_event": "2024-01-02T15:00:00Z",
                "rtype": 33,
                "publisher_id": 1,
                "symbol": "ESH4",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            }
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")
    alias_result = process_file(raw_path, alias_out_path, profile="tier_1_core")

    assert result.status == "FAIL"
    assert result.missing_required_raw_cols == ["instrument_id"]
    assert result.raw_schema_missing_cols == ["instrument_id"]
    assert "missing required raw schema columns: instrument_id" in result.failures
    assert not out_path.exists()
    assert alias_result.status == "FAIL"
    assert alias_result.missing_required_raw_cols == ["instrument_id"]
    assert "missing required raw schema columns: instrument_id" in alias_result.failures
    assert not alias_out_path.exists()


def test_degraded_data_quality_blocks_whole_session_from_causal_valid(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    _write_raw(
        raw_path,
        [
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ES.v.0",
                "ts_event": "2024-01-02T15:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
                "data_quality_status": "available",
                "data_quality_degraded": False,
            },
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ES.v.0",
                "ts_event": "2024-01-02T15:01:00Z",
                "open": 100.5,
                "high": 101.5,
                "low": 100.0,
                "close": 101.0,
                "volume": 12,
                "data_quality_status": "degraded",
                "data_quality_degraded": True,
            },
        ],
    )

    result = process_file(raw_path, out_path, profile="tier_1_core")

    output = pd.read_parquet(out_path)
    raw_rows = output[output["raw_row_present"]]
    assert result.degraded_bar_rows == 1
    assert result.degraded_session_rows == 1
    assert raw_rows["session_data_quality_degraded"].all()
    assert not raw_rows["trainable_data_quality"].any()
    assert not raw_rows["causal_valid"].any()
    assert raw_rows["causal_invalid_reason"].str.contains("degraded_session").all()


def test_degraded_warning_only_triggers_above_threshold(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "ES" / "2024.parquet"
    out_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    high_threshold_config = tmp_path / "configs" / "alpha_tiered.yaml"
    _write_profile_config(high_threshold_config, degraded_pct=100.0)
    rows = []
    for i, degraded in enumerate([False, True, False]):
        rows.append(
            {
                "rtype": 33,
                "publisher_id": 1,
                "instrument_id": 100,
                "symbol": "ES.v.0",
                "ts_event": pd.Timestamp("2024-01-02T15:00:00Z") + pd.Timedelta(minutes=i),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 10,
                "data_quality_status": "degraded" if degraded else "available",
                "data_quality_degraded": degraded,
            }
        )
    _write_raw(raw_path, rows)

    high = process_file(
        raw_path,
        out_path,
        profile="tier_1_core",
        profile_config_path=high_threshold_config,
    )
    low = process_file(
        raw_path,
        tmp_path / "data" / "second" / "ES" / "2024.parquet",
        profile="tier_1_core",
    )

    assert high.degraded_threshold_breached is False
    assert not any("degraded threshold breached" in item for item in high.warnings)
    assert low.degraded_threshold_breached is True
    assert any("degraded threshold breached" in item for item in low.warnings)
