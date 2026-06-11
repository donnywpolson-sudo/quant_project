from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase3_labels.build_labels import (
    LABEL_COLUMNS,
    LABEL_SEMANTICS_ID,
    add_labels,
    load_market_config,
    process_file,
    resolve_profile_inputs,
    write_reports,
)


def _base_rows(count: int = 40, market: str = "ES") -> list[dict[str, object]]:
    start = pd.Timestamp("2024-01-02T15:00:00Z")
    rows: list[dict[str, object]] = []
    for i in range(count):
        open_price = 100.0 + (i * 0.25)
        rows.append(
            {
                "ts": start + pd.Timedelta(minutes=i),
                "market": market,
                "year": 2024,
                "symbol": f"{market}.v.0",
                "open": open_price,
                "high": open_price + 0.25,
                "low": open_price - 0.25,
                "close": open_price,
                "volume": 10,
                "causal_valid": True,
                "session_segment_id": "session_2024-01-02_seg0",
                "is_synthetic": False,
                "valid_ohlcv": True,
                "boundary_session_flag": False,
                "roll_boundary_flag": False,
                "roll_window_flag": False,
                "roll_detection_available": True,
                "minutes_until_session_close": 120,
            }
        )
    return rows


def _write_causal(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_costs(path: Path, markets_blob: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "markets:",
                markets_blob,
            ]
        ),
        encoding="utf-8",
    )


def _write_es_costs(path: Path) -> None:
    _write_costs(
        path,
        "\n".join(
            [
                "  ES:",
                "    tick_size: 0.25",
                "    tick_value: 12.5",
                "    point_value: 50.0",
                "    min_profit_ticks: 2.0",
                "    min_stop_ticks: 4.0",
                "    round_turn_cost_ticks: 2.0",
                "    cost_source: test_costs",
                "    provisional: false",
            ]
        ),
    )


def _rows_with_gross_ticks(gross_ticks: float) -> list[dict[str, object]]:
    rows = _base_rows()
    entry_price = 100.0
    exit_price = entry_price + (gross_ticks * 0.25)
    for row in rows:
        row["open"] = entry_price
        row["high"] = entry_price + 0.25
        row["low"] = entry_price - 0.25
        row["close"] = entry_price
    rows[1]["open"] = entry_price
    rows[16]["open"] = exit_price
    rows[16]["high"] = max(exit_price, entry_price) + 0.25
    rows[16]["low"] = min(exit_price, entry_price) - 0.25
    return rows


def test_entry_exit_alignment_uses_next_bar_open_not_close_t(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[0]["close"] = 999.0
    input_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    output_path = tmp_path / "data" / "labeled" / "ES" / "2024.parquet"
    costs_path = tmp_path / "configs" / "costs.yaml"
    _write_causal(input_path, rows)
    _write_es_costs(costs_path)

    result = process_file(
        input_path,
        output_path,
        profile="tier_1_core",
        costs_config=costs_path,
    )

    assert result.failures == []
    output = pd.read_parquet(output_path)
    row = output.iloc[0]
    assert row["target_entry_ts"] == rows[1]["ts"]
    assert row["target_exit_ts"] == rows[16]["ts"]
    assert row["target_entry_price"] == rows[1]["open"]
    assert row["target_exit_price"] == rows[16]["open"]
    assert row["target_horizon_bars"] == 15
    assert row["target_entry_price"] != rows[0]["close"]


def test_profile_resolution_uses_alpha_tier_aliases(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "causally_gated_normalized"
    config_path = tmp_path / "configs" / "alpha_tiered.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "profiles:",
                "  tier_1_core_recent:",
                "    markets: [CL, ES, ZN]",
                "    years: [2023, 2024, 2025]",
                "aliases:",
                "  tier_1_core: tier_1_core_recent",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_profile_inputs("tier_1_core", input_root, config_path)

    assert resolved[0] == ("CL", 2023, input_root / "CL" / "2023.parquet")
    assert resolved[-1] == ("ZN", 2025, input_root / "ZN" / "2025.parquet")
    assert len(resolved) == 9


def test_invalid_reasons_for_current_and_future_path_flags(tmp_path: Path) -> None:
    cases = [
        ("current_causal_valid_false", 0, "causal_valid", False),
        ("session_segment_cross", 5, "session_segment_id", "session_2024-01-02_seg1"),
        ("synthetic_path", 5, "is_synthetic", True),
        ("invalid_ohlcv_path", 5, "valid_ohlcv", False),
        ("boundary_session_path", 5, "boundary_session_flag", True),
        ("roll_path", 5, "roll_window_flag", True),
    ]

    for reason, row_index, column, value in cases:
        rows = _base_rows()
        rows[row_index][column] = value
        labeled = add_labels(pd.DataFrame(rows), load_market_config("ES", tmp_path / "missing.yaml"))

        assert labeled.loc[0, "target_valid"] == False
        assert labeled.loc[0, "target_invalid_reason"] == reason


def test_tick_dollar_cost_and_deadzone_conversion() -> None:
    rows = _base_rows()
    for i, row in enumerate(rows):
        row["open"] = 100.0
        row["high"] = 100.25
        row["low"] = 99.75
        row["close"] = 100.0
    rows[1]["open"] = 100.0
    rows[16]["open"] = 101.0
    labeled = add_labels(
        pd.DataFrame(rows),
        load_market_config("ES", Path("missing.yaml")),
    )

    row = labeled.iloc[0]
    assert row["target_ret_ticks_15m"] == 4.0
    assert row["target_gross_dollars_15m"] == 50.0
    assert row["target_estimated_cost_ticks"] == 2.0
    assert row["target_estimated_cost_dollars"] == 25.0
    assert row["target_net_ticks_after_est_cost"] == 2.0
    assert row["target_net_dollars_after_est_cost"] == 25.0
    assert row["target_sign_15m"] == 1
    assert row["target_sign_with_deadzone"] == 0
    assert row["target_tradeable_after_cost"] == True


def test_explicit_cost_config_is_loaded_and_reported(tmp_path: Path) -> None:
    costs_path = tmp_path / "configs" / "costs.yaml"
    _write_costs(
        costs_path,
        "\n".join(
            [
                "  ES:",
                "    tick_size: 0.25",
                "    tick_value: 12.5",
                "    point_value: 50.0",
                "    min_profit_ticks: 2.0",
                "    min_stop_ticks: 4.0",
                "    commission_per_contract_dollars: 0.0",
                "    slippage_ticks_per_side: 1.5",
                "    round_turn_cost_ticks: 3.0",
                "    round_turn_cost_dollars: 37.5",
                "    cost_source: test_provisional_costs",
                "    provisional: true",
            ]
        ),
    )

    config = load_market_config("ES", costs_path)

    assert config.estimated_cost_ticks == 3.0
    assert config.estimated_cost_dollars == 37.5
    assert config.cost_source == "test_provisional_costs"
    assert config.provisional == True
    assert config.defaults_used == []


def test_present_cost_config_missing_market_fails_without_output(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    output_path = tmp_path / "data" / "labeled" / "ES" / "2024.parquet"
    costs_path = tmp_path / "configs" / "costs.yaml"
    _write_causal(input_path, _base_rows())
    _write_costs(
        costs_path,
        "\n".join(
            [
                "  CL:",
                "    tick_size: 0.01",
                "    tick_value: 10.0",
                "    point_value: 1000.0",
                "    min_profit_ticks: 2.0",
                "    min_stop_ticks: 4.0",
                "    round_turn_cost_ticks: 3.0",
                "    cost_source: cl_only_costs",
                "    provisional: true",
            ]
        ),
    )

    result = process_file(
        input_path,
        output_path,
        profile="tier_1_core",
        costs_config=costs_path,
    )

    assert result.status == "FAIL"
    assert "market_cost_missing" in result.config["defaults_used"]
    assert any("market config defaults used" in warning for warning in result.warnings)
    assert "placeholder costs used" in result.warnings
    assert any("placeholder/default costs unavailable" in failure for failure in result.failures)
    assert not output_path.exists()


def test_net_ticks_after_cost_semantics() -> None:
    cases = [
        (4.0, 2.0, True),
        (1.0, 0.0, False),
        (-4.0, -2.0, True),
        (-1.0, -0.0, False),
        (0.0, 0.0, False),
        (2.0, 0.0, False),
        (-2.0, -0.0, False),
    ]

    for gross_ticks, expected_net_ticks, expected_tradeable in cases:
        labeled = add_labels(
            pd.DataFrame(_rows_with_gross_ticks(gross_ticks)),
            load_market_config("ES", Path("missing.yaml")),
        )
        row = labeled.iloc[0]
        gross = row["target_ret_ticks_15m"]
        net = row["target_net_ticks_after_est_cost"]

        assert gross == gross_ticks
        assert net == expected_net_ticks
        assert row["target_net_dollars_after_est_cost"] == expected_net_ticks * 12.5
        assert row["target_tradeable_after_cost"] == expected_tradeable
        assert abs(net) <= abs(gross)
        assert net == 0 or gross * net > 0


def test_adaptive_atr_threshold_uses_past_only_data() -> None:
    rows = _base_rows()
    for row in rows:
        row["open"] = 100.0
        row["high"] = 100.25
        row["low"] = 100.0
        row["close"] = 100.0
    rows[1]["open"] = 100.0
    rows[2]["high"] = 100.75
    rows[20]["high"] = 150.0
    rows[20]["low"] = 50.0
    labeled = add_labels(
        pd.DataFrame(rows),
        load_market_config("ES", Path("missing.yaml")),
    )

    assert labeled.loc[0, "fade_long_success_15m"] == True


def test_fade_and_30m_regime_labels() -> None:
    rows = _base_rows(count=45)
    for row in rows:
        row["open"] = 100.0
        row["high"] = 100.25
        row["low"] = 99.75
        row["close"] = 100.0
    rows[1]["open"] = 100.0
    rows[2]["high"] = 101.0
    rows[3]["high"] = 100.75
    rows[4]["low"] = 99.25
    rows[25]["high"] = 102.0
    rows[26]["low"] = 98.0
    rows[28]["low"] = 99.0
    labeled = add_labels(
        pd.DataFrame(rows),
        load_market_config("ES", Path("missing.yaml")),
    )

    row = labeled.iloc[0]
    assert row["mfe_ticks_15m"] == 4.0
    assert row["mae_ticks_15m"] == -3.0
    assert row["fade_long_success_15m"] == True
    assert row["fade_short_success_15m"] == False
    assert row["trend_danger_up_30m"] == True
    assert row["trend_danger_down_30m"] == True
    assert row["revert_to_vwap_30m"] == True
    assert row["revert_to_session_mid_30m"] == True


def test_output_schema_and_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    output_path = tmp_path / "data" / "labeled" / "ES" / "2024.parquet"
    reports_root = tmp_path / "reports" / "labels"
    costs_path = tmp_path / "configs" / "costs.yaml"
    input_df = pd.DataFrame(_base_rows())
    _write_causal(input_path, input_df.to_dict("records"))
    _write_es_costs(costs_path)

    result = process_file(
        input_path,
        output_path,
        profile="tier_1_core",
        costs_config=costs_path,
    )
    write_reports([result], reports_root, "tier_1_core")

    output = pd.read_parquet(output_path)
    assert list(output.columns) == list(input_df.columns) + LABEL_COLUMNS
    assert output["label_semantics"].eq(LABEL_SEMANTICS_ID).all()
    assert output["cost_source"].eq("test_costs").all()
    assert output["cost_provisional"].eq(False).all()
    manifest = json.loads((reports_root / "label_manifest.json").read_text())
    report = json.loads((reports_root / "label_report.json").read_text())
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
    assert provenance_keys <= set(report)
    assert isinstance(manifest["output_file_hashes"][result.output_path], str)
    assert len(manifest["output_file_hashes"][result.output_path]) == 64
    assert manifest["input_file_hashes"][result.input_path] is not None
    assert manifest["profile"] == "tier_1_core"
    assert manifest["markets"] == ["ES"]
    assert manifest["years"] == [2024]
    assert manifest["warning_count"] == len(result.warnings)
    assert manifest["failure_count"] == len(result.failures)
    assert manifest["failures"] == []
    assert manifest["stage"] == "labels"
    assert report["summary"]["target_valid_rows"] == result.target_valid_rows
    assert manifest["outputs"][0]["config"]["tick_size"] == 0.25
    assert manifest["outputs"][0]["warning_count"] == len(result.warnings)
    assert manifest["outputs"][0]["failure_count"] == len(result.failures)
    assert manifest["outputs"][0]["failures"] == result.failures
    assert (
        manifest["label_semantics"]["target_tradeable_after_cost"]
        == "absolute move exceeds estimated cost; not guaranteed profitability"
    )


def test_mixed_roll_detection_availability_is_reported(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    output_path = tmp_path / "data" / "labeled" / "ES" / "2024.parquet"
    reports_root = tmp_path / "reports" / "labels"
    costs_path = tmp_path / "configs" / "costs.yaml"
    rows = _base_rows()
    rows[3]["roll_detection_available"] = False
    rows[4]["roll_detection_available"] = False
    _write_causal(input_path, rows)
    _write_es_costs(costs_path)

    result = process_file(
        input_path,
        output_path,
        profile="tier_1_core",
        costs_config=costs_path,
    )
    write_reports([result], reports_root, "tier_1_core")

    manifest = json.loads((reports_root / "label_manifest.json").read_text())
    report = json.loads((reports_root / "label_report.json").read_text())
    output_row = manifest["outputs"][0]

    assert result.roll_detection_available == False
    assert result.roll_detection_available_rows == len(rows) - 2
    assert result.roll_detection_unavailable_rows == 2
    assert result.roll_protection_unavailable == True
    assert "roll protection unavailable for 2 rows" in result.warnings[-1]
    assert result.status == "FAIL"
    assert "roll protection unavailable for 2 rows" in result.failures[-1]
    assert not output_path.exists()
    assert output_row["roll_detection_available"] == False
    assert output_row["roll_detection_available_rows"] == len(rows) - 2
    assert output_row["roll_detection_unavailable_rows"] == 2
    assert report["summary"]["roll_detection_unavailable_rows"] == 2
