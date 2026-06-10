from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_labels import (
    LABEL_COLUMNS,
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


def test_entry_exit_alignment_uses_next_bar_open_not_close_t(tmp_path: Path) -> None:
    rows = _base_rows()
    rows[0]["close"] = 999.0
    input_path = tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet"
    output_path = tmp_path / "data" / "labeled" / "ES" / "2024.parquet"
    _write_causal(input_path, rows)

    result = process_file(
        input_path,
        output_path,
        profile="tier_1_CL_ES_ZN",
        costs_config=tmp_path / "configs" / "costs.yaml",
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
    input_df = pd.DataFrame(_base_rows())
    _write_causal(input_path, input_df.to_dict("records"))

    result = process_file(
        input_path,
        output_path,
        profile="tier_1_CL_ES_ZN",
        costs_config=tmp_path / "configs" / "costs.yaml",
    )
    write_reports([result], reports_root, "tier_1_CL_ES_ZN")

    output = pd.read_parquet(output_path)
    assert list(output.columns) == list(input_df.columns) + LABEL_COLUMNS
    manifest = json.loads((reports_root / "label_manifest.json").read_text())
    report = json.loads((reports_root / "label_report.json").read_text())
    assert manifest["stage"] == "labels"
    assert report["summary"]["target_valid_rows"] == result.target_valid_rows
    assert manifest["outputs"][0]["config"]["tick_size"] == 0.25
    assert manifest["outputs"][0]["warning_count"] == len(result.warnings)
    assert manifest["outputs"][0]["failure_count"] == len(result.failures)
    assert manifest["outputs"][0]["failures"] == result.failures
