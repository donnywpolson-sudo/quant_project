from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase8_model_selection.audit_policy_signal_alignment import (  # noqa: E402
    OUTPUT_SUFFIXES,
    build_policy_signal_alignment,
    main,
)
from scripts.phase8_model_selection.evaluate_predictions import PolicyConfig  # noqa: E402


def _write_costs(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
markets:
  ES:
    point_value: 50.0
    tick_value: 10.0
    round_turn_cost_dollars: 10.0
    slippage_ticks_per_side: 0.25
  CL:
    point_value: 100.0
    tick_value: 10.0
    round_turn_cost_dollars: 20.0
    slippage_ticks_per_side: 0.50
""".strip(),
        encoding="utf-8",
    )
    return path


def _base_row(
    timestamp: pd.Timestamp,
    *,
    market: str,
    fold_id: str,
    entry: float,
    exit_: float,
) -> dict[str, object]:
    return {
        "market": market,
        "year": 2024,
        "fold_id": fold_id,
        "timestamp": timestamp,
        "session_id": timestamp.strftime("%Y-%m-%d"),
        "session_segment_id": "rth",
        "split_group": "research",
        "prediction_type": "classification_probability",
        "calibration_id": "no_calibration",
        "model_config_hash": "model-hash",
        "feature_config_hash": "feature-hash",
        "execution_open": entry,
        "execution_close": exit_,
        "target_valid": True,
        "target_entry_ts": timestamp + pd.Timedelta(minutes=1),
        "target_exit_ts": timestamp + pd.Timedelta(minutes=16),
        "minutes_until_session_close": 60.0,
    }


def _add_prediction_group(rows: list[dict[str, object]], base: dict[str, object], item: dict[str, object]) -> None:
    rows.extend(
        [
            {
                **base,
                "model_id": "ridge_return_v1",
                "model_family": "ridge_regression",
                "target_name": "target_ret_15m",
                "prediction_type": "regression",
                "y_true": item["ret_true"],
                "y_pred_raw": item["ret_pred"],
                "y_pred_calibrated": item["ret_pred"],
                "p_long": None,
                "p_short": None,
                "p_flat": None,
                "p_fade_success": None,
                "p_trend_danger": None,
            },
            {
                **base,
                "model_id": "logistic_direction_v1",
                "model_family": "logistic_regression",
                "target_name": "target_sign_with_deadzone",
                "y_true": item["direction_true"],
                "y_pred_raw": item["p_long"] - item["p_short"],
                "y_pred_calibrated": item["p_long"] - item["p_short"],
                "p_long": item["p_long"],
                "p_short": item["p_short"],
                "p_flat": item["p_flat"],
                "p_fade_success": None,
                "p_trend_danger": None,
            },
            {
                **base,
                "model_id": "logistic_fade_success_v1",
                "model_family": "logistic_regression",
                "target_name": "target_fade_success_15m",
                "y_true": int(item["p_fade"] >= 0.5),
                "y_pred_raw": item["p_fade"],
                "y_pred_calibrated": item["p_fade"],
                "p_long": None,
                "p_short": None,
                "p_flat": None,
                "p_fade_success": item["p_fade"],
                "p_trend_danger": None,
            },
            {
                **base,
                "model_id": "logistic_trend_danger_v1",
                "model_family": "logistic_regression",
                "target_name": "target_trend_danger_30m",
                "y_true": int(item["p_trend"] >= 0.5),
                "y_pred_raw": item["p_trend"],
                "y_pred_calibrated": item["p_trend"],
                "p_long": None,
                "p_short": None,
                "p_flat": None,
                "p_fade_success": None,
                "p_trend_danger": item["p_trend"],
            },
        ]
    )


def _write_predictions(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2024-01-02T14:00:00Z", periods=6, freq="30min")
    items = [
        {
            "market": "ES",
            "fold": "ES_research_0001",
            "entry": 100.0,
            "exit": 99.0,
            "p_long": 0.80,
            "p_short": 0.10,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": -1,
            "ret_true": -0.01,
            "ret_pred": -0.01,
        },
        {
            "market": "ES",
            "fold": "ES_research_0001",
            "entry": 100.0,
            "exit": 101.0,
            "p_long": 0.10,
            "p_short": 0.80,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.01,
            "ret_pred": 0.01,
        },
        {
            "market": "ES",
            "fold": "ES_research_0002",
            "entry": 100.0,
            "exit": 101.0,
            "p_long": 0.80,
            "p_short": 0.10,
            "p_flat": 0.10,
            "p_fade": 0.30,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.01,
            "ret_pred": 0.01,
        },
        {
            "market": "CL",
            "fold": "CL_research_0001",
            "entry": 80.0,
            "exit": 79.0,
            "p_long": 0.10,
            "p_short": 0.80,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.90,
            "direction_true": -1,
            "ret_true": -0.0125,
            "ret_pred": -0.0125,
        },
        {
            "market": "CL",
            "fold": "CL_research_0002",
            "entry": 80.0,
            "exit": 81.0,
            "p_long": 0.52,
            "p_short": 0.50,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": 0,
            "ret_true": 0.00,
            "ret_pred": 0.00,
        },
        {
            "market": "CL",
            "fold": "CL_research_0002",
            "entry": 81.0,
            "exit": 81.2,
            "p_long": 0.70,
            "p_short": 0.10,
            "p_flat": 0.20,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.0024691358024691358,
            "ret_pred": 0.0024691358024691358,
        },
    ]
    rows: list[dict[str, object]] = []
    for timestamp, item in zip(timestamps, items):
        _add_prediction_group(
            rows,
            _base_row(
                timestamp,
                market=str(item["market"]),
                fold_id=str(item["fold"]),
                entry=float(item["entry"]),
                exit_=float(item["exit"]),
            ),
            item,
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_policy_signal_alignment_detects_bad_traded_subset(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"

    report = build_policy_signal_alignment(
        predictions_path=predictions,
        costs_config=costs,
        output_root=output_root,
        run="fixture",
        policy=PolicyConfig(
            long_short_margin=0.05,
            min_fade_success=0.50,
            max_trend_danger=0.50,
        ),
    )

    for suffix in OUTPUT_SUFFIXES.values():
        assert (output_root / f"fixture_{suffix}").exists()
    assert report["prediction_count"] == 24
    assert report["policy_row_count"] == 6
    assert report["trade_count"] == 3
    assert report["decision"] == "direction_edge_calibration_issue_not_policy_logic_bug"
    assert report["overall"]["traded_target_direction_accuracy"] == 1 / 3
    assert report["overall"]["base_target_direction_accuracy"] == 0.6
    assert report["overall"]["all_row_argmax_direction_accuracy"] == 0.5
    assert report["overall"]["blocked_base_target_direction_accuracy"] == 1.0

    inversion = pd.read_csv(output_root / "fixture_policy_signal_inversion_check.csv")
    assert inversion.iloc[0]["inverted_net_delta"] > 0.0

    gates = pd.read_csv(output_root / "fixture_policy_signal_gate_effect.csv")
    assert "gate_category" in gates.columns

    summary = json.loads((output_root / "fixture_policy_signal_alignment_summary.json").read_text(encoding="utf-8"))
    assert len(summary["top_findings"]) == 5


def test_policy_signal_alignment_main_runs_cleanly(tmp_path: Path, monkeypatch) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_policy_signal_alignment",
            "--predictions",
            predictions.as_posix(),
            "--costs-config",
            costs.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--run",
            "fixture_cli",
        ],
    )

    assert main() == 0
    assert (output_root / "fixture_cli_policy_signal_alignment_summary.json").exists()
