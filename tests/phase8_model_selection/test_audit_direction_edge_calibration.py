from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase8_model_selection.audit_direction_edge_calibration import (  # noqa: E402
    OUTPUT_SUFFIXES,
    build_direction_edge_calibration,
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
    timestamps = pd.date_range("2024-01-02T14:00:00Z", periods=8, freq="30min")
    items = [
        # Current margin trades these high-flat rows and loses.
        ("ES", "ES_research_0001", 100.0, 99.0, 0.46, 0.36, 0.45, -1, -0.01),
        ("ES", "ES_research_0001", 100.0, 99.0, 0.44, 0.34, 0.43, -1, -0.01),
        ("CL", "CL_research_0001", 80.0, 81.0, 0.34, 0.44, 0.43, 1, 0.0125),
        # Stronger direction-vs-flat rows win.
        ("ES", "ES_research_0002", 100.0, 101.0, 0.76, 0.10, 0.14, 1, 0.01),
        ("CL", "CL_research_0002", 80.0, 79.0, 0.10, 0.76, 0.14, -1, -0.0125),
        ("ES", "ES_research_0002", 100.0, 100.0, 0.21, 0.20, 0.70, 0, 0.0),
        ("CL", "CL_research_0002", 80.0, 80.0, 0.20, 0.21, 0.70, 0, 0.0),
        ("ES", "ES_research_0003", 100.0, 101.0, 0.70, 0.10, 0.20, 1, 0.01),
    ]
    rows: list[dict[str, object]] = []
    for timestamp, item in zip(timestamps, items):
        market, fold, entry, exit_, p_long, p_short, p_flat, direction, ret_true = item
        _add_prediction_group(
            rows,
            _base_row(
                timestamp,
                market=market,
                fold_id=fold,
                entry=entry,
                exit_=exit_,
            ),
            {
                "p_long": p_long,
                "p_short": p_short,
                "p_flat": p_flat,
                "p_fade": 0.80,
                "p_trend": 0.10,
                "direction_true": direction,
                "ret_true": ret_true,
                "ret_pred": ret_true,
            },
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_direction_edge_calibration_writes_flat_aware_reports(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"

    report = build_direction_edge_calibration(
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
    assert report["prediction_count"] == 32
    assert report["policy_row_count"] == 8
    assert report["current_edge"]["trade_count"] == 6
    assert report["current_edge"]["target_direction_accuracy"] < 1.0

    scenarios = pd.read_csv(output_root / "fixture_direction_edge_scenarios.csv")
    assert {"current_margin", "direction_beats_flat", "argmax_nonflat"}.issubset(set(scenarios["edge_mode"]))
    assert scenarios.iloc[0]["net_return_dollars"] > report["current_edge"]["net_return_dollars"]

    flat = pd.read_csv(output_root / "fixture_direction_flat_suppression.csv")
    assert flat["max_flat_probability"].min() == 0.25

    summary = json.loads((output_root / "fixture_direction_edge_calibration_summary.json").read_text(encoding="utf-8"))
    assert len(summary["top_findings"]) == 5


def test_direction_edge_calibration_main_runs_cleanly(tmp_path: Path, monkeypatch) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_direction_edge_calibration",
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
    assert (output_root / "fixture_cli_direction_edge_calibration_summary.json").exists()
