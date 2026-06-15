from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase8_model_selection.audit_policy_failure import build_failure_breakdown  # noqa: E402
from scripts.phase8_model_selection.evaluate_predictions import PolicyConfig  # noqa: E402


def _write_costs(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
markets:
  ES:
    point_value: 50.0
    tick_value: 12.5
    round_turn_cost_dollars: 10.0
    slippage_ticks_per_side: 0.25
""".strip(),
        encoding="utf-8",
    )
    return path


def _base_row(timestamp: pd.Timestamp, *, entry: float, exit_: float) -> dict[str, object]:
    return {
        "market": "ES",
        "year": 2024,
        "fold_id": "ES_research_0001",
        "timestamp": timestamp,
        "session_id": "2024-01-01",
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


def _write_predictions(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp("2024-01-02T14:30:00Z")
    base = _base_row(timestamp, entry=100.0, exit_=99.0)
    rows = [
        {
            **base,
            "model_id": "ridge_return_v1",
            "model_family": "ridge_regression",
            "target_name": "target_ret_15m",
            "prediction_type": "regression",
            "y_true": -1.0,
            "y_pred_raw": -0.01,
            "y_pred_calibrated": -0.01,
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
            "y_true": -1,
            "y_pred_raw": -0.60,
            "y_pred_calibrated": -0.60,
            "p_long": 0.10,
            "p_short": 0.70,
            "p_flat": 0.20,
            "p_fade_success": None,
            "p_trend_danger": None,
        },
        {
            **base,
            "model_id": "logistic_fade_success_v1",
            "model_family": "logistic_regression",
            "target_name": "target_fade_success_15m",
            "y_true": 1,
            "y_pred_raw": 0.80,
            "y_pred_calibrated": 0.80,
            "p_long": None,
            "p_short": None,
            "p_flat": None,
            "p_fade_success": 0.80,
            "p_trend_danger": None,
        },
        {
            **base,
            "model_id": "logistic_trend_danger_v1",
            "model_family": "logistic_regression",
            "target_name": "target_trend_danger_30m",
            "y_true": 0,
            "y_pred_raw": 0.20,
            "y_pred_calibrated": 0.20,
            "p_long": None,
            "p_short": None,
            "p_flat": None,
            "p_fade_success": None,
            "p_trend_danger": 0.20,
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_build_failure_breakdown_writes_expected_slices(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "run" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"

    report = build_failure_breakdown(
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

    assert report["prediction_count"] == 4
    assert report["policy_row_count"] == 1
    assert report["trade_count"] == 1
    assert report["overall"]["net_return_dollars"] == 40.0

    summary = json.loads((output_root / "fixture_summary.json").read_text(encoding="utf-8"))
    by_direction = pd.read_csv(output_root / "fixture_by_direction.csv")
    model_target = pd.read_csv(output_root / "fixture_model_target_summary.csv")
    cost_components = pd.read_csv(output_root / "fixture_cost_components.csv")

    assert summary["trade_count"] == 1
    assert set(by_direction["position_label"]) == {"short"}
    assert set(model_target["target_name"]) == {
        "target_ret_15m",
        "target_sign_with_deadzone",
        "target_fade_success_15m",
        "target_trend_danger_30m",
    }
    assert cost_components.loc[0, "slippage_cost_dollars"] == 6.25
    assert cost_components.loc[0, "commission_cost_dollars"] == 3.75
