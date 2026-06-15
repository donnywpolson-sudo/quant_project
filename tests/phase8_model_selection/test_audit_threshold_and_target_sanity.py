from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase8_model_selection.audit_threshold_and_target_sanity import (  # noqa: E402
    build_threshold_and_target_sanity,
    main,
)


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
        ("ES", "ES_research_0001", 100.0, 102.0, 0.80, 0.10, 0.80, 0.10, 1, 0.02, 1000.0),
        ("CL", "CL_research_0001", 80.0, 78.0, 0.75, 0.10, 0.55, 0.10, -1, -0.02, -1000.0),
        ("ES", "ES_research_0002", 101.0, 100.0, 0.10, 0.80, 0.80, 0.10, -1, -0.01, -1000.0),
        ("ES", "ES_research_0002", 100.0, 101.0, 0.80, 0.10, 0.80, 0.90, 1, 0.01, 1000.0),
        ("CL", "CL_research_0002", 80.0, 81.0, 0.52, 0.50, 0.80, 0.10, 0, 0.0, 0.0),
        ("CL", "CL_research_0002", 81.0, 82.0, 0.70, 0.10, 0.30, 0.10, 1, 0.01, 1000.0),
    ]
    rows: list[dict[str, object]] = []
    for timestamp, item in zip(timestamps, items):
        market, fold, entry, exit_, p_long, p_short, p_fade, p_trend, direction, ret_true, ret_pred = item
        _add_prediction_group(
            rows,
            _base_row(timestamp, market=market, fold_id=fold, entry=entry, exit_=exit_),
            {
                "p_long": p_long,
                "p_short": p_short,
                "p_flat": 0.10,
                "p_fade": p_fade,
                "p_trend": p_trend,
                "direction_true": direction,
                "ret_true": ret_true,
                "ret_pred": ret_pred,
            },
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_threshold_and_target_sanity_writes_decision_outputs(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"

    report = build_threshold_and_target_sanity(
        predictions_path=predictions,
        costs_config=costs,
        output_root=output_root,
        run="fixture",
        direction_margin_threshold=0.30,
        min_fade_success=0.50,
        max_trend_danger=0.50,
        min_total_trades=100,
        min_positive_markets=2,
        min_positive_folds=4,
    )

    assert (output_root / "fixture_threshold_stability.csv").exists()
    assert (output_root / "fixture_return_target_scale_audit.csv").exists()
    assert (output_root / "fixture_next_action_summary.json").exists()
    assert report["threshold_stability"]["total_trade_count"] == 3
    assert report["threshold_stability"]["net_return_dollars"] == -90.0
    assert report["threshold_stability"]["stable_threshold_region"] is False
    assert report["next_action"] == "stop_policy_work_and_audit_labels_features"
    assert report["return_target_scale"]["return_target_scale_status"] == "flagged"

    threshold = pd.read_csv(output_root / "fixture_threshold_stability.csv")
    assert {"market", "fold", "side", "month", "hour"}.issubset(set(threshold["scope"]))
    overall = threshold[threshold["scope"].eq("overall")].iloc[0]
    assert overall["gross_return_dollars"] == -50.0
    assert overall["cost_dollars"] == 40.0

    scale = pd.read_csv(output_root / "fixture_return_target_scale_audit.csv")
    global_scale = scale[scale["scope"].eq("overall")].iloc[0]
    assert "extreme" in global_scale["scale_warnings"]

    summary = json.loads((output_root / "fixture_next_action_summary.json").read_text(encoding="utf-8"))
    assert len(summary["top_findings"]) == 5
    assert summary["return_target_action"] == "fix_or_explain_ridge_return_scale_before_return_magnitude_logic"


def test_threshold_and_target_sanity_main_runs_cleanly(tmp_path: Path, monkeypatch) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_threshold_and_target_sanity",
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
    assert (output_root / "fixture_cli_next_action_summary.json").exists()
