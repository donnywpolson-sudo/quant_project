from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase8_model_selection.audit_label_feature_sanity import (  # noqa: E402
    build_label_feature_sanity,
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


def _base_prediction(
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
        "causal_valid": True,
        "close": entry,
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
    timestamps = pd.date_range("2024-01-02T14:00:00Z", periods=4, freq="30min")
    items = [
        ("ES", "ES_research_0001", 100.0, 101.0, 0.80, 0.10, 0.80, 0.10, 1, 0.01, 1000.0),
        ("ES", "ES_research_0001", 100.0, 99.0, 0.75, 0.10, 0.80, 0.10, -1, -0.01, -1000.0),
        ("CL", "CL_research_0001", 80.0, 79.0, 0.10, 0.80, 0.80, 0.10, -1, -0.0125, -1000.0),
        ("CL", "CL_research_0002", 80.0, 81.0, 0.52, 0.50, 0.80, 0.10, 1, 0.0125, 1000.0),
    ]
    rows: list[dict[str, object]] = []
    for timestamp, item in zip(timestamps, items):
        market, fold, entry, exit_, p_long, p_short, p_fade, p_trend, direction, ret_true, ret_pred = item
        _add_prediction_group(
            rows,
            _base_prediction(timestamp, market=market, fold_id=fold, entry=entry, exit_=exit_),
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


def _write_feature_matrix(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "feature_cols.json").write_text(
        json.dumps(["feature_alpha", "feature_beta", "feature_sparse"]),
        encoding="utf-8",
    )
    timestamps = pd.date_range("2024-01-02T14:00:00Z", periods=4, freq="30min")
    rows = [
        {
            "ts": timestamps[0],
            "market": "ES",
            "year": 2024,
            "close": 100.0,
            "causal_valid": True,
            "feature_input_valid": True,
            "feature_row_valid": True,
            "training_row_valid": True,
            "target_ret_15m": 0.01,
            "target_ret_ticks_15m": 4.0,
            "target_net_ticks_after_est_cost": 2.0,
            "target_gross_dollars_15m": 50.0,
            "target_sign_15m": 1,
            "target_sign_with_deadzone": 1,
            "target_tradeable_after_cost": True,
            "target_valid": True,
            "target_entry_price": 100.0,
            "target_exit_price": 101.0,
            "target_entry_ts": timestamps[0] + pd.Timedelta(minutes=1),
            "target_exit_ts": timestamps[0] + pd.Timedelta(minutes=16),
            "feature_alpha": 1.0,
            "feature_beta": 0.0,
            "feature_sparse": 1.0,
        },
        {
            "ts": timestamps[1],
            "market": "ES",
            "year": 2024,
            "close": 100.0,
            "causal_valid": True,
            "feature_input_valid": True,
            "feature_row_valid": True,
            "training_row_valid": True,
            "target_ret_15m": -0.01,
            "target_ret_ticks_15m": -4.0,
            "target_net_ticks_after_est_cost": -2.0,
            "target_gross_dollars_15m": -50.0,
            "target_sign_15m": -1,
            "target_sign_with_deadzone": -1,
            "target_tradeable_after_cost": True,
            "target_valid": True,
            "target_entry_price": 100.0,
            "target_exit_price": 99.0,
            "target_entry_ts": timestamps[1] + pd.Timedelta(minutes=1),
            "target_exit_ts": timestamps[1] + pd.Timedelta(minutes=16),
            "feature_alpha": 10.0,
            "feature_beta": 0.0,
            "feature_sparse": None,
        },
    ]
    es_path = root / "ES" / "2024.parquet"
    es_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(es_path, index=False)

    cl_rows = [
        {
            "ts": timestamps[2],
            "market": "CL",
            "year": 2024,
            "close": 80.0,
            "causal_valid": True,
            "feature_input_valid": True,
            "feature_row_valid": True,
            "training_row_valid": True,
            "target_ret_15m": -0.0125,
            "target_ret_ticks_15m": -10.0,
            "target_net_ticks_after_est_cost": -8.0,
            "target_gross_dollars_15m": -100.0,
            "target_sign_15m": -1,
            "target_sign_with_deadzone": -1,
            "target_tradeable_after_cost": True,
            "target_valid": True,
            "target_entry_price": 80.0,
            "target_exit_price": 79.0,
            "target_entry_ts": timestamps[2] + pd.Timedelta(minutes=1),
            "target_exit_ts": timestamps[2] + pd.Timedelta(minutes=16),
            "feature_alpha": 5.0,
            "feature_beta": 1.0,
            "feature_sparse": 2.0,
        },
        {
            "ts": timestamps[3],
            "market": "CL",
            "year": 2024,
            "close": 80.0,
            "causal_valid": True,
            "feature_input_valid": True,
            "feature_row_valid": True,
            "training_row_valid": True,
            "target_ret_15m": 0.0125,
            "target_ret_ticks_15m": 10.0,
            "target_net_ticks_after_est_cost": 8.0,
            "target_gross_dollars_15m": 100.0,
            "target_sign_15m": 1,
            "target_sign_with_deadzone": 1,
            "target_tradeable_after_cost": True,
            "target_valid": True,
            "target_entry_price": 80.0,
            "target_exit_price": 81.0,
            "target_entry_ts": timestamps[3] + pd.Timedelta(minutes=1),
            "target_exit_ts": timestamps[3] + pd.Timedelta(minutes=16),
            "feature_alpha": 2.0,
            "feature_beta": 1.0,
            "feature_sparse": None,
        },
    ]
    cl_path = root / "CL" / "2024.parquet"
    cl_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cl_rows).to_parquet(cl_path, index=False)
    return root


def test_label_feature_sanity_writes_alignment_and_shift_reports(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    feature_root = _write_feature_matrix(tmp_path / "data" / "feature_matrices" / "baseline")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"

    report = build_label_feature_sanity(
        predictions_path=predictions,
        costs_config=costs,
        feature_root=feature_root,
        output_root=output_root,
        run="fixture",
        max_shift_features=10,
    )

    assert (output_root / "fixture_label_feature_sanity_summary.json").exists()
    assert (output_root / "fixture_target_alignment.csv").exists()
    assert (output_root / "fixture_label_balance_by_market_fold.csv").exists()
    assert (output_root / "fixture_feature_shift_top.csv").exists()
    assert (output_root / "fixture_return_training_scale_check.csv").exists()
    assert report["matched_feature_row_count"] == 4
    assert report["target_alignment_overall"]["observed_feature_return_match_rate"] == 1.0
    assert report["target_alignment_overall"]["max_abs_feature_vs_execution_return_diff"] < 1e-12

    balance = pd.read_csv(output_root / "fixture_label_balance_by_market_fold.csv")
    assert set(balance["market"]) == {"ES", "CL"}

    shift = pd.read_csv(output_root / "fixture_feature_shift_top.csv")
    assert "feature_alpha" in set(shift["feature"])

    scale = pd.read_csv(output_root / "fixture_return_training_scale_check.csv")
    overall = scale[scale["scope"].eq("overall")].iloc[0]
    assert overall["same_target_units_reported"]
    assert overall["prediction_to_y_true_std_ratio"] > 100.0

    summary = json.loads((output_root / "fixture_label_feature_sanity_summary.json").read_text(encoding="utf-8"))
    assert len(summary["top_findings"]) == 5


def test_label_feature_sanity_main_runs_cleanly(tmp_path: Path, monkeypatch) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    feature_root = _write_feature_matrix(tmp_path / "data" / "feature_matrices" / "baseline")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_label_feature_sanity",
            "--predictions",
            predictions.as_posix(),
            "--costs-config",
            costs.as_posix(),
            "--feature-root",
            feature_root.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--run",
            "fixture_cli",
        ],
    )

    assert main() == 0
    assert (output_root / "fixture_cli_label_feature_sanity_summary.json").exists()
