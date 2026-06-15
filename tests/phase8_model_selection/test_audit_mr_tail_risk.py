from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase8_model_selection.audit_mr_tail_risk import (  # noqa: E402
    MRTailPolicyConfig,
    OUTPUT_SUFFIXES,
    build_mr_tail_audit,
    build_mr_tail_policy_frame,
    main,
)


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
    realized_volatility: float,
    mae_ticks: float,
    mfe_ticks: float,
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
        "realized_volatility_15m": realized_volatility,
        "mae_ticks_15m": mae_ticks,
        "mfe_ticks_15m": mfe_ticks,
    }


def _add_prediction_group(
    rows: list[dict[str, object]],
    base: dict[str, object],
    item: dict[str, object],
) -> None:
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
            "exit": 101.0,
            "p_long": 0.80,
            "p_short": 0.10,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.01,
            "ret_pred": 0.01,
            "vol": 0.10,
            "mae": -2.0,
            "mfe": 5.0,
        },
        {
            "market": "ES",
            "fold": "ES_research_0001",
            "entry": 100.0,
            "exit": 98.0,
            "p_long": 0.80,
            "p_short": 0.10,
            "p_flat": 0.10,
            "p_fade": 0.82,
            "p_trend": 0.90,
            "direction_true": -1,
            "ret_true": -0.02,
            "ret_pred": 0.01,
            "vol": 0.50,
            "mae": -10.0,
            "mfe": 1.0,
        },
        {
            "market": "CL",
            "fold": "CL_research_0001",
            "entry": 80.0,
            "exit": 79.5,
            "p_long": 0.10,
            "p_short": 0.80,
            "p_flat": 0.10,
            "p_fade": 0.70,
            "p_trend": 0.20,
            "direction_true": -1,
            "ret_true": -0.00625,
            "ret_pred": -0.01,
            "vol": 0.20,
            "mae": -3.0,
            "mfe": 6.0,
        },
        {
            "market": "ES",
            "fold": "ES_research_0002",
            "entry": 100.0,
            "exit": 101.0,
            "p_long": 0.75,
            "p_short": 0.10,
            "p_flat": 0.15,
            "p_fade": 0.40,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.01,
            "ret_pred": 0.01,
            "vol": 0.30,
            "mae": -1.0,
            "mfe": 4.0,
        },
        {
            "market": "ES",
            "fold": "ES_research_0002",
            "entry": 100.0,
            "exit": 101.0,
            "p_long": 0.70,
            "p_short": 0.10,
            "p_flat": 0.20,
            "p_fade": 0.65,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.01,
            "ret_pred": 0.003,
            "vol": 0.40,
            "mae": -1.0,
            "mfe": 2.0,
        },
        {
            "market": "CL",
            "fold": "CL_research_0002",
            "entry": 80.0,
            "exit": 81.0,
            "p_long": 0.52,
            "p_short": 0.50,
            "p_flat": 0.10,
            "p_fade": 0.75,
            "p_trend": 0.10,
            "direction_true": 0,
            "ret_true": 0.0,
            "ret_pred": 0.02,
            "vol": 0.60,
            "mae": -1.0,
            "mfe": 3.0,
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
                realized_volatility=float(item["vol"]),
                mae_ticks=float(item["mae"]),
                mfe_ticks=float(item["mfe"]),
            ),
            item,
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _config() -> MRTailPolicyConfig:
    return MRTailPolicyConfig(
        long_short_margin=0.05,
        trend_danger_cutoff=0.50,
        fade_success_cutoff=0.50,
        edge_buffer_dollars=5.0,
    )


def test_policy_overlay_has_no_future_leakage(tmp_path: Path) -> None:
    predictions_path = _write_predictions(
        tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet"
    )
    costs_path = _write_costs(tmp_path / "configs" / "costs.yaml")
    predictions = pd.read_parquet(predictions_path)

    frame, failures, _ = build_mr_tail_policy_frame(predictions, costs_path, _config())
    assert failures == []

    mutated = predictions.copy()
    mutated["execution_close"] = mutated["execution_open"]
    mutated.loc[mutated["target_name"].eq("target_ret_15m"), "y_true"] = 99.0
    mutated["mae_ticks_15m"] = -999.0
    mutated["mfe_ticks_15m"] = 999.0
    mutated_frame, mutated_failures, _ = build_mr_tail_policy_frame(mutated, costs_path, _config())
    assert mutated_failures == []

    decision_cols = [
        "market",
        "timestamp",
        "no_trend_position",
        "overlay_position",
        "policy_reason",
        "blocked_by_trend_danger",
    ]
    assert frame[decision_cols].equals(mutated_frame[decision_cols])
    assert frame["net_dollars"].sum() != mutated_frame["net_dollars"].sum()


def test_mr_tail_audit_costs_trend_block_and_reports(tmp_path: Path) -> None:
    predictions_path = _write_predictions(
        tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet"
    )
    costs_path = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_mr_tail_audit"

    report = build_mr_tail_audit(
        predictions_path=predictions_path,
        costs_config=costs_path,
        output_root=output_root,
        run="fixture",
        config=_config(),
    )

    for suffix in OUTPUT_SUFFIXES.values():
        assert (output_root / f"fixture_{suffix}").exists()

    assert report["prediction_count"] == 24
    assert report["policy_row_count"] == 6
    assert report["overall"]["trade_count"] == 2
    assert report["overall"]["gross_return_dollars"] == 100.0
    assert report["overall"]["cost_dollars"] == 30.0
    assert report["overall"]["net_return_dollars"] == 70.0
    assert report["overall"]["net_cvar_95_dollars"] == 30.0
    assert report["overall"]["mean_trade_mae_ticks"] == -4.0
    assert report["trend_block_trade_delta"] == -1
    assert report["trend_block_net_delta"] == 110.0

    comparison = pd.read_csv(output_root / "fixture_policy_comparison.csv").set_index("scenario")
    assert comparison.loc["no_trend_danger_block", "trade_count"] == 3
    assert comparison.loc["no_trend_danger_block", "net_return_dollars"] == -40.0
    assert comparison.loc["mr_tail_overlay", "net_return_dollars"] == 70.0

    buckets = pd.read_csv(output_root / "fixture_bucket_summary.csv")
    assert {
        "trend_danger_decile",
        "fade_success_decile",
        "realized_volatility_regime",
        "market",
        "session_hour_utc",
        "net_cvar_99_dollars",
    }.issubset(buckets.columns)

    blocked = pd.read_csv(output_root / "fixture_blocked_trend_danger_opportunities.csv")
    assert blocked["trade_count"].sum() == 1
    assert blocked["net_return_dollars"].sum() == -110.0

    summary = json.loads((output_root / "fixture_summary.json").read_text(encoding="utf-8"))
    assert summary["max_losing_streak"] == 0


def test_mr_tail_reports_are_deterministic(tmp_path: Path) -> None:
    predictions_path = _write_predictions(
        tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet"
    )
    costs_path = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_mr_tail_audit"

    build_mr_tail_audit(
        predictions_path=predictions_path,
        costs_config=costs_path,
        output_root=output_root,
        run="fixture",
        config=_config(),
    )
    first = {
        path.name: path.read_bytes()
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }
    build_mr_tail_audit(
        predictions_path=predictions_path,
        costs_config=costs_path,
        output_root=output_root,
        run="fixture",
        config=_config(),
    )
    second = {
        path.name: path.read_bytes()
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }
    assert first == second


def test_mr_tail_main_runs_cleanly(tmp_path: Path, monkeypatch) -> None:
    predictions_path = _write_predictions(
        tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet"
    )
    costs_path = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_mr_tail_audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_mr_tail_risk",
            "--predictions",
            predictions_path.as_posix(),
            "--costs-config",
            costs_path.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--run",
            "fixture_cli",
            "--edge-buffer-dollars",
            "5.0",
        ],
    )

    assert main() == 0
    assert (output_root / "fixture_cli_summary.json").exists()
