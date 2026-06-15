from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase8_model_selection.audit_trade_failure_drilldown import (  # noqa: E402
    OUTPUT_SUFFIXES,
    build_trade_failure_drilldown,
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
    regime: str,
    atr: float,
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
        "signal_regime": regime,
        "feature_atr_60": atr,
    }


def _add_policy_rows(rows: list[dict[str, object]], base: dict[str, object], item: dict[str, object]) -> None:
    rows.append(
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
        }
    )
    rows.append(
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
        }
    )
    rows.append(
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
        }
    )
    rows.append(
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
        }
    )


def _write_predictions(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2024-01-02T14:00:00Z", periods=6, freq="30min")
    items = [
        {
            "market": "ES",
            "fold": "ES_research_0001",
            "entry": 100.0,
            "exit": 102.0,
            "p_long": 0.80,
            "p_short": 0.10,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.02,
            "ret_pred": 0.01,
            "regime": "trend",
            "atr": 1.0,
        },
        {
            "market": "CL",
            "fold": "CL_research_0001",
            "entry": 80.0,
            "exit": 78.0,
            "p_long": 0.75,
            "p_short": 0.10,
            "p_flat": 0.15,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": -1,
            "ret_true": -0.02,
            "ret_pred": 0.01,
            "regime": "range",
            "atr": 2.0,
        },
        {
            "market": "ES",
            "fold": "ES_research_0002",
            "entry": 101.0,
            "exit": 100.0,
            "p_long": 0.10,
            "p_short": 0.80,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.10,
            "direction_true": -1,
            "ret_true": -0.01,
            "ret_pred": -0.01,
            "regime": "trend",
            "atr": 3.0,
        },
        {
            "market": "ES",
            "fold": "ES_research_0002",
            "entry": 100.0,
            "exit": 101.0,
            "p_long": 0.80,
            "p_short": 0.10,
            "p_flat": 0.10,
            "p_fade": 0.80,
            "p_trend": 0.90,
            "direction_true": 1,
            "ret_true": 0.01,
            "ret_pred": 0.01,
            "regime": "trend",
            "atr": 4.0,
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
            "regime": "range",
            "atr": 5.0,
        },
        {
            "market": "CL",
            "fold": "CL_research_0002",
            "entry": 81.0,
            "exit": 82.0,
            "p_long": 0.70,
            "p_short": 0.10,
            "p_flat": 0.20,
            "p_fade": 0.30,
            "p_trend": 0.10,
            "direction_true": 1,
            "ret_true": 0.01,
            "ret_pred": 0.01,
            "regime": "range",
            "atr": 6.0,
        },
    ]
    rows: list[dict[str, object]] = []
    for timestamp, item in zip(timestamps, items):
        base = _base_row(
            timestamp,
            market=str(item["market"]),
            fold_id=str(item["fold"]),
            entry=float(item["entry"]),
            exit_=float(item["exit"]),
            regime=str(item["regime"]),
            atr=float(item["atr"]),
        )
        _add_policy_rows(rows, base, item)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_trade_failure_drilldown_writes_files_and_reconciles_metrics(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"

    report = build_trade_failure_drilldown(
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
    assert report["totals"]["gross_return_dollars"] == -50.0
    assert report["totals"]["cost_dollars"] == 40.0
    assert report["totals"]["net_return_dollars"] == -90.0

    counterfactuals = pd.read_csv(output_root / "fixture_cost_counterfactuals.csv")
    current = counterfactuals.set_index("scenario").loc["current"]
    zero_cost = counterfactuals.set_index("scenario").loc["zero_cost"]
    half_slippage = counterfactuals.set_index("scenario").loc["slippage_50pct"]
    assert current["net_return_dollars"] == -90.0
    assert zero_cost["net_return_dollars"] == -50.0
    assert half_slippage["net_return_dollars"] == -80.0

    long_breakdown = pd.read_csv(output_root / "fixture_long_failure_breakdown.csv")
    long_overall = long_breakdown[long_breakdown["scope"].eq("long_overall")].iloc[0]
    assert long_overall["trade_count"] == 2
    assert long_overall["net_return_dollars"] == -130.0

    blocked = pd.read_csv(output_root / "fixture_blocked_vs_traded_opportunity.csv")
    assert set(blocked["blocked_category"]) == {"traded", "no-direction", "trend-danger", "other"}
    assert blocked.set_index("blocked_category").loc["trend-danger", "row_count"] == 1

    volatility = pd.read_csv(output_root / "fixture_pnl_by_volatility_bucket.csv")
    regime = pd.read_csv(output_root / "fixture_pnl_by_regime_bucket.csv")
    assert "unavailable_reason" not in volatility.columns
    assert "unavailable_reason" not in regime.columns
    assert report["unavailable_diagnostics"] == []

    summary = json.loads((output_root / "fixture_trade_drilldown_summary.json").read_text(encoding="utf-8"))
    assert summary["gross_positive_before_costs"] is False
    assert len(summary["top_findings"]) == 5


def test_trade_failure_drilldown_main_runs_cleanly(tmp_path: Path, monkeypatch) -> None:
    predictions = _write_predictions(tmp_path / "data" / "predictions" / "fixture" / "oos_predictions.parquet")
    costs = _write_costs(tmp_path / "configs" / "costs.yaml")
    output_root = tmp_path / "reports" / "phase8_failure_breakdown"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_trade_failure_drilldown",
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
    assert (output_root / "fixture_cli_trade_drilldown_summary.json").exists()
