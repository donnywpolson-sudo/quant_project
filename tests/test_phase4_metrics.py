from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from scripts.build_phase4_metrics import build_phase4_metrics


def _write_inputs(root: Path, *, low_sharpe: bool = False, nonfinite: bool = False, empty: bool = False) -> dict[str, Path]:
    reports = root / "reports" / "wfa"
    reports.mkdir(parents=True, exist_ok=True)
    cfg = root / "configs"
    markets = cfg / "markets"
    markets.mkdir(parents=True, exist_ok=True)
    matrix = root / "data" / "feature_matrices" / "baseline"
    matrix.mkdir(parents=True, exist_ok=True)

    (cfg / "alpha_tiered.yaml").write_text(
        """
active_profile: tier_0_smoke_pipeline
base:
  execution:
    tx_cost_per_roundturn: 0.0
    commission_per_contract: 0.0
profiles:
  tier_0_smoke_pipeline: {}
""".strip(),
        encoding="utf-8",
    )
    (markets / "ES.yaml").write_text(
        """
metadata:
  ticker: ES
  contract_multiplier: 50
contract_specs:
  tick_size: 0.25
  tick_value: 12.5
risk:
  slippage_k: 0.00001
""".strip(),
        encoding="utf-8",
    )
    for name, payload in [
        ("feature_cols.json", ["f1"]),
        ("target_cols.json", ["target_ret_15m", "target_valid"]),
        ("metadata_cols.json", ["market", "prediction_ts"]),
        ("excluded_cols.json", ["open", "close"]),
    ]:
        (matrix / name).write_text(json.dumps(payload), encoding="utf-8")

    rows = []
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    if not empty:
        for i in range(20):
            pred = 1.0 if i % 2 == 0 else -1.0
            target = (-0.001 * pred) if low_sharpe else (0.001 if pred > 0 else -0.001)
            rows.append(
                {
                    "market": "ES",
                    "prediction_ts": ts + timedelta(minutes=i),
                    "bar_end_ts": ts + timedelta(minutes=i),
                    "session_segment_id": 1,
                    "target_ret_15m": target,
                    "target_valid": True,
                    "fold_id": 0,
                    "prediction": float("nan") if nonfinite and i == 0 else pred,
                    "feature_count": 1,
                    "train_start": ts - timedelta(days=10),
                    "train_end": ts,
                    "test_start": ts,
                    "test_end": ts + timedelta(days=1),
                    "purge_bars": 15,
                    "embargo_bars": 15,
                }
            )
        rows.append(
            {
                "market": "ES",
                "prediction_ts": ts - timedelta(minutes=1),
                "bar_end_ts": ts - timedelta(minutes=1),
                "session_segment_id": 1,
                "target_ret_15m": 0.10,
                "target_valid": False,
                "fold_id": 0,
                "prediction": 99.0,
                "feature_count": 1,
                "train_start": ts - timedelta(days=10),
                "train_end": ts,
                "test_start": ts,
                "test_end": ts + timedelta(days=1),
                "purge_bars": 15,
                "embargo_bars": 15,
            }
        )
    schema = {
        "market": pl.String,
        "prediction_ts": pl.Datetime(time_zone="UTC"),
        "bar_end_ts": pl.Datetime(time_zone="UTC"),
        "session_segment_id": pl.Int64,
        "target_ret_15m": pl.Float64,
        "target_valid": pl.Boolean,
        "fold_id": pl.Int64,
        "prediction": pl.Float64,
        "feature_count": pl.Int64,
        "train_start": pl.Datetime(time_zone="UTC"),
        "train_end": pl.Datetime(time_zone="UTC"),
        "test_start": pl.Datetime(time_zone="UTC"),
        "test_end": pl.Datetime(time_zone="UTC"),
        "purge_bars": pl.Int64,
        "embargo_bars": pl.Int64,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(reports / "oos_predictions.parquet")
    pl.DataFrame([{"fold_id": 0, "market": "ES", "train_rows": 100, "test_rows": len(rows)}]).write_csv(
        reports / "fold_summary.csv"
    )
    pl.DataFrame([{"fold_id": 0, "market": "ES", "status": "planned"}]).write_csv(reports / "split_plan.csv")
    (reports / "manifest.json").write_text(json.dumps({"fold_count": 1, "oos_prediction_rows": len(rows)}))
    return {
        "predictions": reports / "oos_predictions.parquet",
        "fold_summary": reports / "fold_summary.csv",
        "split_plan": reports / "split_plan.csv",
        "wfa_manifest": reports / "manifest.json",
        "config": cfg / "alpha_tiered.yaml",
        "market_config_dir": markets,
        "registry_root": matrix,
        "out": root / "reports" / "metrics",
    }


def _run(paths: dict[str, Path]) -> dict:
    return build_phase4_metrics(
        predictions_path=paths["predictions"],
        fold_summary_path=paths["fold_summary"],
        split_plan_path=paths["split_plan"],
        wfa_manifest_path=paths["wfa_manifest"],
        config_path=paths["config"],
        market_config_dir=paths["market_config_dir"],
        registry_root=paths["registry_root"],
        out_dir=paths["out"],
    )


def test_metrics_are_oos_only_and_write_gross_net_costs_gate_and_groups(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    summary = _run(paths)
    assert summary["oos_rows_used"] == 20
    assert summary["rows_skipped"] == 1
    assert "cost_assumptions" in summary
    assert summary["gate"]["passed"] is True
    for rel in [
        "baseline_metrics.csv",
        "fold_metrics.csv",
        "market_metrics.csv",
        "diagnostics.csv",
        "cost_breakdown.csv",
        "turnover_diagnostics.csv",
        "signal_diagnostics.csv",
        "run_summary.json",
        "baseline_gate.json",
        "strategy_gate.json",
    ]:
        assert (paths["out"] / rel).exists()
    baseline = pl.read_csv(paths["out"] / "baseline_metrics.csv")
    assert "gross_return" in baseline.columns
    assert "net_return" in baseline.columns
    overall = baseline.row(0, named=True)
    assert abs((overall["gross_return"] - overall["net_return"]) - overall["cost_drag"]) < 1e-9
    assert overall["sharpe_bars_per_year"] == 69552
    assert pl.read_csv(paths["out"] / "market_metrics.csv").height == 1
    assert pl.read_csv(paths["out"] / "fold_metrics.csv").height == 1
    costs = pl.read_csv(paths["out"] / "cost_breakdown.csv").filter(pl.col("group") == "overall").row(0, named=True)
    assert costs["cost_drag_equals_gross_minus_net"]
    assert costs["slippage_cost"] > 0
    assert costs["commission_proxy_cost"] > 0
    assert costs["turnover_cost"] > 0
    saved_summary = json.loads((paths["out"] / "run_summary.json").read_text(encoding="utf-8"))
    assert saved_summary["cost_units"]["target_ret_15m"] == "decimal_return"
    assert saved_summary["cost_assumptions"]["ES"]["commission_return_per_contract_change"] == 0.00005
    assert saved_summary["metric_conventions"]["bars_per_year"] == 69552
    strategy_gate = json.loads((paths["out"] / "strategy_gate.json").read_text(encoding="utf-8"))
    assert strategy_gate["gate_name"] == "strategy_accept_reject_gate"
    assert strategy_gate["source_gate"] == "phase4_baseline_sanity_gate"
    assert set(["structural_gate", "performance_gate", "overall_passed"]).issubset(strategy_gate)
    assert strategy_gate["structural_gate"]["passed"] is True
    assert strategy_gate["performance_gate"]["passed"] is False
    assert strategy_gate["overall_passed"] is False
    assert "turnover_per_bar_extreme" in strategy_gate["performance_gate"]["failures"]
    assert "max_turnover_per_bar_fail" in strategy_gate["performance_gate"]["thresholds"]


def test_gate_fails_for_missing_predictions(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, empty=True)
    summary = _run(paths)
    gate = json.loads((paths["out"] / "baseline_gate.json").read_text(encoding="utf-8"))
    strategy_gate = json.loads((paths["out"] / "strategy_gate.json").read_text(encoding="utf-8"))
    assert summary["oos_rows_used"] == 0
    assert gate["passed"] is False
    assert "no_oos_predictions" in gate["failures"]
    assert strategy_gate["structural_gate"]["passed"] is False
    assert strategy_gate["overall_passed"] is False


def test_gate_fails_for_nonfinite_predictions(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, nonfinite=True)
    summary = _run(paths)
    assert summary["gate"]["passed"] is False
    assert "nonfinite_predictions" in summary["gate"]["failures"]


def test_gate_does_not_fail_merely_because_sharpe_is_low(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, low_sharpe=True)
    summary = _run(paths)
    assert summary["metrics"]["net_sharpe"] < 0
    assert summary["gate"]["passed"] is True
    assert "weak_sharpe" not in " ".join(summary["gate"]["failures"])
    assert summary["strategy_gate"]["structural_gate"]["passed"] is True
    assert summary["strategy_gate"]["performance_gate"]["passed"] is False
    assert "net_sharpe_below_minimum" in summary["strategy_gate"]["performance_gate"]["failures"]
    assert "negative_net_return" in summary["strategy_gate"]["performance_gate"]["failures"]
    assert summary["strategy_gate"]["final_verdict"] == "economically_rejected"


def test_strategy_gate_does_not_mark_economic_rejection_as_passed(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, low_sharpe=True)
    summary = _run(paths)
    strategy_gate = json.loads((paths["out"] / "strategy_gate.json").read_text(encoding="utf-8"))

    assert strategy_gate["structural_gate"]["passed"] is True
    assert strategy_gate["performance_gate"]["passed"] is False
    assert strategy_gate["overall_passed"] is False
    assert strategy_gate["final_verdict"] == "economically_rejected"
    assert "net_sharpe_below_minimum" not in strategy_gate["structural_gate"]["failures"]
    assert "net_sharpe_below_minimum" in strategy_gate["performance_gate"]["failures"]
    assert summary["strategy_gate"] == strategy_gate


def test_turnover_trade_count_and_long_short_flip_cost_are_explicit(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    _run(paths)
    turnover = pl.read_csv(paths["out"] / "turnover_diagnostics.csv").filter(pl.col("group") == "overall").row(0, named=True)
    assert turnover["trade_count_position_change_events"] == 20
    assert turnover["position_turnover_units"] == 39.0
    assert turnover["flip_count"] == 19
    assert turnover["long_to_short_flip_count"] == 10
    assert turnover["short_to_long_flip_count"] == 9
    assert turnover["long_short_flip_position_change_units"] == 2.0
    assert turnover["cost_treatment"] == "costs_charged_per_absolute_position_change_unit_not_per_bar"
    signal = pl.read_csv(paths["out"] / "signal_diagnostics.csv").filter(pl.col("group") == "overall").row(0, named=True)
    assert signal["hold_previous_signal_until_sign_changes"]
