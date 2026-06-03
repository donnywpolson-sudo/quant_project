import math

import polars as pl

from pipeline.analytics.aggregate import compute_backtest_metrics
from pipeline.analytics.failure_attribution import compute_failure_attribution
from pipeline.common.config import ExecutionConfig, RootConfig, TargetConfig
from pipeline.execution.cost_model import attach_execution_cost_model
from pipeline.cli import run_modeling_pipeline


def test_execution_cost_model_uses_futures_usd_units_for_known_symbol():
    cfg = RootConfig(
        target=TargetConfig(target_scale_factor=100.0),
        execution=ExecutionConfig(
            entry_lag_bars=1,
            commission_per_contract=1.50,
            exchange_fees_per_contract=0.0,
            slippage_ticks=1.0,
            spread_ticks=1.0,
            max_contracts=1,
        ),
    )
    df = pl.DataFrame(
        {
            "ts_event": [1, 2],
            "open": [100.0, 100.0],
            "target_15m_ret": [1.0, -1.0],
            "label_target_scale_factor": [100.0, 100.0],
            "raw_signal": [1, -1],
        }
    )
    out = attach_execution_cost_model(df, target_col="target_15m_ret", config=cfg, symbol="ES")
    first = out.row(0, named=True)
    expected_gross = 100.0 * (math.exp(0.01) - 1.0) * 50.0
    assert abs(first["gross_pnl"] - expected_gross) < 1e-9
    assert first["fees"] == 1.5
    assert first["slippage"] == 18.75
    assert first["pnl"] == first["gross_pnl"] - first["fees"] - first["slippage"]
    assert first["pnl_unit"] == "USD"


def test_metrics_trade_count_uses_position_changes_not_bars():
    df = pl.DataFrame({"pnl": [0.0, 1.0, -1.0, 0.0], "position_delta": [0, 1, 0, 2]})
    metrics = compute_backtest_metrics(df)
    assert metrics["bars"] == 4
    assert metrics["trades"] == 2
    assert metrics["trade_count"] == 2
    assert metrics["position_change_events"] == 2


def test_failure_attribution_identifies_turnover_cost_drag():
    df = pl.DataFrame(
        {
            "gross_pnl": [1.0, 1.0, 1.0, 1.0],
            "fees": [1.0, 1.0, 1.0, 1.0],
            "slippage": [1.0, 1.0, 1.0, 1.0],
            "costs": [2.0, 2.0, 2.0, 2.0],
            "net_pnl": [-1.0, -1.0, -1.0, -1.0],
            "position_after": [1, -1, 1, -1],
            "position_delta": [1, 2, 2, 2],
            "prediction": [0.1, -0.1, 0.2, -0.2],
        }
    )
    diag = compute_failure_attribution(df, symbol="ES", split_id="1", modeling_mode="full_research")
    assert diag["dominant_failure"] == "turnover_cost_drag"
    assert diag["position_change_events"] == 4
    assert diag["costs"] > abs(diag["gross_pnl"])


def test_prediction_entry_threshold_flattens_small_minimal_predictions():
    cfg = RootConfig(
        execution=ExecutionConfig(prediction_entry_threshold=0.5, slippage_ticks=0.0, spread_ticks=0.0, commission_per_contract=0.0),
    )
    df = pl.DataFrame(
        {
            "ts_event": [1, 2, 3, 4],
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0] * 4,
            "volume": [1] * 4,
            "x": [-0.1, 0.1, 1.0, -1.0],
            "target_15m_ret": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = run_modeling_pipeline(df, ["x"], "target_15m_ret", None, None, None, None, {"config": cfg, "symbol": "ES"})
    assert out["raw_signal"].to_list() == [0, 0, 1, -1]
    assert out["signal_entry_threshold"].unique().to_list() == [0.5]


def test_min_position_hold_bars_reduces_immediate_flips():
    cfg = RootConfig(
        target=TargetConfig(target_scale_factor=1.0),
        execution=ExecutionConfig(
            min_position_hold_bars=2,
            slippage_ticks=0.0,
            spread_ticks=0.0,
            commission_per_contract=0.0,
        ),
    )
    df = pl.DataFrame(
        {
            "ts_event": [1, 2, 3, 4, 5, 6],
            "open": [100.0] * 6,
            "target_15m_ret": [0.0] * 6,
            "raw_signal": [1, -1, -1, 0, 0, 1],
        }
    )
    out = attach_execution_cost_model(df, target_col="target_15m_ret", config=cfg, symbol="ES")
    assert out["position_after"].to_list() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert out["position_delta"].abs().sum() == 2.0
    assert out["min_position_hold_bars"].unique().to_list() == [2]
