import polars as pl
import pytest

from pipeline.common.config import config
from pipeline.execution.simulator import (
    _compute_pnl_from_target_exec,
    validate_execution_simulation_result,
)


def test_execution_rounds_fractional_futures_contracts_to_integer() -> None:
    config.EPS = 1e-9
    config.STOP_LOSS_PCT = 0.0
    config.TAKE_PROFIT_PCT = 0.0
    config.GAP_SLIPPAGE_PCT = 0.0
    config.COMMISSION_PER_CONTRACT = 0.0
    config.TX_COST_PER_ROUNDTURN = 0.0
    config.EQUITY = 100000.0

    df = pl.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "target_exec": [0.34, 0.51, -0.51],
        }
    )

    out = _compute_pnl_from_target_exec(df, contract_multiplier=50.0)

    assert out["target_exec_sized"].to_list() == pytest.approx([0.34, 0.51, -0.51])
    assert out["target_exec"].to_list() == [0.0, 1.0, -1.0]
    assert out["position"].to_list() == [0.0, 0.0, 1.0]


def test_execution_validation_rejects_fractional_contracts() -> None:
    df = pl.DataFrame(
        {
            "ts_event": [1, 2],
            "prediction_prob": [0.6, 0.4],
            "raw_signal": [1.0, -1.0],
            "target_exec": [0.5, -1.0],
            "target_exec_sized": [0.5, -1.0],
            "position": [0.0, 1.0],
            "pos_change": [0.0, 1.0],
            "ret_exec": [0.0, 0.01],
            "gross_pnl": [0.0, 10.0],
            "pnl": [0.0, 9.0],
            "notional_traded": [0.0, 1000.0],
            "commission_cost": [0.0, 1.0],
            "transaction_cost": [0.0, 0.0],
            "slippage_cost": [0.0, 0.0],
            "execution_cost": [0.0, 1.0],
            "return_on_equity": [0.0, 0.00009],
            "gross_return_on_equity": [0.0, 0.00010],
            "equity_curve": [100000.0, 100009.0],
            "gross_equity_curve": [100000.0, 100010.0],
            "drawdown_pct": [0.0, 0.0],
        }
    )

    with pytest.raises(RuntimeError, match="fractional contracts"):
        validate_execution_simulation_result(df)


def test_execution_validation_rejects_pnl_accounting_mismatch() -> None:
    df = pl.DataFrame(
        {
            "ts_event": [1, 2],
            "prediction_prob": [0.6, 0.4],
            "raw_signal": [1.0, -1.0],
            "target_exec": [1.0, -1.0],
            "target_exec_sized": [1.0, -1.0],
            "position": [0.0, 1.0],
            "pos_change": [0.0, 1.0],
            "ret_exec": [0.0, 0.01],
            "gross_pnl": [0.0, 10.0],
            "pnl": [0.0, 10.0],
            "notional_traded": [0.0, 1000.0],
            "commission_cost": [0.0, 1.0],
            "transaction_cost": [0.0, 0.0],
            "slippage_cost": [0.0, 0.0],
            "execution_cost": [0.0, 1.0],
            "return_on_equity": [0.0, 0.00010],
            "gross_return_on_equity": [0.0, 0.00010],
            "equity_curve": [100000.0, 100010.0],
            "gross_equity_curve": [100000.0, 100010.0],
            "drawdown_pct": [0.0, 0.0],
        }
    )

    with pytest.raises(RuntimeError, match="pnl != gross_pnl"):
        validate_execution_simulation_result(df)
