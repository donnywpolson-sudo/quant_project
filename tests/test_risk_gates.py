from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.common.config import config
from pipeline.risk.risk import RiskGateError, run_risk_gates


def _risk_df(pnl: list[float], position: list[float] | None = None) -> pl.DataFrame:
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    n = len(pnl)
    position = position if position is not None else [0.0] * n
    equity = []
    cur = 100000.0
    peak = 100000.0
    dd = []
    for x in pnl:
        cur += x
        peak = max(peak, cur)
        equity.append(cur)
        dd.append(cur / peak - 1.0)
    return pl.DataFrame(
        {
            "ts_event": [start + timedelta(minutes=5 * i) for i in range(n)],
            "pnl": pnl,
            "position": position,
            "equity_curve": equity,
            "drawdown_pct": dd,
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("pnl").cast(pl.Float32),
        pl.col("position").cast(pl.Float32),
        pl.col("equity_curve").cast(pl.Float32),
        pl.col("drawdown_pct").cast(pl.Float32),
    )


def test_risk_gates_pass_without_optional_limits(monkeypatch):
    monkeypatch.setattr(config, "EQUITY", 100000.0, raising=False)
    monkeypatch.setattr(config, "MAX_POSITION_SIZE", float("inf"), raising=False)
    monkeypatch.delenv("QUANT_DAILY_LOSS_LIMIT", raising=False)
    monkeypatch.delenv("QUANT_MAX_DRAWDOWN_PCT", raising=False)

    report = run_risk_gates(_risk_df([10.0, -5.0, 2.0]))

    assert report["status"] == "PASS"
    assert report["total_pnl"] == 7.0


def test_risk_gates_fail_daily_loss_limit(monkeypatch):
    monkeypatch.setattr(config, "EQUITY", 100000.0, raising=False)
    monkeypatch.setattr(config, "MAX_POSITION_SIZE", float("inf"), raising=False)
    monkeypatch.setenv("QUANT_DAILY_LOSS_LIMIT", "100")

    with pytest.raises(RiskGateError, match="daily_loss_limit"):
        run_risk_gates(_risk_df([-50.0, -60.0]))


def test_risk_gates_fail_position_limit(monkeypatch):
    monkeypatch.setattr(config, "EQUITY", 100000.0, raising=False)
    monkeypatch.setattr(config, "MAX_POSITION_SIZE", 1.0, raising=False)
    monkeypatch.delenv("QUANT_DAILY_LOSS_LIMIT", raising=False)

    with pytest.raises(RiskGateError, match="max_position_size"):
        run_risk_gates(_risk_df([0.0, 1.0], position=[0.0, 2.0]))
