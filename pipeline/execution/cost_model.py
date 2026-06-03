from __future__ import annotations

from typing import Any

import polars as pl

from pipeline.common.market import get_contract_multiplier, get_tick_value


def instrument_terms(symbol: str | None) -> dict[str, float | str]:
    if not symbol:
        return {"symbol": "UNKNOWN", "contract_multiplier": 1.0, "tick_value": 1.0, "unit": "synthetic_unit"}
    try:
        return {
            "symbol": symbol,
            "contract_multiplier": get_contract_multiplier(symbol),
            "tick_value": get_tick_value(symbol),
            "unit": "USD",
        }
    except Exception:
        return {"symbol": symbol, "contract_multiplier": 1.0, "tick_value": 1.0, "unit": "synthetic_unit"}


def attach_execution_cost_model(
    df: pl.DataFrame,
    *,
    target_col: str,
    config: Any,
    symbol: str | None = None,
    feature_set_id: str = "",
) -> pl.DataFrame:
    if "raw_signal" not in df.columns:
        raise ValueError("execution cost model requires raw_signal")
    out = df
    if "ts_event" in out.columns and "prediction_time" not in out.columns:
        if out["ts_event"].dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            exec_time = pl.col("ts_event") + int(config.execution.entry_lag_bars)
        else:
            exec_time = pl.col("ts_event") + pl.duration(minutes=int(config.execution.entry_lag_bars))
        out = out.with_columns(pl.col("ts_event").alias("prediction_time"), exec_time.alias("execution_time"))
    fill = "open" if "open" in out.columns else "close"
    if fill not in out.columns:
        raise ValueError("missing fill price column: open/close")
    terms = instrument_terms(symbol)
    multiplier = float(terms["contract_multiplier"])
    tick_value = float(terms["tick_value"])
    max_contracts = float(getattr(config.execution, "max_contracts", 1))
    min_hold = int(getattr(config.execution, "min_position_hold_bars", 0) or 0)
    commission = float(getattr(config.execution, "commission_per_contract", 0.0))
    exchange = float(getattr(config.execution, "exchange_fees_per_contract", 0.0))
    spread_ticks = float(getattr(config.execution, "spread_ticks", 0.0))
    slippage_ticks = float(getattr(config.execution, "slippage_ticks", 0.0))
    scale = float(getattr(getattr(config, "target", object()), "target_scale_factor", 1.0) or 1.0)
    if "label_target_scale_factor" in out.columns:
        scale_expr = pl.col("label_target_scale_factor").cast(pl.Float64).fill_null(scale)
    else:
        scale_expr = pl.lit(scale)
    target_log_ret = pl.col(target_col).cast(pl.Float64).fill_null(0.0) / scale_expr
    target_return = target_log_ret.exp() - 1.0
    desired = out.select(pl.col("raw_signal").clip(-max_contracts, max_contracts).cast(pl.Float64)).to_series().to_list()
    position_values = _apply_min_position_hold(desired, min_hold)
    out = out.with_columns(
        pl.Series("position_after", position_values),
        pl.Series("position", position_values),
        pl.col(fill).cast(pl.Float64).alias("assumed_fill_price"),
        pl.lit(min_hold).alias("min_position_hold_bars"),
    )
    out = out.with_columns(pl.col("position_after").shift(1).fill_null(0).alias("position_before"))
    out = out.with_columns((pl.col("position_after") - pl.col("position_before")).alias("position_delta"))
    fee_per_delta = commission + exchange
    slip_per_delta = (slippage_ticks + spread_ticks * 0.5) * tick_value
    out = out.with_columns(
        pl.col(target_col).cast(pl.Float64).alias("ret_exec"),
        pl.col(target_col).cast(pl.Float64).alias("target_exec"),
        target_return.alias("target_return_exec"),
        (pl.col("assumed_fill_price") * target_return * multiplier).alias("target_exec_usd_per_contract"),
        (pl.col("position_after") * pl.col("assumed_fill_price") * target_return * multiplier).alias("gross_pnl"),
        (pl.col("position_delta").abs() * fee_per_delta).alias("fees"),
        (pl.col("position_delta").abs() * slip_per_delta).alias("slippage"),
        pl.lit(multiplier).alias("contract_multiplier"),
        pl.lit(tick_value).alias("tick_value"),
        pl.lit(str(terms["unit"])).alias("pnl_unit"),
        pl.lit(feature_set_id).alias("feature_set_id"),
    )
    out = out.with_columns((pl.col("fees") + pl.col("slippage")).alias("costs"))
    out = out.with_columns((pl.col("gross_pnl") - pl.col("costs")).alias("pnl"))
    return out.with_columns(
        pl.col("pnl").alias("net_pnl"),
        pl.col("pnl").cum_sum().alias("equity_curve"),
    ).with_columns((pl.col("equity_curve") - pl.col("equity_curve").cum_max()).alias("drawdown_pct"))


def _apply_min_position_hold(desired: list[float], min_hold_bars: int) -> list[float]:
    if min_hold_bars <= 0:
        return desired
    out: list[float] = []
    pos = 0.0
    bars_held = min_hold_bars
    for raw in desired:
        want = float(raw or 0.0)
        if want != pos and bars_held >= min_hold_bars:
            pos = want
            bars_held = 0
        else:
            bars_held += 1
        out.append(pos)
    return out
