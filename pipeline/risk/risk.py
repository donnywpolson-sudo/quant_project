from __future__ import annotations

import math
import os
from typing import Any

import polars as pl

from pipeline.common.config import config


class RiskGateError(RuntimeError):
    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = value.replace("$", "").replace(",", "")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _limit_from_config_or_env(config_name: str, env_name: str) -> float | None:
    return _float_or_none(os.environ.get(env_name, getattr(config, config_name, None)))


def _gate(name: str, passed: bool, value: Any, limit: Any = None, severity: str = "FAIL") -> dict:
    return {
        "name": name,
        "status": "PASS" if passed else severity,
        "value": value,
        "limit": limit,
    }


def run_risk_gates(df: pl.DataFrame, *, context: dict | None = None) -> dict:
    """
    Step 10 boundary: post-execution risk gates.

    Hard-fails only configured limits. Always returns a report if gates pass.
    """
    if df.height == 0:
        raise RiskGateError("RISK FAIL: empty execution result")
    required = ["ts_event", "pnl", "position", "equity_curve", "drawdown_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RiskGateError(f"RISK FAIL: missing columns {missing}")

    starting_equity = _float_or_none(getattr(config, "EQUITY", 100000.0)) or 100000.0
    if starting_equity <= 0:
        raise RiskGateError(f"RISK FAIL: EQUITY must be positive, got {starting_equity}")

    pnl = df["pnl"].to_numpy().astype(float)
    pos = df["position"].to_numpy().astype(float)
    equity = df["equity_curve"].to_numpy().astype(float)
    dd = df["drawdown_pct"].to_numpy().astype(float)

    arrays = {"pnl": pnl, "position": pos, "equity_curve": equity, "drawdown_pct": dd}
    for name, arr in arrays.items():
        if not math.isfinite(float(arr.sum())):
            raise RiskGateError(f"RISK FAIL: non-finite values in {name}")

    total_pnl = float(pnl.sum())
    max_abs_position = float(abs(pos).max()) if len(pos) else 0.0
    min_equity = float(equity.min()) if len(equity) else starting_equity
    max_drawdown_pct = float(dd.min()) if len(dd) else 0.0

    daily = (
        df.with_columns(pl.col("ts_event").dt.date().alias("_risk_date"))
        .group_by("_risk_date")
        .agg(pl.col("pnl").sum().alias("daily_pnl"))
    )
    min_daily_pnl = float(daily["daily_pnl"].min()) if daily.height else 0.0

    max_pos_limit = _float_or_none(getattr(config, "MAX_POSITION_SIZE", None))
    if max_pos_limit is not None and math.isinf(max_pos_limit):
        max_pos_limit = None
    daily_loss_limit = _limit_from_config_or_env("DAILY_LOSS_LIMIT", "QUANT_DAILY_LOSS_LIMIT")
    max_dd_limit = _limit_from_config_or_env("MAX_DRAWDOWN_PCT", "QUANT_MAX_DRAWDOWN_PCT")
    if max_dd_limit is not None and max_dd_limit > 0:
        max_dd_limit = -abs(max_dd_limit)

    gates = [
        _gate("equity_positive", min_equity > 0.0, round(min_equity, 6), 0.0),
        _gate(
            "max_position_size",
            True if max_pos_limit is None else max_abs_position <= max_pos_limit + 1e-9,
            round(max_abs_position, 6),
            max_pos_limit,
        ),
        _gate(
            "daily_loss_limit",
            True if daily_loss_limit is None else min_daily_pnl >= -abs(daily_loss_limit) - 1e-9,
            round(min_daily_pnl, 6),
            None if daily_loss_limit is None else -abs(daily_loss_limit),
        ),
        _gate(
            "max_drawdown_pct",
            True if max_dd_limit is None else max_drawdown_pct >= max_dd_limit - 1e-12,
            round(max_drawdown_pct, 8),
            max_dd_limit,
        ),
    ]

    failed = [g for g in gates if g["status"] == "FAIL"]
    report = {
        "status": "FAIL" if failed else "PASS",
        "context": context or {},
        "rows": df.height,
        "total_pnl": round(total_pnl, 6),
        "starting_equity": round(starting_equity, 2),
        "min_equity": round(min_equity, 6),
        "max_drawdown_pct": round(max_drawdown_pct, 8),
        "min_daily_pnl": round(min_daily_pnl, 6),
        "max_abs_position": round(max_abs_position, 6),
        "gates": gates,
    }
    if failed:
        sample = "; ".join(f'{g["name"]} value={g["value"]} limit={g["limit"]}' for g in failed)
        raise RiskGateError(f"RISK FAIL: {sample}", report=report)
    print(
        f'[RISK] PASS rows={df.height} pnl={total_pnl:.2f} '
        f'max_dd={max_drawdown_pct:.4%} min_daily_pnl={min_daily_pnl:.2f} '
        f'max_abs_position={max_abs_position:.3f}',
        flush=True,
    )
    return report
