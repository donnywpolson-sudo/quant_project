from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pipeline.common.io_safe import atomic_write_json


TRACE_COLS = [
    "ts_event", "prediction_time", "execution_time", "prediction_prob", "raw_signal",
    "signal_entry_threshold", "min_position_hold_bars",
    "position_before", "position_after", "position_delta", "order_side",
    "assumed_fill_price", "slippage", "fees", "gross_pnl", "net_pnl", "pnl",
    "equity_curve", "drawdown_pct", "reason_flat", "reason_trade",
]


def build_execution_trace(df: pl.DataFrame, context: dict[str, Any] | None = None, max_rows: int = 200) -> pl.DataFrame:
    cols = [c for c in TRACE_COLS if c in df.columns]
    return df.select(cols).head(max_rows) if cols else df.head(max_rows)


def validate_execution_trace(trace_df: pl.DataFrame, config: Any) -> dict[str, Any]:
    failures = []
    cols = set(trace_df.columns)
    if getattr(config.execution, "reject_same_bar_fill", True) and {"prediction_time", "execution_time"}.issubset(cols):
        bad = trace_df.filter(pl.col("execution_time") <= pl.col("prediction_time")).height
        if bad:
            failures.append(f"same-bar/noncausal fills rows={bad}")
    for c in ["fees", "slippage"]:
        if c in cols:
            bad = trace_df.filter(pl.col(c) < 0).height
            if bad:
                failures.append(f"negative {c} rows={bad}")
    if {"pnl", "gross_pnl", "fees", "slippage"}.issubset(cols):
        bad = trace_df.filter((pl.col("pnl") - (pl.col("gross_pnl") - pl.col("fees") - pl.col("slippage"))).abs() > 1e-8).height
        if bad:
            failures.append(f"pnl arithmetic mismatch rows={bad}")
    limit = getattr(config.execution, "max_contracts", None) or getattr(config.execution, "max_position_size", None)
    if limit and "position_after" in cols:
        bad = trace_df.filter(pl.col("position_after").abs() > float(limit)).height
        if bad:
            failures.append(f"position exceeds limit rows={bad}")
    for c in [c for c in ["assumed_fill_price", "pnl", "gross_pnl", "net_pnl"] if c in cols]:
        bad = trace_df.filter(~pl.col(c).is_finite()).height
        if bad:
            failures.append(f"non-finite {c} rows={bad}")
    return {"status": "FAIL" if failures else "PASS", "failures": failures}


def write_execution_trace_outputs(trace: pl.DataFrame, report: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace.write_parquet(out_dir / "execution_trace.parquet")
    trace.write_csv(out_dir / "execution_trace.csv")
    atomic_write_json(out_dir / "execution_trace_report.json", report)
