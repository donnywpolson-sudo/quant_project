from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows


def compute_failure_attribution(df: pl.DataFrame, *, symbol: str = "", split_id: str = "", modeling_mode: str = "") -> dict[str, Any]:
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "pnl"
    pnl = df[pnl_col].cast(pl.Float64) if pnl_col in df.columns else pl.Series([0.0])
    gross = df["gross_pnl"].cast(pl.Float64) if "gross_pnl" in df.columns else pnl
    costs = df["costs"].cast(pl.Float64) if "costs" in df.columns else pl.Series([0.0] * df.height)
    fees = df["fees"].cast(pl.Float64) if "fees" in df.columns else pl.Series([0.0] * df.height)
    slip = df["slippage"].cast(pl.Float64) if "slippage" in df.columns else pl.Series([0.0] * df.height)
    pos = df["position_after"].cast(pl.Float64) if "position_after" in df.columns else (df["position"].cast(pl.Float64) if "position" in df.columns else pl.Series([0.0] * df.height))
    delta = df["position_delta"].abs().cast(pl.Float64) if "position_delta" in df.columns else pl.Series([0.0] * df.height)
    pred = df["prediction"].cast(pl.Float64) if "prediction" in df.columns else pl.Series([])
    target_usd = df["target_exec_usd_per_contract"].cast(pl.Float64) if "target_exec_usd_per_contract" in df.columns else gross
    n = max(df.height, 1)
    gross_sum = float(gross.sum() or 0.0)
    cost_sum = float(costs.sum() or 0.0)
    row = {
        "symbol": symbol,
        "split_id": split_id,
        "modeling_mode": modeling_mode,
        "rows": df.height,
        "gross_pnl": gross_sum,
        "fees": float(fees.sum() or 0.0),
        "slippage": float(slip.sum() or 0.0),
        "costs": cost_sum,
        "net_pnl": float(pnl.sum() or 0.0),
        "cost_to_abs_gross_ratio": cost_sum / max(abs(gross_sum), 1e-12),
        "position_change_events": int((delta > 0).sum()) if len(delta) else 0,
        "position_turnover": float(delta.sum() or 0.0),
        "turnover_per_bar": float(delta.mean() or 0.0),
        "long_frac": float((pos > 0).sum() or 0) / n,
        "short_frac": float((pos < 0).sum() or 0) / n,
        "flat_frac": float((pos == 0).sum() or 0) / n,
        "mean_abs_position": float(pos.abs().mean() or 0.0),
        "prediction_mean": float(pred.mean() or 0.0) if len(pred) else 0.0,
        "prediction_std": float(pred.std() or 0.0) if len(pred) else 0.0,
        "prediction_min": float(pred.min() or 0.0) if len(pred) else 0.0,
        "prediction_max": float(pred.max() or 0.0) if len(pred) else 0.0,
        "sign_flipped_net_pnl": float((-gross - costs).sum() or 0.0),
        "always_flat_net_pnl": 0.0,
        "always_long_gross_pnl": float(target_usd.sum() or 0.0),
        "always_short_gross_pnl": float((-target_usd).sum() or 0.0),
    }
    row["dominant_failure"] = _dominant_failure(row)
    return row


def write_failure_attribution_report(df: pl.DataFrame, out_prefix: str | Path, **context: Any) -> dict[str, Any]:
    row = compute_failure_attribution(df, **context)
    report = {"status": "PASS", "diagnostic": row}
    out_prefix = Path(out_prefix)
    atomic_write_json(out_prefix.with_suffix(".json"), report)
    write_csv_rows(out_prefix.with_suffix(".csv"), [row])
    return report


def _dominant_failure(row: dict[str, Any]) -> str:
    if row["prediction_std"] == 0:
        return "constant_predictions"
    if row["position_turnover"] > row["rows"] * 0.5 and row["costs"] > abs(row["gross_pnl"]):
        return "turnover_cost_drag"
    if abs(row["gross_pnl"]) < row["costs"]:
        return "weak_gross_edge_vs_costs"
    if row["net_pnl"] < 0 and row["sign_flipped_net_pnl"] > 0:
        return "possible_signal_sign_issue"
    return "requires_review"
