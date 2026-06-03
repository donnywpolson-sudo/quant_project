from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows


def compute_metrics(df: pl.DataFrame) -> dict[str, Any]:
    pnl_col = "net_pnl" if "net_pnl" in df.columns else "pnl"
    pnl = df[pnl_col].cast(pl.Float64)
    total = float(pnl.sum() or 0.0)
    mean = float(pnl.mean() or 0.0)
    std = float(pnl.std() or 0.0)
    sharpe = 0.0 if std == 0 else mean / std * math.sqrt(252)
    downside = pnl.filter(pnl < 0)
    sortino = 0.0 if downside.std() in (None, 0) else mean / float(downside.std()) * math.sqrt(252)
    equity = pnl.cum_sum()
    peak = equity.cum_max()
    dd = equity - peak
    max_dd = float(dd.min() or 0.0)
    wins = pnl.filter(pnl > 0)
    losses = pnl.filter(pnl < 0)
    gross_win = float(wins.sum() or 0.0)
    gross_loss = abs(float(losses.sum() or 0.0))
    pos_delta = df["position_delta"].abs() if "position_delta" in df.columns else pl.Series([0.0] * df.height)
    true_trades = int((pos_delta > 0).sum()) if len(pos_delta) else 0
    return {
        "net_pnl": total,
        "sharpe": sharpe,
        "sharpe_annualized": sharpe,
        "sortino": sortino,
        "calmar": 0.0 if max_dd == 0 else total / abs(max_dd),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd / max(abs(float(peak.max() or 0.0)), 1.0),
        "bars": int(df.height),
        "trades": true_trades,
        "trade_count": true_trades,
        "hit_rate": float((pnl > 0).mean() or 0.0),
        "active_bar_hit_rate": float((pnl.filter(pnl != 0) > 0).mean() or 0.0) if pnl.filter(pnl != 0).len() else 0.0,
        "trade_hit_rate": float((pnl > 0).mean() or 0.0),
        "average_win": float(wins.mean() or 0.0),
        "average_loss": float(losses.mean() or 0.0),
        "average_win_loss_ratio": 0.0 if float(losses.mean() or 0.0) == 0 else abs(float(wins.mean() or 0.0) / float(losses.mean())),
        "profit_factor": float("inf") if gross_loss == 0 and gross_win > 0 else (gross_win / gross_loss if gross_loss else 0.0),
        "turnover": float(pos_delta.sum() or 0.0) if len(pos_delta) else 0.0,
        "turnover_per_bar": float(pos_delta.mean() or 0.0) if len(pos_delta) else 0.0,
        "tail_loss_5pct": float(pnl.quantile(0.05) or 0.0),
        "skew": float(pnl.skew() or 0.0),
        "kurtosis": float(pnl.kurtosis() or 0.0),
    }


def _sharpe_pair(values, bars_per_year: float = 252.0) -> tuple[float, float]:
    s = pl.Series(values).cast(pl.Float64)
    mean = float(s.mean() or 0.0)
    std = float(s.std() or 0.0)
    sharpe = 0.0 if std == 0 else mean / std * math.sqrt(float(bars_per_year))
    return mean, sharpe


def compute_ic(pred, target) -> dict[str, Any]:
    df = pl.DataFrame({"pred": pred, "target": target}).drop_nulls()
    if df.height < 3:
        return {"spearman_ic": None, "pearson_ic": None}
    try:
        pearson = df.select(pl.corr("pred", "target").alias("c")).item()
        spearman = df.with_columns(
            pl.col("pred").rank().alias("pred_rank"),
            pl.col("target").rank().alias("target_rank"),
        ).select(pl.corr("pred_rank", "target_rank").alias("c")).item()
    except Exception:
        return {"spearman_ic": None, "pearson_ic": None}
    return {"spearman_ic": spearman, "pearson_ic": pearson}


def compute_backtest_metrics(df: pl.DataFrame) -> dict[str, Any]:
    metrics = compute_metrics(df)
    pnl_col = "pnl" if "pnl" in df.columns else ("net_pnl" if "net_pnl" in df.columns else None)
    pnl = df[pnl_col].cast(pl.Float64) if pnl_col else pl.Series([0.0])
    bars_per_year = 252 * 390
    if "ts_event" in df.columns and df.height > 2:
        try:
            ts_min = df["ts_event"].min()
            ts_max = df["ts_event"].max()
            days = max((ts_max - ts_min).total_seconds() / 86400.0, 1.0)
            bars_per_year = max(int(df.height / days * 365.25), 1)
        except Exception:
            pass
    mean = float(pnl.mean() or 0.0)
    std = float(pnl.std() or 0.0)
    sharpe_per_bar = 0.0 if std == 0 else mean / std
    pos_delta = df["position_delta"].abs() if "position_delta" in df.columns else pl.Series([0.0] * df.height)
    active = pnl.filter(pnl != 0)
    trade_pnl = pnl.filter(pos_delta > 0) if len(pos_delta) == len(pnl) else active
    metrics.update(
        {
            "total_pnl": float(pnl.sum() or 0.0),
            "bars_per_year": bars_per_year,
            "sharpe_per_bar": sharpe_per_bar,
            "sharpe_annualized": sharpe_per_bar * math.sqrt(bars_per_year),
            "bar_hit_rate_all_bars": float((pnl > 0).mean() or 0.0),
            "bar_hit_rate_all_bars_n": df.height,
            "bar_hit_rate_active_bars": float((active > 0).mean() or 0.0) if active.len() else 0.0,
            "bar_hit_rate_active_bars_n": active.len(),
            "trade_hit_rate": float((trade_pnl > 0).mean() or 0.0) if trade_pnl.len() else "NA",
            "trade_hit_rate_n": trade_pnl.len(),
            "position_change_events": int((pos_delta > 0).sum()) if len(pos_delta) else 0,
            "position_turnover": float(pos_delta.sum() or 0.0) if len(pos_delta) else 0.0,
            "bars": df.height,
            "trades": int((pos_delta > 0).sum()) if len(pos_delta) else 0,
            "trade_count": int((pos_delta > 0).sum()) if len(pos_delta) else 0,
        }
    )
    return metrics


def write_metrics_report(df: pl.DataFrame, profile: str, report_dir: str = "reports/metrics") -> dict[str, Any]:
    metrics = compute_metrics(df)
    metrics.setdefault("modeling_mode", "unknown")
    path = Path(report_dir) / f"{profile}_metrics_report.json"
    atomic_write_json(path, metrics)
    write_csv_rows(Path(report_dir) / f"{profile}_metrics_summary.csv", [metrics])
    return metrics
