from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

from pipeline.analytics.aggregate import compute_backtest_metrics
from pipeline.common.config import RootConfig
from pipeline.execution.cost_model import attach_execution_cost_model
from pipeline.validation.diagnostic_io import stringify_diagnostic_keys


DIAG_CSV = Path("reports/validation/prediction_threshold_diagnostics.csv")
DIAG_JSON = Path("reports/validation/prediction_threshold_diagnostics.json")
GRID_CSV = Path("reports/validation/threshold_candidate_grid.csv")
GRID_JSON = Path("reports/validation/threshold_candidate_grid.json")
ECON_CSV = Path("reports/validation/threshold_candidate_economics.csv")
ECON_JSON = Path("reports/validation/threshold_candidate_economics.json")

QUANTILES = {
    "prediction_p001": 0.001,
    "prediction_p005": 0.005,
    "prediction_p01": 0.01,
    "prediction_p05": 0.05,
    "prediction_p10": 0.10,
    "prediction_p25": 0.25,
    "prediction_p50": 0.50,
    "prediction_p75": 0.75,
    "prediction_p90": 0.90,
    "prediction_p95": 0.95,
    "prediction_p99": 0.99,
    "prediction_p995": 0.995,
    "prediction_p999": 0.999,
}
ABS_QUANTILES = {
    "abs_prediction_p90": 0.90,
    "abs_prediction_p95": 0.95,
    "abs_prediction_p99": 0.99,
    "abs_prediction_p995": 0.995,
    "abs_prediction_p999": 0.999,
}
FIXED_THRESHOLDS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25]
CANDIDATE_THRESHOLD_COUNT = len(ABS_QUANTILES) + len(FIXED_THRESHOLDS)

DIAG_FIELDS = [
    "run_id", "profile", "config_env", "created_at",
    "symbol", "split", "prediction_col", "prediction_nonnull", "prediction_min",
    *QUANTILES.keys(),
    "prediction_max", "prediction_mean", "prediction_std",
    "current_long_threshold", "current_short_threshold",
    "bars_above_current_long", "bars_below_current_short", "active_pct_at_current_threshold",
    *ABS_QUANTILES.keys(),
]
GRID_FIELDS = [
    "run_id", "profile", "config_env", "created_at",
    "symbol", "split", "threshold_type", "threshold_value", "long_bars", "short_bars",
    "active_bar_pct", "turnover_proxy", "current_threshold_flag",
]
ECON_FIELDS = [
    "run_id", "profile", "config_env", "created_at",
    "symbol", "split", "threshold_type", "threshold_value", "threshold_source",
    "long_bars", "short_bars", "active_bar_pct", "turnover",
    "gross_pnl", "net_pnl", "cost_drag", "cost_drag_pct_of_gross",
    "pnl_per_turnover", "sharpe_annualized", "current_threshold_flag",
]


def build_prediction_threshold_diagnostics(
    df: pl.DataFrame,
    *,
    symbol: str,
    split: str | int,
    config: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_col = "prediction" if "prediction" in df.columns else ("prediction_prob" if "prediction_prob" in df.columns else "")
    pred = df[pred_col].cast(pl.Float64).drop_nulls() if pred_col else pl.Series([], dtype=pl.Float64)
    current = _current_threshold(df, config)
    row = {
        "symbol": symbol,
        "split": split,
        "prediction_col": pred_col,
        "prediction_nonnull": int(pred.len()),
        "prediction_min": _float(pred.min()) if pred.len() else "",
        "prediction_max": _float(pred.max()) if pred.len() else "",
        "prediction_mean": _float(pred.mean()) if pred.len() else "",
        "prediction_std": _float(pred.std()) if pred.len() else "",
        "current_long_threshold": current,
        "current_short_threshold": -current if current != "" else "",
        "bars_above_current_long": 0,
        "bars_below_current_short": 0,
        "active_pct_at_current_threshold": 0.0,
    }
    for key, q in QUANTILES.items():
        row[key] = _float(pred.quantile(q)) if pred.len() else ""
    abs_pred = pred.abs()
    for key, q in ABS_QUANTILES.items():
        row[key] = _float(abs_pred.quantile(q)) if abs_pred.len() else ""
    if pred.len() and current != "":
        row["bars_above_current_long"] = int((pred > float(current)).sum() or 0)
        row["bars_below_current_short"] = int((pred < -float(current)).sum() or 0)
        row["active_pct_at_current_threshold"] = (
            row["bars_above_current_long"] + row["bars_below_current_short"]
        ) / pred.len()

    grid = []
    thresholds: list[tuple[str, float]] = []
    for key in ABS_QUANTILES:
        val = row.get(key)
        thresholds.append((key.replace("abs_prediction_", ""), float(val) if val != "" else 0.0))
    thresholds.extend((f"fixed_{v:g}", float(v)) for v in FIXED_THRESHOLDS)
    for typ, value in thresholds:
        grid.append(_candidate_row(pred, symbol=symbol, split=split, threshold_type=typ, threshold=value, current=current))
    return row, grid


def write_prediction_threshold_diagnostics(
    df: pl.DataFrame,
    *,
    symbol: str,
    split: str | int,
    config: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row, grid = build_prediction_threshold_diagnostics(df, symbol=symbol, split=split, config=config)
    meta = _metadata()
    _append_csv_json(DIAG_CSV, DIAG_JSON, DIAG_FIELDS, [{**meta, **row}], key_fields=["run_id", "symbol", "split"])
    _append_csv_json(
        GRID_CSV,
        GRID_JSON,
        GRID_FIELDS,
        [{**meta, **r} for r in grid],
        key_fields=["run_id", "symbol", "split", "threshold_type", "threshold_value"],
    )
    econ = build_threshold_candidate_economics(df, grid, symbol=symbol, split=split, config=config)
    _append_csv_json(
        ECON_CSV,
        ECON_JSON,
        ECON_FIELDS,
        [{**meta, **r} for r in econ],
        key_fields=["run_id", "symbol", "split", "threshold_type", "threshold_value"],
    )
    return row, grid


def build_threshold_candidate_economics(
    df: pl.DataFrame,
    grid: list[dict[str, Any]],
    *,
    symbol: str,
    split: str | int,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    """Diagnostic-only economics for candidate thresholds; does not affect live signals."""
    if config is None:
        config = _minimal_config()
    pred_col = "prediction" if "prediction" in df.columns else ("prediction_prob" if "prediction_prob" in df.columns else "")
    target_col = getattr(getattr(config, "walkforward", object()), "walkforward_target", "target_15m_ret")
    required = {pred_col, target_col}
    if not pred_col or not required.issubset(set(df.columns)):
        return [_empty_econ_row(r, symbol=symbol, split=split) for r in grid]

    drop_cols = [
        "raw_signal", "position", "position_after", "position_before", "position_delta",
        "assumed_fill_price", "min_position_hold_bars", "ret_exec", "target_exec",
        "target_return_exec", "target_exec_usd_per_contract", "gross_pnl", "fees",
        "slippage", "costs", "pnl", "net_pnl", "equity_curve", "drawdown_pct",
        "signal_entry_threshold",
    ]
    base = df.drop([c for c in drop_cols if c in df.columns])
    rows = []
    for candidate in grid:
        threshold = float(candidate.get("threshold_value") or 0.0)
        try:
            simulated = base.with_columns(
                pl.when(pl.col(pred_col).cast(pl.Float64) > threshold).then(1)
                .when(pl.col(pred_col).cast(pl.Float64) < -threshold).then(-1)
                .otherwise(0).alias("raw_signal"),
                pl.lit(threshold).alias("signal_entry_threshold"),
            )
            simulated = attach_execution_cost_model(
                simulated,
                target_col=target_col,
                config=config,
                symbol=symbol,
                feature_set_id="threshold_candidate_diagnostic",
            )
            metrics = compute_backtest_metrics(simulated)
            gross = float(simulated["gross_pnl"].sum() or 0.0) if "gross_pnl" in simulated.columns else 0.0
            net = float(simulated["pnl"].sum() or 0.0) if "pnl" in simulated.columns else 0.0
            turnover = float(simulated["position_delta"].abs().sum() or 0.0) if "position_delta" in simulated.columns else 0.0
            cost_drag = gross - net
            rows.append({
                "symbol": symbol,
                "split": split,
                "threshold_type": candidate.get("threshold_type", ""),
                "threshold_value": threshold,
                "threshold_source": "test_distribution_diagnostic_only",
                "long_bars": candidate.get("long_bars", 0),
                "short_bars": candidate.get("short_bars", 0),
                "active_bar_pct": candidate.get("active_bar_pct", 0.0),
                "turnover": turnover,
                "gross_pnl": gross,
                "net_pnl": net,
                "cost_drag": cost_drag,
                "cost_drag_pct_of_gross": _safe_div(cost_drag, gross),
                "pnl_per_turnover": _safe_div(net, turnover),
                "sharpe_annualized": metrics.get("sharpe_annualized", metrics.get("sharpe", 0.0)),
                "current_threshold_flag": candidate.get("current_threshold_flag", False),
            })
        except Exception:
            rows.append(_empty_econ_row(candidate, symbol=symbol, split=split))
    return rows


def print_threshold_diagnostic_summary(expected_splits: int | None = None, expected_run_id: str | None = None, allow_env_fallback: bool = False) -> None:
    run_id = _resolve_expected_run_id(expected_run_id, allow_env_fallback=allow_env_fallback)
    rows = [r for r in _read_json_list(DIAG_JSON) if str(r.get("run_id", "manual")) == run_id]
    grid = [r for r in _read_json_list(GRID_JSON) if str(r.get("run_id", "manual")) == run_id]
    denom = expected_splits or len(rows)
    active_rows = [r for r in rows if float(r.get("active_pct_at_current_threshold") or 0.0) > 0.0]
    if expected_splits is not None and len(active_rows) > expected_splits:
        raise RuntimeError(
            f"THRESHOLD DIAG INTEGRITY FAIL: active_splits={len(active_rows)} expected_rows={expected_splits}"
        )
    current_pcts = [float(r.get("active_pct_at_current_threshold") or 0.0) for r in rows]
    print(
        f"[THRESHOLD DIAG] current threshold active splits={len(active_rows)}/{denom} "
        f"active_bar_pct_median={_median(current_pcts):.6g}",
        flush=True,
    )
    for typ in ["p99", "p995"]:
        vals = [float(r.get("active_bar_pct") or 0.0) for r in grid if r.get("threshold_type") == typ]
        print(
            f"[THRESHOLD DIAG] candidate threshold {typ} active_bar_pct_median={_median(vals):.6g}",
            flush=True,
        )


def validate_current_run_diagnostics(
    expected_rows: int,
    candidate_threshold_count: int = CANDIDATE_THRESHOLD_COUNT,
    require_threshold_used: bool = True,
    expected_run_id: str | None = None,
    allow_env_fallback: bool = False,
) -> dict[str, Any]:
    run_id = _resolve_expected_run_id(expected_run_id, allow_env_fallback=allow_env_fallback)
    threshold_used = [r for r in _read_json_list(Path("reports/validation/threshold_used.json")) if str(r.get("run_id", "manual")) == run_id]
    signal = [r for r in _read_json_list(Path("reports/validation/signal_activation_debug.json")) if str(r.get("run_id", "manual")) == run_id]
    grid = [r for r in _read_json_list(GRID_JSON) if str(r.get("run_id", "manual")) == run_id]
    if require_threshold_used:
        _assert_count_unique(
            "threshold_used",
            threshold_used,
            expected_rows,
            ["run_id", "symbol", "split"],
            Path("reports/validation/threshold_used.json"),
            run_id,
        )
    _assert_count_unique(
        "signal_activation_debug",
        signal,
        expected_rows,
        ["run_id", "symbol", "split"],
        Path("reports/validation/signal_activation_debug.json"),
        run_id,
    )
    _assert_count_unique(
        "threshold_candidate_grid",
        grid,
        expected_rows * candidate_threshold_count,
        ["run_id", "symbol", "split", "threshold_type", "threshold_value"],
        GRID_JSON,
        run_id,
    )
    active_rows = [r for r in _read_json_list(DIAG_JSON) if str(r.get("run_id", "manual")) == run_id and float(r.get("active_pct_at_current_threshold") or 0.0) > 0.0]
    if len(active_rows) > expected_rows:
        raise RuntimeError(
            "THRESHOLD DIAG INTEGRITY FAIL: "
            f"active_splits={len(active_rows)} expected_rows={expected_rows}; "
            f"{_report_details(DIAG_JSON, run_id)}"
        )
    return {"status": "PASS", "run_id": run_id, "expected_rows": expected_rows}


def _candidate_row(pred: pl.Series, *, symbol: str, split: str | int, threshold_type: str, threshold: float, current: float | str) -> dict[str, Any]:
    if not pred.len():
        long_bars = short_bars = 0
        active_pct = turnover = 0.0
    else:
        desired = [1 if x > threshold else (-1 if x < -threshold else 0) for x in pred.to_list()]
        long_bars = sum(1 for x in desired if x > 0)
        short_bars = sum(1 for x in desired if x < 0)
        active_pct = (long_bars + short_bars) / len(desired)
        prev = 0
        turnover = 0.0
        for x in desired:
            turnover += abs(x - prev)
            prev = x
    return {
        "symbol": symbol,
        "split": split,
        "threshold_type": threshold_type,
        "threshold_value": threshold,
        "long_bars": long_bars,
        "short_bars": short_bars,
        "active_bar_pct": active_pct,
        "turnover_proxy": turnover,
        "current_threshold_flag": bool(current != "" and abs(float(current) - threshold) < 1e-15),
    }


def _current_threshold(df: pl.DataFrame, config: Any | None) -> float | str:
    if "signal_entry_threshold" in df.columns:
        vals = df["signal_entry_threshold"].drop_nulls()
        if vals.len():
            return _float(vals[0])
    if config is not None:
        try:
            return float(config.execution.prediction_entry_threshold)
        except Exception:
            return ""
    return ""


def _append_csv_json(csv_path: Path, json_path: Path, fields: list[str], rows: list[dict[str, Any]], key_fields: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = _metadata()["run_id"]
    rows = [stringify_diagnostic_keys(r) for r in rows]
    existing = _read_json_list(json_path)
    existing = [r for r in existing if str(r.get("run_id", "manual")) == run_id]
    new_keys = {tuple(str(r.get(k, "")) for k in key_fields) for r in rows}
    existing = [r for r in existing if tuple(str(r.get(k, "")) for k in key_fields) not in new_keys]
    existing.extend(stringify_diagnostic_keys({k: r.get(k, "") for k in fields}) for r in rows)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
    json_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _median(vals: list[float]) -> float:
    return float(median(vals)) if vals else 0.0


def _metadata() -> dict[str, str]:
    config_env = os.environ.get("CONFIG_ENV") or os.environ.get("QUANT_ENV") or ""
    return {
        "run_id": os.environ.get("PARENT_RUN_ID") or os.environ.get("QUANT_RUN_ID") or "manual",
        "profile": os.environ.get("QUANT_RUN_PROFILE") or config_env,
        "config_env": config_env,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _resolve_expected_run_id(expected_run_id: str | None, *, allow_env_fallback: bool) -> str:
    if expected_run_id:
        return str(expected_run_id)
    if not allow_env_fallback:
        raise RuntimeError("THRESHOLD DIAG INTEGRITY FAIL: expected_run_id is required")
    return _metadata()["run_id"]


def _assert_count_unique(name: str, rows: list[dict[str, Any]], expected: int, key_fields: list[str], path: Path, run_id: str) -> None:
    if len(rows) != expected:
        raise RuntimeError(
            "THRESHOLD DIAG INTEGRITY FAIL: "
            f"{name} rows={len(rows)} expected={expected}; {_report_details(path, run_id)}"
        )
    keys = [tuple(str(r.get(k, "")) for k in key_fields) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "THRESHOLD DIAG INTEGRITY FAIL: "
            f"duplicate {name} keys; {_report_details(path, run_id)}"
        )


def _report_details(path: Path, expected_run_id: str) -> str:
    all_rows = _read_json_list(path)
    return (
        f"expected_run_id={expected_run_id} report_path={path} total_rows={len(all_rows)} "
        f"unique_run_ids={sorted({str(r.get('run_id', '')) for r in all_rows})} "
        f"unique_profiles={sorted({str(r.get('profile', '')) for r in all_rows})} "
        f"first_5_rows={all_rows[:5]}"
    )


def _float(value: Any) -> float | str:
    try:
        if value is None:
            return ""
        return float(value)
    except Exception:
        return ""


def _safe_div(num: float, den: float) -> float:
    return 0.0 if abs(float(den or 0.0)) < 1e-12 else float(num) / float(den)


def _minimal_config() -> RootConfig:
    return RootConfig()


def _empty_econ_row(candidate: dict[str, Any], *, symbol: str, split: str | int) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "split": split,
        "threshold_type": candidate.get("threshold_type", ""),
        "threshold_value": candidate.get("threshold_value", 0.0),
        "threshold_source": "unavailable",
        "long_bars": candidate.get("long_bars", 0),
        "short_bars": candidate.get("short_bars", 0),
        "active_bar_pct": candidate.get("active_bar_pct", 0.0),
        "turnover": 0.0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "cost_drag": 0.0,
        "cost_drag_pct_of_gross": 0.0,
        "pnl_per_turnover": 0.0,
        "sharpe_annualized": 0.0,
        "current_threshold_flag": candidate.get("current_threshold_flag", False),
    }
