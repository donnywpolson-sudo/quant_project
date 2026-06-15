#!/usr/bin/env python3
"""Audit mean-reversion tail risk and trend-danger fade suppression."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.phase8_model_selection.evaluate_predictions import (
    DEFAULT_COSTS_CONFIG,
    PREDICTION_KEY_CANDIDATES,
    PolicyConfig,
    build_policy_frame,
)


DEFAULT_RUN = "tier1_locked_baseline"
DEFAULT_PREDICTIONS = Path("data/predictions/tier1_locked_baseline/oos_predictions.parquet")
DEFAULT_OUTPUT_ROOT = Path("reports/phase8_mr_tail_audit")

OUTPUT_SUFFIXES = {
    "summary": "summary.json",
    "bucket_summary": "bucket_summary.csv",
    "policy_comparison": "policy_comparison.csv",
    "worst_sessions": "worst_sessions.csv",
    "losing_streaks": "losing_streaks.csv",
    "blocked_trend": "blocked_trend_danger_opportunities.csv",
    "readme": "README.md",
}

OPTIONAL_VALUE_HINTS = (
    "mae",
    "mfe",
    "atr",
    "volatility",
    "vol_",
    "_vol",
    "regime",
)

VOLATILITY_REGIME_COLUMNS = (
    "realized_volatility_regime",
    "volatility_regime",
    "regime",
    "market_regime",
)

VOLATILITY_VALUE_COLUMNS = (
    "realized_volatility_15m",
    "realized_volatility",
    "volatility_15m",
    "atr_15m",
    "atr",
)


@dataclass(frozen=True)
class MRTailPolicyConfig:
    long_short_margin: float = 0.05
    trend_danger_cutoff: float = 0.50
    fade_success_cutoff: float = 0.50
    edge_buffer_dollars: float = 0.0
    momentum_escape_enabled: bool = False
    momentum_escape_min_edge_dollars: float = 0.0


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _sum_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _cvar(series: pd.Series, confidence: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    threshold = values.quantile(1.0 - confidence)
    tail = values[values.le(threshold)]
    return float(tail.mean()) if not tail.empty else None


def _max_losing_streak(values: pd.Series) -> int:
    max_streak = 0
    current = 0
    for value in pd.to_numeric(values, errors="coerce").fillna(0.0):
        if float(value) < 0.0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _output_paths(output_root: Path, run: str) -> dict[str, Path]:
    return {key: output_root / f"{run}_{suffix}" for key, suffix in OUTPUT_SUFFIXES.items()}


def _key_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in PREDICTION_KEY_CANDIDATES if column in frame.columns]


def _optional_columns(predictions: pd.DataFrame, policy_frame: pd.DataFrame) -> list[str]:
    keys = set(_key_columns(predictions))
    existing = set(policy_frame.columns)
    columns: list[str] = []
    for column in predictions.columns:
        lower = column.lower()
        if column in keys or column in existing:
            continue
        if any(hint in lower for hint in OPTIONAL_VALUE_HINTS):
            columns.append(column)
    return columns


def _merge_optional_values(predictions: pd.DataFrame, policy_frame: pd.DataFrame) -> pd.DataFrame:
    key_cols = [column for column in _key_columns(predictions) if column in policy_frame.columns]
    value_cols = _optional_columns(predictions, policy_frame)
    if not key_cols or not value_cols:
        return policy_frame
    optional = predictions[key_cols + value_cols].groupby(key_cols, dropna=False, as_index=False).first()
    return policy_frame.merge(optional, on=key_cols, how="left")


def _attach_probability_decile(frame: pd.DataFrame, source: str, output: str) -> pd.DataFrame:
    values = pd.to_numeric(frame[source], errors="coerce")
    ranks = values.rank(method="first", pct=True)
    deciles = np.ceil(ranks * 10.0).clip(1, 10)
    labels = deciles.astype("Int64").astype(str).str.zfill(2)
    out = frame.copy()
    out[output] = "missing"
    out.loc[values.notna(), output] = "d" + labels.loc[values.notna()]
    return out


def _select_volatility_regime_source(frame: pd.DataFrame) -> tuple[str | None, bool]:
    for column in VOLATILITY_REGIME_COLUMNS:
        if column in frame.columns and frame[column].notna().any():
            return column, False
    for column in VOLATILITY_VALUE_COLUMNS:
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column, True
    for column in frame.columns:
        lower = column.lower()
        values = pd.to_numeric(frame[column], errors="coerce")
        if ("volatility" in lower or "atr" in lower or "vol_" in lower) and values.notna().any():
            return column, True
    return None, False


def _attach_volatility_regime(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    source, numeric_source = _select_volatility_regime_source(frame)
    out = frame.copy()
    if source is None:
        out["realized_volatility_regime"] = "missing"
        return out, None
    if not numeric_source:
        out["realized_volatility_regime"] = out[source].fillna("missing").astype(str)
        return out, source

    values = pd.to_numeric(out[source], errors="coerce")
    out["realized_volatility_regime"] = "missing"
    if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
        out.loc[values.notna(), "realized_volatility_regime"] = "single"
        return out, source
    try:
        buckets = pd.qcut(
            values,
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        out.loc[values.notna(), "realized_volatility_regime"] = "single"
        return out, source
    out.loc[values.notna(), "realized_volatility_regime"] = buckets.loc[values.notna()]
    return out, source


def _attach_time_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    timestamp = pd.to_datetime(out.get("timestamp"), errors="coerce", utc=True)
    out["session_hour_utc"] = timestamp.dt.hour.astype("Int64").astype(str)
    out.loc[timestamp.isna(), "session_hour_utc"] = "missing"
    if "session_id" in out.columns:
        session = out["session_id"].fillna(timestamp.dt.strftime("%Y-%m-%d")).astype(str)
    else:
        session = timestamp.dt.strftime("%Y-%m-%d").fillna("missing").astype(str)
    out["audit_session_id"] = out["market"].astype(str) + "|" + session
    return out


def _edge_dollars(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["base_position"], errors="coerce").fillna(0.0)
        * pd.to_numeric(frame["expected_return"], errors="coerce").fillna(0.0)
        * pd.to_numeric(frame["execution_open"], errors="coerce").fillna(0.0)
        * pd.to_numeric(frame["point_value"], errors="coerce").fillna(0.0)
    )


def _apply_position_economics(frame: pd.DataFrame, position_col: str, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    position = pd.to_numeric(out[position_col], errors="coerce").fillna(0).astype(int)
    out[f"{prefix}_position"] = position
    out[f"{prefix}_trade_count"] = position.ne(0).astype(int)
    out[f"{prefix}_gross_dollars"] = (
        position
        * pd.to_numeric(out["price_move"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out["point_value"], errors="coerce").fillna(0.0)
    )
    round_turn = pd.to_numeric(out["round_turn_cost_dollars"], errors="coerce").fillna(0.0)
    tick_value = pd.to_numeric(out.get("tick_value"), errors="coerce").fillna(0.0)
    slippage_ticks = pd.to_numeric(out.get("slippage_ticks_per_side"), errors="coerce").fillna(0.0)
    slippage = 2.0 * slippage_ticks * tick_value
    out[f"{prefix}_slippage_cost_dollars"] = np.where(position.ne(0), slippage, 0.0)
    out[f"{prefix}_cost_dollars"] = np.where(position.ne(0), round_turn, 0.0)
    out[f"{prefix}_commission_cost_dollars"] = np.maximum(
        out[f"{prefix}_cost_dollars"] - out[f"{prefix}_slippage_cost_dollars"],
        0.0,
    )
    out[f"{prefix}_net_dollars"] = out[f"{prefix}_gross_dollars"] - out[f"{prefix}_cost_dollars"]
    sort_cols = [column for column in ("market", "fold_id", "timestamp") if column in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    group_cols = [column for column in ("market", "fold_id") if column in out.columns]
    previous = out.groupby(group_cols, dropna=False)[f"{prefix}_position"].shift(1).fillna(0)
    out[f"{prefix}_position_change_abs"] = (out[f"{prefix}_position"] - previous).abs()
    return out


def _copy_overlay_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mappings = {
        "overlay_position": "position",
        "overlay_trade_count": "trade_count",
        "overlay_gross_dollars": "gross_dollars",
        "overlay_slippage_cost_dollars": "slippage_cost_dollars",
        "overlay_commission_cost_dollars": "commission_cost_dollars",
        "overlay_cost_dollars": "cost_dollars",
        "overlay_net_dollars": "net_dollars",
        "overlay_position_change_abs": "position_change_abs",
    }
    for source, target in mappings.items():
        out[target] = out[source]
    return out


def _attach_mae_mfe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"mae_ticks_15m", "mfe_ticks_15m"}.issubset(out.columns):
        mae = pd.to_numeric(out["mae_ticks_15m"], errors="coerce")
        mfe = pd.to_numeric(out["mfe_ticks_15m"], errors="coerce")
        position = pd.to_numeric(out["position"], errors="coerce").fillna(0)
        out["trade_mae_ticks"] = np.where(position.gt(0), mae, np.where(position.lt(0), -mfe, np.nan))
        out["trade_mfe_ticks"] = np.where(position.gt(0), mfe, np.where(position.lt(0), -mae, np.nan))
        tick_value = pd.to_numeric(out.get("tick_value"), errors="coerce")
        out["trade_mae_dollars"] = out["trade_mae_ticks"] * tick_value
        out["trade_mfe_dollars"] = out["trade_mfe_ticks"] * tick_value
    return out


def build_mr_tail_policy_frame(
    predictions: pd.DataFrame,
    costs_config: Path,
    config: MRTailPolicyConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    policy_frame, failures, warnings = build_policy_frame(
        predictions,
        costs_config,
        PolicyConfig(
            long_short_margin=config.long_short_margin,
            min_fade_success=config.fade_success_cutoff,
            max_trend_danger=config.trend_danger_cutoff,
        ),
    )
    if failures:
        return policy_frame, failures, warnings

    frame = _merge_optional_values(predictions, policy_frame)
    frame = _attach_probability_decile(frame, "p_trend_danger", "trend_danger_decile")
    frame = _attach_probability_decile(frame, "p_fade_success", "fade_success_decile")
    frame, volatility_source = _attach_volatility_regime(frame)
    frame = _attach_time_fields(frame)

    frame["expected_edge_dollars"] = _edge_dollars(frame)
    frame["cost_plus_buffer_dollars"] = (
        pd.to_numeric(frame["round_turn_cost_dollars"], errors="coerce").fillna(np.inf)
        + config.edge_buffer_dollars
    )
    frame["edge_exceeds_cost_buffer"] = frame["expected_edge_dollars"].gt(
        frame["cost_plus_buffer_dollars"]
    )
    frame["fade_success_gate"] = pd.to_numeric(
        frame["p_fade_success"], errors="coerce"
    ).ge(config.fade_success_cutoff)
    trend_probability = pd.to_numeric(frame["p_trend_danger"], errors="coerce")
    frame["trend_danger_block"] = trend_probability.isna() | trend_probability.ge(
        config.trend_danger_cutoff
    )
    frame["candidate_fade_trade"] = frame["base_position"].ne(0).astype(bool)
    frame["no_trend_position"] = np.where(
        frame["candidate_fade_trade"] & frame["fade_success_gate"] & frame["edge_exceeds_cost_buffer"],
        frame["base_position"],
        0,
    ).astype(int)
    frame["blocked_by_trend_danger"] = frame["no_trend_position"].ne(0) & frame["trend_danger_block"]
    frame["overlay_position"] = np.where(
        frame["no_trend_position"].ne(0) & ~frame["trend_danger_block"],
        frame["no_trend_position"],
        0,
    ).astype(int)

    if config.momentum_escape_enabled:
        escape_gate = (
            frame["blocked_by_trend_danger"]
            & frame["expected_edge_dollars"].gt(
                frame["cost_plus_buffer_dollars"] + config.momentum_escape_min_edge_dollars
            )
        )
        frame.loc[escape_gate, "overlay_position"] = frame.loc[escape_gate, "base_position"].astype(int)
        frame["momentum_escape_trade"] = escape_gate
    else:
        frame["momentum_escape_trade"] = False

    frame["policy_reason"] = "trade"
    frame.loc[frame["base_position"].eq(0), "policy_reason"] = "no_direction_edge"
    frame.loc[
        frame["base_position"].ne(0) & ~frame["fade_success_gate"],
        "policy_reason",
    ] = "fade_success_below_cutoff"
    frame.loc[
        frame["base_position"].ne(0)
        & frame["fade_success_gate"]
        & ~frame["edge_exceeds_cost_buffer"],
        "policy_reason",
    ] = "edge_below_cost_plus_buffer"
    frame.loc[frame["blocked_by_trend_danger"], "policy_reason"] = "trend_danger_block"
    frame.loc[frame["momentum_escape_trade"], "policy_reason"] = "momentum_escape"

    frame = _apply_position_economics(frame, "no_trend_position", "no_trend")
    frame = _apply_position_economics(frame, "overlay_position", "overlay")
    frame = _copy_overlay_columns(frame)
    frame = _attach_mae_mfe(frame)
    warnings.append(
        "expected_return is interpreted as fractional target_ret_15m; edge gate uses "
        "base_position * expected_return * execution_open * point_value"
    )
    if volatility_source is None:
        warnings.append("realized volatility regime unavailable; bucket set to missing")
    else:
        warnings.append(f"realized volatility regime source: {volatility_source}")
    return frame, [], warnings


def _metrics(frame: pd.DataFrame, scope: str, keys: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(len(frame))
    traded = frame[frame["trade_count"].eq(1)] if "trade_count" in frame else frame.iloc[0:0]
    trades = int(len(traded))
    gross = _sum_float(frame["gross_dollars"]) if "gross_dollars" in frame else 0.0
    slippage = _sum_float(frame["slippage_cost_dollars"]) if "slippage_cost_dollars" in frame else 0.0
    commission = _sum_float(frame["commission_cost_dollars"]) if "commission_cost_dollars" in frame else 0.0
    cost = _sum_float(frame["cost_dollars"]) if "cost_dollars" in frame else 0.0
    net = _sum_float(frame["net_dollars"]) if "net_dollars" in frame else 0.0
    trade_net = traded["net_dollars"] if trades else pd.Series(dtype=float)
    record: dict[str, Any] = {
        "scope": scope,
        **dict(keys),
        "row_count": rows,
        "candidate_fade_trades": int(frame["candidate_fade_trade"].sum())
        if "candidate_fade_trade" in frame
        else 0,
        "trade_count": trades,
        "blocked_by_trend_danger": int(frame["blocked_by_trend_danger"].sum())
        if "blocked_by_trend_danger" in frame
        else 0,
        "momentum_escape_trades": int(frame["momentum_escape_trade"].sum())
        if "momentum_escape_trade" in frame
        else 0,
        "gross_return_dollars": gross,
        "slippage_cost_dollars": slippage,
        "commission_cost_dollars": commission,
        "cost_dollars": cost,
        "net_return_dollars": net,
        "turnover": float(frame["position_change_abs"].sum()) if "position_change_abs" in frame else None,
        "turnover_per_row": float(frame["position_change_abs"].sum()) / rows
        if rows and "position_change_abs" in frame
        else None,
        "avg_cost_per_trade": cost / trades if trades else None,
        "mean_net_per_trade": _mean_or_none(trade_net) if trades else None,
        "net_cvar_95_dollars": _cvar(trade_net, 0.95) if trades else None,
        "net_cvar_99_dollars": _cvar(trade_net, 0.99) if trades else None,
        "max_losing_streak": _max_losing_streak(trade_net) if trades else 0,
    }
    for column in ("trade_mae_ticks", "trade_mfe_ticks", "trade_mae_dollars", "trade_mfe_dollars"):
        if column in traded:
            record[f"mean_{column}"] = _mean_or_none(traded[column])
    return record


def _group_metrics(frame: pd.DataFrame, scope: str, group_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        records.append(_metrics(group, scope, dict(zip(group_cols, keys))))
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(group_cols).reset_index(drop=True)


def _prefixed_scenario(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    for suffix, target in (
        ("position", "position"),
        ("trade_count", "trade_count"),
        ("gross_dollars", "gross_dollars"),
        ("slippage_cost_dollars", "slippage_cost_dollars"),
        ("commission_cost_dollars", "commission_cost_dollars"),
        ("cost_dollars", "cost_dollars"),
        ("net_dollars", "net_dollars"),
        ("position_change_abs", "position_change_abs"),
    ):
        out[target] = out[f"{prefix}_{suffix}"]
    return _attach_mae_mfe(out)


def _policy_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        ("no_trend_danger_block", _prefixed_scenario(frame, "no_trend")),
        ("mr_tail_overlay", _prefixed_scenario(frame, "overlay")),
    ]
    records = []
    for scenario, scenario_frame in scenarios:
        record = _metrics(scenario_frame, "policy_scenario", {"scenario": scenario})
        records.append(record)
    return pd.DataFrame(records)


def _worst_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = _group_metrics(frame, "session", ["audit_session_id"])
    if grouped.empty:
        return grouped
    return grouped.sort_values("net_return_dollars", ascending=True).head(20).reset_index(drop=True)


def _losing_streaks(frame: pd.DataFrame) -> pd.DataFrame:
    records = [_metrics(frame, "overall", {})]
    for market, group in frame.groupby("market", dropna=False):
        records.append(_metrics(group, "market", {"market": market}))
    columns = ["scope", "market", "trade_count", "max_losing_streak", "net_return_dollars"]
    return pd.DataFrame(records).reindex(columns=columns)


def _blocked_trend_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    blocked = frame[frame["blocked_by_trend_danger"]].copy()
    if blocked.empty:
        return pd.DataFrame(
            [
                {
                    "unavailable_reason": "no trades changed by trend-danger block",
                }
            ]
        )
    opportunity = _prefixed_scenario(blocked, "no_trend")
    return _group_metrics(
        opportunity,
        "blocked_trend_danger",
        ["trend_danger_decile", "fade_success_decile", "market", "session_hour_utc"],
    )


def _write_readme(path: Path, summary: Mapping[str, Any], outputs: Mapping[str, str]) -> None:
    files = "\n".join(f"- `{value}`" for key, value in sorted(outputs.items()) if key != "readme")
    text = f"""# Phase 8 MR Tail Audit

Run: `{summary['run']}`

This report recomputes a mean-reversion policy overlay from saved WFA
predictions. It does not modify labels, raw data, features, models, or saved
predictions.

## Files

{files}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_mr_tail_audit(
    *,
    predictions_path: Path,
    costs_config: Path,
    output_root: Path,
    run: str,
    config: MRTailPolicyConfig,
) -> dict[str, Any]:
    if not predictions_path.exists():
        raise SystemExit(f"prediction parquet missing: {_relative_path(predictions_path)}")

    predictions = pd.read_parquet(predictions_path)
    policy_frame, failures, warnings = build_mr_tail_policy_frame(predictions, costs_config, config)
    if failures:
        raise SystemExit("; ".join(failures))

    bucket_cols = [
        "trend_danger_decile",
        "fade_success_decile",
        "realized_volatility_regime",
        "market",
        "session_hour_utc",
    ]
    bucket_summary = _group_metrics(policy_frame, "mr_tail_bucket", bucket_cols)
    policy_comparison = _policy_comparison(policy_frame)
    worst_sessions = _worst_sessions(policy_frame)
    losing_streaks = _losing_streaks(policy_frame)
    blocked_trend = _blocked_trend_opportunities(policy_frame)
    overall = _metrics(policy_frame, "overall", {})

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, run)
    _write_csv(paths["bucket_summary"], bucket_summary)
    _write_csv(paths["policy_comparison"], policy_comparison)
    _write_csv(paths["worst_sessions"], worst_sessions)
    _write_csv(paths["losing_streaks"], losing_streaks)
    _write_csv(paths["blocked_trend"], blocked_trend)

    comparison_by_scenario = policy_comparison.set_index("scenario").to_dict(orient="index")
    no_trend = comparison_by_scenario.get("no_trend_danger_block", {})
    overlay = comparison_by_scenario.get("mr_tail_overlay", {})
    trend_block_trade_delta = int(overlay.get("trade_count", 0) or 0) - int(
        no_trend.get("trade_count", 0) or 0
    )
    trend_block_net_delta = float(overlay.get("net_return_dollars", 0.0) or 0.0) - float(
        no_trend.get("net_return_dollars", 0.0) or 0.0
    )
    outputs = {key: _relative_path(path) for key, path in paths.items()}
    summary: dict[str, Any] = {
        "run": run,
        "report_version": 1,
        "prediction_path": _relative_path(predictions_path),
        "output_root": _relative_path(output_root),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(policy_frame)),
        "config": asdict(config),
        "overall": overall,
        "trend_block_trade_delta": trend_block_trade_delta,
        "trend_block_net_delta": trend_block_net_delta,
        "policy_comparison": policy_comparison.to_dict(orient="records"),
        "worst_sessions": worst_sessions.head(10).to_dict(orient="records"),
        "max_losing_streak": int(overall["max_losing_streak"]),
        "warnings": warnings,
        "outputs": outputs,
    }
    _write_json(paths["summary"], summary)
    _write_readme(paths["readme"], summary, outputs)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS.as_posix())
    parser.add_argument("--costs-config", default=DEFAULT_COSTS_CONFIG.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--long-short-margin", type=float, default=0.05)
    parser.add_argument("--trend-danger-cutoff", type=float, default=0.50)
    parser.add_argument("--fade-success-cutoff", type=float, default=0.50)
    parser.add_argument("--edge-buffer-dollars", type=float, default=0.0)
    parser.add_argument("--momentum-escape", action="store_true")
    parser.add_argument("--momentum-escape-min-edge-dollars", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_mr_tail_audit(
        predictions_path=Path(args.predictions),
        costs_config=Path(args.costs_config),
        output_root=Path(args.output_root),
        run=args.run,
        config=MRTailPolicyConfig(
            long_short_margin=args.long_short_margin,
            trend_danger_cutoff=args.trend_danger_cutoff,
            fade_success_cutoff=args.fade_success_cutoff,
            edge_buffer_dollars=args.edge_buffer_dollars,
            momentum_escape_enabled=args.momentum_escape,
            momentum_escape_min_edge_dollars=args.momentum_escape_min_edge_dollars,
        ),
    )
    overall = summary["overall"]
    print(
        "PASS mr tail audit: "
        f"predictions={summary['prediction_count']} "
        f"policy_rows={summary['policy_row_count']} "
        f"trades={overall['trade_count']} "
        f"net_dollars={overall['net_return_dollars']} "
        f"trend_block_net_delta={summary['trend_block_net_delta']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
