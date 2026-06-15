#!/usr/bin/env python3
"""Trace Phase 8 predictions through policy rows, trades, PnL, and costs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from scripts.phase8_model_selection.audit_policy_failure import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREDICTIONS,
    DEFAULT_RUN,
    POSITION_LABELS,
    _relative_path,
    _write_csv,
    _write_json,
)
from scripts.phase8_model_selection.evaluate_predictions import (
    DEFAULT_COSTS_CONFIG,
    PREDICTION_KEY_CANDIDATES,
    PolicyConfig,
    build_policy_frame,
)


OUTPUT_SUFFIXES = {
    "summary": "trade_drilldown_summary.json",
    "market_side": "pnl_by_market_side.csv",
    "market_side_hour": "pnl_by_market_side_hour.csv",
    "fold": "pnl_by_fold.csv",
    "confidence": "pnl_by_confidence_bucket.csv",
    "regime": "pnl_by_regime_bucket.csv",
    "volatility": "pnl_by_volatility_bucket.csv",
    "counterfactuals": "cost_counterfactuals.csv",
    "blocked": "blocked_vs_traded_opportunity.csv",
    "long_failure": "long_failure_breakdown.csv",
    "readme": "failure_readme.md",
}

OPTIONAL_DIAGNOSTIC_HINTS = (
    "confidence",
    "probability",
    "regime",
    "atr",
    "volatility",
    "vol_",
    "feature_",
)

REGIME_COLUMN_CANDIDATES = (
    "regime",
    "market_regime",
    "volatility_regime",
    "feature_regime",
    "session_segment_id",
    "policy_reason",
)


def _sum_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _median_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _win_rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.gt(0.0).mean()) if not values.empty else None


def _empty_unavailable(reason: str) -> pd.DataFrame:
    return pd.DataFrame([{"unavailable_reason": reason}])


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
        if any(hint in lower for hint in OPTIONAL_DIAGNOSTIC_HINTS):
            columns.append(column)
    return columns


def _first_optional_values(
    predictions: pd.DataFrame,
    policy_frame: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = [column for column in _key_columns(predictions) if column in policy_frame.columns]
    value_cols = _optional_columns(predictions, policy_frame)
    if not key_cols or not value_cols:
        return policy_frame
    optional = predictions[key_cols + value_cols].groupby(key_cols, dropna=False, as_index=False).first()
    return policy_frame.merge(optional, on=key_cols, how="left")


def _metrics(frame: pd.DataFrame, scope: str, keys: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(len(frame))
    traded = frame[frame["trade_count"].eq(1)] if "trade_count" in frame else frame.iloc[0:0]
    trade_count = int(len(traded))
    gross = _sum_float(frame["gross_dollars"]) if "gross_dollars" in frame else 0.0
    slippage = _sum_float(frame["slippage_cost_dollars"]) if "slippage_cost_dollars" in frame else 0.0
    commission = _sum_float(frame["commission_cost_dollars"]) if "commission_cost_dollars" in frame else 0.0
    cost = _sum_float(frame["cost_dollars"]) if "cost_dollars" in frame else 0.0
    net = _sum_float(frame["net_dollars"]) if "net_dollars" in frame else 0.0
    return {
        "scope": scope,
        **dict(keys),
        "row_count": rows,
        "trade_count": trade_count,
        "block_count": rows - trade_count,
        "long_count": int(frame["position"].eq(1).sum()) if "position" in frame else 0,
        "short_count": int(frame["position"].eq(-1).sum()) if "position" in frame else 0,
        "flat_count": int(frame["position"].eq(0).sum()) if "position" in frame else rows,
        "gross_return_dollars": gross,
        "slippage_cost_dollars": slippage,
        "commission_cost_dollars": commission,
        "cost_dollars": cost,
        "net_return_dollars": net,
        "mean_gross_per_trade": _mean_or_none(traded["gross_dollars"]) if trade_count else None,
        "median_gross_per_trade": _median_or_none(traded["gross_dollars"]) if trade_count else None,
        "mean_net_per_trade": _mean_or_none(traded["net_dollars"]) if trade_count else None,
        "median_net_per_trade": _median_or_none(traded["net_dollars"]) if trade_count else None,
        "gross_win_rate": _win_rate(traded["gross_dollars"]) if trade_count else None,
        "net_win_rate": _win_rate(traded["net_dollars"]) if trade_count else None,
        "avg_slippage_per_trade": slippage / trade_count if trade_count else None,
        "avg_commission_per_trade": commission / trade_count if trade_count else None,
        "avg_cost_per_trade": cost / trade_count if trade_count else None,
        "cost_drag_to_abs_gross": cost / abs(gross) if abs(gross) > 0.0 else None,
    }


def _group_metrics(
    frame: pd.DataFrame,
    *,
    scope: str,
    group_cols: list[str],
    sort_by_net: bool = True,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        records.append(_metrics(group, scope, dict(zip(group_cols, keys))))
    out = pd.DataFrame(records)
    if out.empty or not sort_by_net:
        return out
    return out.sort_values("net_return_dollars", ascending=True).reset_index(drop=True)


def _attach_side(policy_frame: pd.DataFrame) -> pd.DataFrame:
    frame = policy_frame.copy()
    frame["side"] = frame["position"].map(POSITION_LABELS).fillna("unknown")
    return frame


def _attach_hour(policy_frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if "timestamp" not in policy_frame.columns:
        return policy_frame, "timestamp column missing"
    frame = policy_frame.copy()
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamp.isna().all():
        return policy_frame, "timestamp column could not be parsed"
    frame["hour_utc"] = timestamp.dt.hour
    return frame, None


def _attach_confidence_bucket(policy_frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if "direction_margin" not in policy_frame.columns:
        return policy_frame, "direction_margin column missing"
    frame = policy_frame.copy()
    confidence = pd.to_numeric(frame["direction_margin"], errors="coerce").abs()
    if confidence.isna().all():
        return policy_frame, "direction_margin has no numeric values"
    frame["confidence_bucket"] = pd.cut(
        confidence,
        bins=[-np.inf, 0.05, 0.10, 0.20, 0.40, np.inf],
        labels=["<=0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">0.40"],
    ).astype(str)
    return frame, None


def _select_regime_column(policy_frame: pd.DataFrame) -> str | None:
    for column in REGIME_COLUMN_CANDIDATES:
        if column in policy_frame.columns:
            return column
    for column in policy_frame.columns:
        if "regime" in column.lower():
            return column
    return None


def _attach_regime_bucket(policy_frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    column = _select_regime_column(policy_frame)
    if column is None:
        return policy_frame, None, "no regime or policy bucket column available"
    frame = policy_frame.copy()
    frame["regime_bucket"] = frame[column].fillna("missing").astype(str)
    if "policy_reason" in frame.columns and column != "policy_reason":
        frame["regime_bucket"] = frame["regime_bucket"] + "|" + frame["policy_reason"].astype(str)
    return frame, column, None


def _select_volatility_column(policy_frame: pd.DataFrame) -> str | None:
    candidates: list[str] = []
    for column in policy_frame.columns:
        lower = column.lower()
        if "atr" in lower or "volatility" in lower or lower.endswith("_vol") or "vol_" in lower:
            candidates.append(column)
    for column in candidates:
        values = pd.to_numeric(policy_frame[column], errors="coerce")
        if values.notna().any():
            return column
    return None


def _attach_quantile_bucket(
    policy_frame: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
) -> tuple[pd.DataFrame, str | None]:
    values = pd.to_numeric(policy_frame[source_column], errors="coerce")
    if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
        return policy_frame, f"{source_column} has insufficient numeric variation"
    frame = policy_frame.copy()
    try:
        frame[output_column] = pd.qcut(
            values,
            q=4,
            labels=["q1_low", "q2", "q3", "q4_high"],
            duplicates="drop",
        ).astype(str)
    except ValueError as exc:
        return policy_frame, f"{source_column} could not be bucketed: {exc}"
    frame.loc[values.isna(), output_column] = "missing"
    return frame, None


def _blocked_category_frame(policy_frame: pd.DataFrame) -> pd.DataFrame:
    frame = policy_frame.copy()
    frame["blocked_category"] = "other"
    frame.loc[frame["position"].ne(0), "blocked_category"] = "traded"
    if "no_direction_signal" in frame.columns:
        frame.loc[frame["position"].eq(0) & frame["no_direction_signal"], "blocked_category"] = "no-direction"
    if "blocked_by_trend_danger" in frame.columns:
        frame.loc[
            frame["position"].eq(0) & frame["blocked_by_trend_danger"],
            "blocked_category",
        ] = "trend-danger"
    frame["opportunity_gross_dollars"] = 0.0
    if {"base_position", "price_move", "point_value"}.issubset(frame.columns):
        frame["opportunity_gross_dollars"] = (
            frame["base_position"].fillna(0).astype(float)
            * frame["price_move"].fillna(0.0).astype(float)
            * frame["point_value"].fillna(0.0).astype(float)
        )
    if "round_turn_cost_dollars" in frame.columns:
        frame["opportunity_cost_dollars"] = np.where(
            frame["base_position"].fillna(0).ne(0),
            frame["round_turn_cost_dollars"].fillna(0.0),
            0.0,
        )
    else:
        frame["opportunity_cost_dollars"] = 0.0
    frame["opportunity_net_dollars"] = (
        frame["opportunity_gross_dollars"] - frame["opportunity_cost_dollars"]
    )
    records: list[dict[str, Any]] = []
    for category, group in frame.groupby("blocked_category", dropna=False):
        record = _metrics(group, "blocked_category", {"blocked_category": category})
        record.update(
            {
                "base_signal_count": int(group["base_position"].fillna(0).ne(0).sum())
                if "base_position" in group
                else None,
                "opportunity_gross_dollars": _sum_float(group["opportunity_gross_dollars"]),
                "opportunity_cost_dollars": _sum_float(group["opportunity_cost_dollars"]),
                "opportunity_net_dollars": _sum_float(group["opportunity_net_dollars"]),
            }
        )
        records.append(record)
    return pd.DataFrame(records).sort_values("blocked_category").reset_index(drop=True)


def _cost_counterfactuals(policy_frame: pd.DataFrame) -> pd.DataFrame:
    traded = policy_frame[policy_frame["trade_count"].eq(1)].copy()
    gross = _sum_float(traded["gross_dollars"]) if not traded.empty else 0.0
    slippage = _sum_float(traded["slippage_cost_dollars"]) if not traded.empty else 0.0
    commission = _sum_float(traded["commission_cost_dollars"]) if not traded.empty else 0.0
    scenarios = [
        ("current", 1.0, True),
        ("zero_cost", 0.0, False),
        ("commission_only", 0.0, True),
        ("slippage_only", 1.0, False),
        ("slippage_25pct", 0.25, True),
        ("slippage_50pct", 0.50, True),
        ("slippage_75pct", 0.75, True),
    ]
    records: list[dict[str, Any]] = []
    for scenario, slippage_scale, include_commission in scenarios:
        scenario_slippage = slippage * slippage_scale
        scenario_commission = commission if include_commission else 0.0
        scenario_cost = scenario_slippage + scenario_commission
        records.append(
            {
                "scenario": scenario,
                "trade_count": int(len(traded)),
                "gross_return_dollars": gross,
                "slippage_scale": slippage_scale,
                "slippage_cost_dollars": scenario_slippage,
                "commission_cost_dollars": scenario_commission,
                "cost_dollars": scenario_cost,
                "net_return_dollars": gross - scenario_cost,
            }
        )
    return pd.DataFrame(records)


def _long_failure_breakdown(policy_frame: pd.DataFrame) -> pd.DataFrame:
    long_trades = policy_frame[policy_frame["position"].eq(1)].copy()
    if long_trades.empty:
        return _empty_unavailable("no long trades")
    records = [_metrics(long_trades, "long_overall", {})]
    for market, group in long_trades.groupby("market", dropna=False):
        records.append(_metrics(group, "long_market", {"market": market}))
    for (market, fold_id), group in long_trades.groupby(["market", "fold_id"], dropna=False):
        records.append(_metrics(group, "long_market_fold", {"market": market, "fold_id": fold_id}))
    return pd.DataFrame(records).sort_values("net_return_dollars").reset_index(drop=True)


def _major_positive_examples(frames: Iterable[pd.DataFrame]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for frame in frames:
        if frame.empty or "gross_return_dollars" not in frame or "net_return_dollars" not in frame:
            continue
        positive = frame[
            frame["gross_return_dollars"].astype(float).gt(0.0)
            & frame["net_return_dollars"].astype(float).gt(0.0)
        ]
        examples.extend(positive.head(5).to_dict(orient="records"))
    return examples[:5]


def _top_findings(
    *,
    overall: Mapping[str, Any],
    by_market: pd.DataFrame,
    by_side: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    blocked: pd.DataFrame,
    any_positive_bucket: bool,
) -> list[str]:
    findings: list[str] = []
    gross = float(overall.get("gross_return_dollars") or 0.0)
    cost = float(overall.get("cost_dollars") or 0.0)
    net = float(overall.get("net_return_dollars") or 0.0)
    findings.append(
        f"Gross before costs was {'positive' if gross > 0 else 'negative'} at {gross:.2f}; net after costs was {net:.2f}."
    )
    if abs(gross) > 0:
        findings.append(f"Total costs were {cost / abs(gross):.2f}x absolute gross PnL.")
    else:
        findings.append("Gross PnL was zero, so any nonzero cost directly reduced net PnL.")
    if not by_market.empty:
        row = by_market.iloc[0]
        findings.append(
            f"Worst market by net was {row.get('market')} at {float(row.get('net_return_dollars')):.2f}."
        )
    side_rows = by_side[by_side["side"].isin(["long", "short"])] if "side" in by_side else pd.DataFrame()
    if not side_rows.empty:
        row = side_rows.sort_values("net_return_dollars").iloc[0]
        findings.append(
            f"{row.get('side')} trades caused the largest side damage at {float(row.get('net_return_dollars')):.2f}."
        )
    zero_cost = counterfactuals[counterfactuals["scenario"].eq("zero_cost")]
    if not zero_cost.empty:
        findings.append(
            f"With zero costs, net would be {float(zero_cost.iloc[0]['net_return_dollars']):.2f}."
        )
    traded = blocked[blocked["blocked_category"].eq("traded")]
    trend = blocked[blocked["blocked_category"].eq("trend-danger")]
    if not traded.empty and not trend.empty:
        findings.append(
            f"Policy traded {int(traded.iloc[0]['row_count'])} rows and trend-danger blocked {int(trend.iloc[0]['row_count'])} rows."
        )
    findings.append(
        "At least one major bucket was gross-positive and net-positive."
        if any_positive_bucket
        else "No major bucket was both gross-positive and net-positive."
    )
    return findings[:5]


def _write_readme(path: Path, summary: Mapping[str, Any], outputs: Mapping[str, str]) -> None:
    findings = "\n".join(f"- {item}" for item in summary["top_findings"])
    files = "\n".join(f"- `{value}`" for key, value in sorted(outputs.items()) if key != "readme")
    text = f"""# Phase 8 Trade Failure Drilldown

Run: `{summary['run']}`

This report traces saved WFA predictions through Phase 8 policy rows, executed
research-policy trades, gross PnL, costs, and net PnL. It does not change
labels, models, WFA splits, predictions, or policy behavior.

## Top Findings

{findings}

## Files

{files}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_trade_failure_drilldown(
    *,
    predictions_path: Path,
    costs_config: Path,
    output_root: Path,
    run: str,
    policy: PolicyConfig,
) -> dict[str, Any]:
    if not predictions_path.exists():
        raise SystemExit(f"prediction parquet missing: {_relative_path(predictions_path)}")

    predictions = pd.read_parquet(predictions_path)
    policy_frame, failures, warnings = build_policy_frame(predictions, costs_config, policy)
    if failures:
        raise SystemExit("; ".join(failures))
    policy_frame = _first_optional_values(predictions, policy_frame)
    policy_frame = _attach_side(policy_frame)

    unavailable: list[str] = []
    by_market_side = _group_metrics(
        policy_frame,
        scope="market_side",
        group_cols=["market", "side"],
    )

    by_hour_frame, hour_error = _attach_hour(policy_frame)
    if hour_error:
        unavailable.append(f"pnl_by_market_side_hour: {hour_error}")
        by_market_side_hour = _empty_unavailable(hour_error)
    else:
        by_market_side_hour = _group_metrics(
            by_hour_frame,
            scope="market_side_hour",
            group_cols=["market", "side", "hour_utc"],
        )

    by_fold = _group_metrics(policy_frame, scope="fold", group_cols=["fold_id"])

    confidence_frame, confidence_error = _attach_confidence_bucket(policy_frame)
    if confidence_error:
        unavailable.append(f"pnl_by_confidence_bucket: {confidence_error}")
        by_confidence = _empty_unavailable(confidence_error)
    else:
        by_confidence = _group_metrics(
            confidence_frame,
            scope="confidence_bucket",
            group_cols=["confidence_bucket"],
        )

    regime_frame, regime_column, regime_error = _attach_regime_bucket(policy_frame)
    if regime_error:
        unavailable.append(f"pnl_by_regime_bucket: {regime_error}")
        by_regime = _empty_unavailable(regime_error)
    else:
        by_regime = _group_metrics(
            regime_frame,
            scope="regime_bucket",
            group_cols=["regime_bucket"],
        )
        by_regime.insert(1, "source_column", regime_column)

    volatility_column = _select_volatility_column(policy_frame)
    if volatility_column is None:
        unavailable.append("pnl_by_volatility_bucket: no numeric ATR or volatility column available")
        by_volatility = _empty_unavailable("no numeric ATR or volatility column available")
    else:
        volatility_frame, volatility_error = _attach_quantile_bucket(
            policy_frame,
            source_column=volatility_column,
            output_column="volatility_bucket",
        )
        if volatility_error:
            unavailable.append(f"pnl_by_volatility_bucket: {volatility_error}")
            by_volatility = _empty_unavailable(volatility_error)
        else:
            by_volatility = _group_metrics(
                volatility_frame,
                scope="volatility_bucket",
                group_cols=["volatility_bucket"],
            )
            by_volatility.insert(1, "source_column", volatility_column)

    blocked = _blocked_category_frame(policy_frame)
    counterfactuals = _cost_counterfactuals(policy_frame)
    long_failure = _long_failure_breakdown(policy_frame)
    by_side = _group_metrics(policy_frame, scope="side", group_cols=["side"])
    overall = _metrics(policy_frame, "overall", {})
    positive_examples = _major_positive_examples(
        [by_market_side, by_fold, by_confidence, by_regime, by_volatility]
    )
    any_positive = bool(positive_examples)

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, run)
    _write_csv(paths["market_side"], by_market_side)
    _write_csv(paths["market_side_hour"], by_market_side_hour)
    _write_csv(paths["fold"], by_fold)
    _write_csv(paths["confidence"], by_confidence)
    _write_csv(paths["regime"], by_regime)
    _write_csv(paths["volatility"], by_volatility)
    _write_csv(paths["counterfactuals"], counterfactuals)
    _write_csv(paths["blocked"], blocked)
    _write_csv(paths["long_failure"], long_failure)

    outputs = {key: _relative_path(path) for key, path in paths.items()}
    top_findings = _top_findings(
        overall=overall,
        by_market=_group_metrics(policy_frame, scope="market", group_cols=["market"]),
        by_side=by_side,
        counterfactuals=counterfactuals,
        blocked=blocked,
        any_positive_bucket=any_positive,
    )
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "output_root": _relative_path(output_root),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(policy_frame)),
        "trade_count": int(policy_frame["trade_count"].sum()),
        "block_counts": blocked[["blocked_category", "row_count"]].to_dict(orient="records"),
        "totals": overall,
        "gross_positive_before_costs": float(overall["gross_return_dollars"]) > 0.0,
        "any_major_bucket_gross_positive_and_net_positive": any_positive,
        "major_positive_bucket_examples": positive_examples,
        "worst_markets_by_net": _group_metrics(
            policy_frame,
            scope="market",
            group_cols=["market"],
        )
        .head(5)
        .to_dict(orient="records"),
        "side_damage": by_side.sort_values("net_return_dollars").to_dict(orient="records"),
        "top_findings": top_findings,
        "unavailable_diagnostics": unavailable,
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
    parser.add_argument("--min-fade-success", type=float, default=0.50)
    parser.add_argument("--max-trend-danger", type=float, default=0.50)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_trade_failure_drilldown(
        predictions_path=Path(args.predictions),
        costs_config=Path(args.costs_config),
        output_root=Path(args.output_root),
        run=args.run,
        policy=PolicyConfig(
            long_short_margin=args.long_short_margin,
            min_fade_success=args.min_fade_success,
            max_trend_danger=args.max_trend_danger,
        ),
    )
    totals = summary["totals"]
    print(
        "PASS trade failure drilldown: "
        f"predictions={summary['prediction_count']} "
        f"policy_rows={summary['policy_row_count']} "
        f"trades={summary['trade_count']} "
        f"net_dollars={totals['net_return_dollars']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
