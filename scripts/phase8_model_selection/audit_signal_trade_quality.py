#!/usr/bin/env python3
"""Audit traded signal quality and threshold sensitivity for Phase 8 outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.phase8_model_selection.audit_policy_failure import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREDICTIONS,
    DEFAULT_RUN,
    _relative_path,
    _write_csv,
    _write_json,
)
from scripts.phase8_model_selection.audit_trade_failure_drilldown import (
    _empty_unavailable,
    _group_metrics,
    _metrics,
)
from scripts.phase8_model_selection.evaluate_predictions import (
    DEFAULT_COSTS_CONFIG,
    PolicyConfig,
    _direction_accuracy,
    _score_column_for_target,
    build_policy_frame,
)


OUTPUT_SUFFIXES = {
    "summary": "signal_trade_quality_summary.json",
    "traded_by_market_side": "traded_signal_by_market_side.csv",
    "traded_by_fold_side": "traded_signal_by_fold_side.csv",
    "traded_by_confidence_side": "traded_signal_by_confidence_side.csv",
    "target_prediction_scale": "target_prediction_scale.csv",
    "threshold_sensitivity": "policy_threshold_sensitivity.csv",
    "long_failure": "long_signal_failure_concentration.csv",
    "block_counts": "signal_block_counts.csv",
    "readme": "signal_trade_quality_readme.md",
}


def _output_paths(output_root: Path, run: str) -> dict[str, Path]:
    return {key: output_root / f"{run}_{suffix}" for key, suffix in OUTPUT_SUFFIXES.items()}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_accuracy(predicted: pd.Series, actual: pd.Series) -> float | None:
    aligned = pd.DataFrame({"predicted": predicted, "actual": actual}).dropna()
    if aligned.empty:
        return None
    aligned = aligned[aligned["predicted"].ne(0) & aligned["actual"].isin([-1, 0, 1])]
    if aligned.empty:
        return None
    return float(aligned["predicted"].eq(aligned["actual"]).mean())


def _attach_signal_fields(policy_frame: pd.DataFrame) -> pd.DataFrame:
    frame = policy_frame.copy()
    frame["side"] = frame["position"].map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    frame["base_side"] = frame["base_position"].map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    frame["confidence_abs_margin"] = _numeric(frame, "direction_margin").abs()
    frame["confidence_bucket"] = pd.cut(
        frame["confidence_abs_margin"],
        bins=[-np.inf, 0.05, 0.10, 0.20, 0.40, np.inf],
        labels=["<=0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">0.40"],
    ).astype(str)
    actual = _numeric(frame, "observed_direction_target")
    frame["direction_correct"] = frame["position"].eq(actual).where(
        frame["position"].ne(0),
        pd.NA,
    ).astype("boolean")
    frame["base_direction_correct"] = frame["base_position"].eq(actual).where(
        frame["base_position"].ne(0),
        pd.NA,
    ).astype("boolean")
    return frame


def _quality_metrics(frame: pd.DataFrame, scope: str, keys: Mapping[str, Any]) -> dict[str, Any]:
    record = _metrics(frame, scope, keys)
    traded = frame[frame["trade_count"].eq(1)]
    base_signals = frame[frame["base_position"].ne(0)]
    record.update(
        {
            "avg_confidence_abs_margin": float(traded["confidence_abs_margin"].mean())
            if not traded.empty
            else None,
            "median_confidence_abs_margin": float(traded["confidence_abs_margin"].median())
            if not traded.empty
            else None,
            "traded_direction_accuracy": _finite_accuracy(
                traded["position"], _numeric(traded, "observed_direction_target")
            )
            if not traded.empty and "observed_direction_target" in traded
            else None,
            "base_signal_direction_accuracy": _finite_accuracy(
                base_signals["base_position"], _numeric(base_signals, "observed_direction_target")
            )
            if not base_signals.empty and "observed_direction_target" in base_signals
            else None,
            "avg_p_long": float(traded["p_long"].mean()) if not traded.empty else None,
            "avg_p_short": float(traded["p_short"].mean()) if not traded.empty else None,
            "avg_p_fade_success": float(traded["p_fade_success"].mean()) if not traded.empty else None,
            "avg_p_trend_danger": float(traded["p_trend_danger"].mean()) if not traded.empty else None,
        }
    )
    return record


def _group_quality(
    frame: pd.DataFrame,
    *,
    scope: str,
    group_cols: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        records.append(_quality_metrics(group, scope, dict(zip(group_cols, keys))))
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values("net_return_dollars", ascending=True).reset_index(drop=True)


def _target_prediction_scale(predictions: pd.DataFrame, unavailable: list[str]) -> pd.DataFrame:
    required = {"model_id", "model_family", "target_name", "market", "fold_id", "y_true"}
    if not required.issubset(predictions.columns):
        unavailable.append(
            "target_prediction_scale: missing columns "
            + ",".join(sorted(required - set(predictions.columns)))
        )
        return _empty_unavailable("missing required prediction scale columns")

    records: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["model_id", "model_family", "target_name"], dropna=False):
        model_id, model_family, target_name = keys
        try:
            score = _score_column_for_target(group, str(target_name))
        except KeyError as exc:
            unavailable.append(f"target_prediction_scale: {target_name} missing score column {exc}")
            continue
        y_true = _numeric(group, "y_true")
        aligned = pd.DataFrame({"score": score, "y_true": y_true}).dropna()
        pred_std = float(aligned["score"].std(ddof=0)) if not aligned.empty else None
        target_std = float(aligned["y_true"].std(ddof=0)) if not aligned.empty else None
        scale_ratio = (
            abs(pred_std) / abs(target_std)
            if pred_std is not None and target_std not in (None, 0.0)
            else None
        )
        warnings: list[str] = []
        if scale_ratio is not None and (scale_ratio > 100.0 or scale_ratio < 0.01):
            warnings.append("prediction/target scale ratio is extreme")
        record: dict[str, Any] = {
            "model_id": model_id,
            "model_family": model_family,
            "target_name": target_name,
            "row_count": int(len(group)),
            "market_count": int(group["market"].nunique(dropna=True)),
            "fold_count": int(group["fold_id"].nunique(dropna=True)),
            "prediction_mean": float(aligned["score"].mean()) if not aligned.empty else None,
            "prediction_std": pred_std,
            "prediction_abs_p95": float(aligned["score"].abs().quantile(0.95))
            if not aligned.empty
            else None,
            "target_mean": float(aligned["y_true"].mean()) if not aligned.empty else None,
            "target_std": target_std,
            "target_abs_p95": float(aligned["y_true"].abs().quantile(0.95))
            if not aligned.empty
            else None,
            "prediction_to_target_std_ratio": scale_ratio,
            "scale_warnings": "; ".join(warnings),
        }
        if target_name == "target_sign_with_deadzone":
            record["all_row_direction_accuracy"] = _direction_accuracy(group)
        records.append(record)
    if not records:
        return _empty_unavailable("no target prediction scale records")
    return pd.DataFrame(records).sort_values(["model_id", "target_name"]).reset_index(drop=True)


def _scenario_metrics(
    policy_frame: pd.DataFrame,
    *,
    direction_margin: float,
    min_fade_success: float,
    max_trend_danger: float,
) -> dict[str, Any]:
    frame = policy_frame.copy()
    margin = _numeric(frame, "direction_margin")
    base_position = pd.Series(0, index=frame.index)
    base_position.loc[margin.ge(direction_margin)] = 1
    base_position.loc[margin.le(-direction_margin)] = -1
    fade_allowed = _numeric(frame, "p_fade_success").ge(min_fade_success).fillna(False)
    trend_ok = _numeric(frame, "p_trend_danger").lt(max_trend_danger).fillna(False)
    frame["position"] = np.where(fade_allowed & trend_ok, base_position, 0).astype(int)
    frame["trade_count"] = frame["position"].ne(0).astype(int)
    frame["long_count"] = frame["position"].eq(1).astype(int)
    frame["short_count"] = frame["position"].eq(-1).astype(int)
    frame["flat_count"] = frame["position"].eq(0).astype(int)
    frame["gross_dollars"] = (
        frame["position"]
        * _numeric(frame, "price_move").fillna(0.0)
        * _numeric(frame, "point_value").fillna(0.0)
    )
    row_slippage = (
        2.0
        * _numeric(frame, "slippage_ticks_per_side").fillna(0.0)
        * _numeric(frame, "tick_value").fillna(0.0)
    )
    round_turn = _numeric(frame, "round_turn_cost_dollars").fillna(0.0)
    row_commission = np.maximum(round_turn - row_slippage, 0.0)
    frame["slippage_cost_dollars"] = np.where(frame["position"].ne(0), row_slippage, 0.0)
    frame["commission_cost_dollars"] = np.where(frame["position"].ne(0), row_commission, 0.0)
    frame["cost_dollars"] = np.where(frame["position"].ne(0), round_turn, 0.0)
    frame["net_dollars"] = frame["gross_dollars"] - frame["cost_dollars"]
    record = _quality_metrics(
        _attach_signal_fields(frame),
        "threshold_scenario",
        {
            "direction_margin_threshold": direction_margin,
            "min_fade_success": min_fade_success,
            "max_trend_danger": max_trend_danger,
        },
    )
    record["gross_positive"] = float(record["gross_return_dollars"]) > 0.0
    record["net_positive"] = float(record["net_return_dollars"]) > 0.0
    return record


def _threshold_sensitivity(policy_frame: pd.DataFrame, unavailable: list[str]) -> pd.DataFrame:
    required = {
        "direction_margin",
        "p_fade_success",
        "p_trend_danger",
        "price_move",
        "point_value",
        "round_turn_cost_dollars",
        "slippage_ticks_per_side",
        "tick_value",
    }
    missing = sorted(required - set(policy_frame.columns))
    if missing:
        unavailable.append("threshold_sensitivity: missing columns " + ",".join(missing))
        return _empty_unavailable("missing threshold sensitivity columns")
    records = [
        _scenario_metrics(
            policy_frame,
            direction_margin=direction_margin,
            min_fade_success=min_fade_success,
            max_trend_danger=max_trend_danger,
        )
        for direction_margin in (0.05, 0.10, 0.15, 0.20, 0.30)
        for min_fade_success in (0.50, 0.60, 0.70, 0.80, 0.90)
        for max_trend_danger in (0.20, 0.30, 0.40, 0.50)
    ]
    return (
        pd.DataFrame(records)
        .sort_values("net_return_dollars", ascending=False)
        .reset_index(drop=True)
    )


def _block_counts(policy_frame: pd.DataFrame) -> pd.DataFrame:
    frame = policy_frame.copy()
    frame["block_category"] = "other"
    frame.loc[frame["position"].ne(0), "block_category"] = "traded"
    frame.loc[frame["position"].eq(0) & frame["no_direction_signal"], "block_category"] = "no_direction"
    frame.loc[frame["position"].eq(0) & frame["blocked_by_trend_danger"], "block_category"] = "trend_danger"
    frame.loc[frame["position"].eq(0) & frame["blocked_by_fade_filter"], "block_category"] = "fade_filter"
    return _group_metrics(frame, scope="block_category", group_cols=["block_category"], sort_by_net=False)


def _long_failure_concentration(policy_frame: pd.DataFrame) -> pd.DataFrame:
    long_trades = policy_frame[policy_frame["position"].eq(1)].copy()
    if long_trades.empty:
        return _empty_unavailable("no long trades")
    return _group_quality(
        long_trades,
        scope="long_market_fold_confidence",
        group_cols=["market", "fold_id", "confidence_bucket"],
    )


def _write_readme(path: Path, summary: Mapping[str, Any], outputs: Mapping[str, str]) -> None:
    findings = "\n".join(f"- {item}" for item in summary["top_findings"])
    files = "\n".join(f"- `{value}`" for key, value in sorted(outputs.items()) if key != "readme")
    text = f"""# Phase 8 Signal Trade Quality

Run: `{summary['run']}`

This diagnostic reads saved WFA predictions and recomputes Phase 8 policy rows
for signal-quality analysis only. It does not change labels, features, model
training, WFA splits, saved predictions, or policy behavior.

## Top Findings

{findings}

## Files

{files}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _top_findings(
    *,
    overall: Mapping[str, Any],
    target_scale: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    long_quality: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    gross = float(overall.get("gross_return_dollars") or 0.0)
    net = float(overall.get("net_return_dollars") or 0.0)
    findings.append(
        f"Traded rows had gross PnL {gross:.2f} and net PnL {net:.2f}; costs are not the only issue."
    )
    direction = target_scale[target_scale["target_name"].eq("target_sign_with_deadzone")]
    if not direction.empty and "all_row_direction_accuracy" in direction:
        value = direction.iloc[0].get("all_row_direction_accuracy")
        if pd.notna(value):
            findings.append(f"All-row direction accuracy was {float(value):.4f}.")
    scale_flags = target_scale[target_scale.get("scale_warnings", pd.Series(dtype=str)).astype(str).ne("")]
    if not scale_flags.empty:
        row = scale_flags.iloc[0]
        findings.append(
            f"{row['model_id']} has an extreme prediction/target scale ratio of {float(row['prediction_to_target_std_ratio']):.2f}."
        )
    positive = threshold_sensitivity[
        threshold_sensitivity.get("gross_positive", pd.Series(dtype=bool)).astype(bool)
        & threshold_sensitivity.get("net_positive", pd.Series(dtype=bool)).astype(bool)
    ]
    if positive.empty:
        findings.append("No tested policy-threshold scenario was both gross-positive and net-positive.")
    else:
        row = positive.iloc[0]
        findings.append(
            "Best positive threshold scenario net was "
            f"{float(row['net_return_dollars']):.2f} with margin "
            f"{float(row['direction_margin_threshold']):.2f}."
        )
    if not long_quality.empty and "unavailable_reason" not in long_quality:
        row = long_quality.sort_values("net_return_dollars").iloc[0]
        findings.append(
            f"Worst long bucket was {row.get('market')} {row.get('fold_id')} "
            f"{row.get('confidence_bucket')} at {float(row.get('net_return_dollars')):.2f}."
        )
    return findings[:5]


def build_signal_trade_quality(
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
    policy_frame = _attach_signal_fields(policy_frame)
    traded = policy_frame[policy_frame["trade_count"].eq(1)].copy()
    unavailable: list[str] = []

    traded_by_market_side = _group_quality(
        traded,
        scope="traded_market_side",
        group_cols=["market", "side"],
    )
    traded_by_fold_side = _group_quality(
        traded,
        scope="traded_fold_side",
        group_cols=["fold_id", "side"],
    )
    traded_by_confidence_side = _group_quality(
        traded,
        scope="traded_confidence_side",
        group_cols=["confidence_bucket", "side"],
    )
    target_scale = _target_prediction_scale(predictions, unavailable)
    threshold_sensitivity = _threshold_sensitivity(policy_frame, unavailable)
    long_failure = _long_failure_concentration(policy_frame)
    block_counts = _block_counts(policy_frame)
    overall = _quality_metrics(policy_frame, "overall", {})
    positive_thresholds = threshold_sensitivity[
        threshold_sensitivity.get("gross_positive", pd.Series(dtype=bool)).astype(bool)
        & threshold_sensitivity.get("net_positive", pd.Series(dtype=bool)).astype(bool)
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, run)
    _write_csv(paths["traded_by_market_side"], traded_by_market_side)
    _write_csv(paths["traded_by_fold_side"], traded_by_fold_side)
    _write_csv(paths["traded_by_confidence_side"], traded_by_confidence_side)
    _write_csv(paths["target_prediction_scale"], target_scale)
    _write_csv(paths["threshold_sensitivity"], threshold_sensitivity)
    _write_csv(paths["long_failure"], long_failure)
    _write_csv(paths["block_counts"], block_counts)

    outputs = {key: _relative_path(path) for key, path in paths.items()}
    top_findings = _top_findings(
        overall=overall,
        target_scale=target_scale,
        threshold_sensitivity=threshold_sensitivity,
        long_quality=long_failure,
    )
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(policy_frame)),
        "trade_count": int(policy_frame["trade_count"].sum()),
        "overall": overall,
        "threshold_scenarios_tested": int(len(threshold_sensitivity))
        if "unavailable_reason" not in threshold_sensitivity
        else 0,
        "positive_gross_and_net_threshold_scenario_count": int(len(positive_thresholds)),
        "best_threshold_scenarios_by_net": threshold_sensitivity.head(10).to_dict(orient="records"),
        "long_failure_worst_buckets": long_failure.head(10).to_dict(orient="records"),
        "target_prediction_scale_warnings": target_scale[
            target_scale.get("scale_warnings", pd.Series(dtype=str)).astype(str).ne("")
        ].to_dict(orient="records")
        if "scale_warnings" in target_scale
        else [],
        "block_counts": block_counts.to_dict(orient="records"),
        "recommend_label_feature_audit": bool(positive_thresholds.empty),
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
    summary = build_signal_trade_quality(
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
    print(
        "PASS signal trade quality: "
        f"predictions={summary['prediction_count']} "
        f"policy_rows={summary['policy_row_count']} "
        f"trades={summary['trade_count']} "
        f"net_dollars={summary['overall']['net_return_dollars']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
