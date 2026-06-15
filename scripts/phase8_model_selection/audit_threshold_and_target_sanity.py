#!/usr/bin/env python3
"""Audit threshold stability and return-target scale for Phase 8 outputs."""

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
from scripts.phase8_model_selection.evaluate_predictions import (
    DEFAULT_COSTS_CONFIG,
    PolicyConfig,
    build_policy_frame,
)


RETURN_TARGET = "target_ret_15m"
RETURN_MODEL = "ridge_return_v1"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _sum_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _median_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _std_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=0)) if not values.empty else None


def _quantile_or_none(series: pd.Series, q: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else None


def _win_rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.gt(0.0).mean()) if not values.empty else None


def _direction_accuracy(position: pd.Series, actual: pd.Series) -> float | None:
    aligned = pd.DataFrame({"position": position, "actual": actual}).dropna()
    aligned = aligned[aligned["position"].ne(0) & aligned["actual"].isin([-1, 0, 1])]
    if aligned.empty:
        return None
    return float(aligned["position"].eq(aligned["actual"]).mean())


def _output_paths(output_root: Path, run: str) -> dict[str, Path]:
    return {
        "threshold_stability": output_root / f"{run}_threshold_stability.csv",
        "return_target_scale": output_root / f"{run}_return_target_scale_audit.csv",
        "summary": output_root / f"{run}_next_action_summary.json",
    }


def _scenario_frame(
    policy_frame: pd.DataFrame,
    *,
    direction_margin_threshold: float,
    min_fade_success: float,
    max_trend_danger: float,
) -> pd.DataFrame:
    frame = policy_frame.copy()
    direction_margin = _numeric(frame, "direction_margin")
    position = pd.Series(0, index=frame.index)
    position.loc[direction_margin.ge(direction_margin_threshold)] = 1
    position.loc[direction_margin.le(-direction_margin_threshold)] = -1
    fade_allowed = _numeric(frame, "p_fade_success").ge(min_fade_success).fillna(False)
    trend_ok = _numeric(frame, "p_trend_danger").lt(max_trend_danger).fillna(False)
    frame["scenario_position"] = np.where(fade_allowed & trend_ok, position, 0).astype(int)
    frame["side"] = frame["scenario_position"].map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    frame["scenario_trade_count"] = frame["scenario_position"].ne(0).astype(int)
    frame["scenario_long_count"] = frame["scenario_position"].eq(1).astype(int)
    frame["scenario_short_count"] = frame["scenario_position"].eq(-1).astype(int)
    frame["scenario_flat_count"] = frame["scenario_position"].eq(0).astype(int)
    frame["scenario_gross_dollars"] = (
        frame["scenario_position"]
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
    frame["scenario_slippage_cost_dollars"] = np.where(
        frame["scenario_position"].ne(0),
        row_slippage,
        0.0,
    )
    frame["scenario_commission_cost_dollars"] = np.where(
        frame["scenario_position"].ne(0),
        row_commission,
        0.0,
    )
    frame["scenario_cost_dollars"] = np.where(frame["scenario_position"].ne(0), round_turn, 0.0)
    frame["scenario_net_dollars"] = frame["scenario_gross_dollars"] - frame["scenario_cost_dollars"]
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["scenario_year"] = timestamp.dt.year
    frame["scenario_month"] = timestamp.dt.strftime("%Y-%m")
    frame["scenario_hour_utc"] = timestamp.dt.hour
    return frame


def _scenario_metrics(frame: pd.DataFrame, scope: str, keys: Mapping[str, Any]) -> dict[str, Any]:
    traded = frame[frame["scenario_trade_count"].eq(1)]
    rows = int(len(frame))
    trades = int(len(traded))
    gross = _sum_float(frame["scenario_gross_dollars"])
    slippage = _sum_float(frame["scenario_slippage_cost_dollars"])
    commission = _sum_float(frame["scenario_commission_cost_dollars"])
    cost = _sum_float(frame["scenario_cost_dollars"])
    net = _sum_float(frame["scenario_net_dollars"])
    return {
        "scope": scope,
        **dict(keys),
        "row_count": rows,
        "trade_count": trades,
        "long_count": int(frame["scenario_long_count"].sum()),
        "short_count": int(frame["scenario_short_count"].sum()),
        "flat_count": int(frame["scenario_flat_count"].sum()),
        "gross_return_dollars": gross,
        "slippage_cost_dollars": slippage,
        "commission_cost_dollars": commission,
        "cost_dollars": cost,
        "net_return_dollars": net,
        "mean_gross_per_trade": _mean_or_none(traded["scenario_gross_dollars"]) if trades else None,
        "median_gross_per_trade": _median_or_none(traded["scenario_gross_dollars"]) if trades else None,
        "mean_net_per_trade": _mean_or_none(traded["scenario_net_dollars"]) if trades else None,
        "median_net_per_trade": _median_or_none(traded["scenario_net_dollars"]) if trades else None,
        "gross_win_rate": _win_rate(traded["scenario_gross_dollars"]) if trades else None,
        "net_win_rate": _win_rate(traded["scenario_net_dollars"]) if trades else None,
        "avg_cost_per_trade": cost / trades if trades else None,
        "avg_slippage_per_trade": slippage / trades if trades else None,
        "avg_commission_per_trade": commission / trades if trades else None,
        "direction_accuracy": _direction_accuracy(
            traded["scenario_position"],
            _numeric(traded, "observed_direction_target"),
        )
        if trades and "observed_direction_target" in traded
        else None,
        "gross_positive": gross > 0.0,
        "net_positive": net > 0.0,
    }


def _threshold_stability_rows(scenario: pd.DataFrame) -> pd.DataFrame:
    group_specs = [
        ("overall", []),
        ("market", ["market"]),
        ("fold", ["fold_id"]),
        ("side", ["side"]),
        ("year", ["scenario_year"]),
        ("month", ["scenario_month"]),
        ("hour", ["scenario_hour_utc"]),
        ("market_side", ["market", "side"]),
        ("market_fold", ["market", "fold_id"]),
        ("side_hour", ["side", "scenario_hour_utc"]),
        ("market_month", ["market", "scenario_month"]),
    ]
    records: list[dict[str, Any]] = []
    for scope, group_cols in group_specs:
        if not group_cols:
            records.append(_scenario_metrics(scenario, scope, {}))
            continue
        for keys, group in scenario.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            records.append(_scenario_metrics(group, scope, dict(zip(group_cols, keys))))
    frame = pd.DataFrame(records)
    sort_cols = ["scope", "net_return_dollars", "trade_count"]
    return frame.sort_values(sort_cols, ascending=[True, False, False]).reset_index(drop=True)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return abs(numerator) / abs(denominator)


def _correlation_or_none(left: pd.Series, right: pd.Series) -> float | None:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(aligned) < 2 or aligned["left"].nunique() < 2 or aligned["right"].nunique() < 2:
        return None
    return float(aligned["left"].corr(aligned["right"]))


def _sign_agreement_or_none(prediction: pd.Series, target: pd.Series) -> float | None:
    aligned = pd.DataFrame({"prediction": prediction, "target": target}).dropna()
    aligned = aligned[aligned["prediction"].ne(0) & aligned["target"].ne(0)]
    if aligned.empty:
        return None
    return float(np.sign(aligned["prediction"]).eq(np.sign(aligned["target"])).mean())


def _scale_record(group: pd.DataFrame, scope: str, keys: Mapping[str, Any]) -> dict[str, Any]:
    prediction = _numeric(group, "y_pred_calibrated")
    raw_prediction = _numeric(group, "y_pred_raw")
    target = _numeric(group, "y_true")
    pred_std = _std_or_none(prediction)
    target_std = _std_or_none(target)
    pred_abs_p95 = _quantile_or_none(prediction.abs(), 0.95)
    target_abs_p95 = _quantile_or_none(target.abs(), 0.95)
    std_ratio = _safe_ratio(pred_std, target_std)
    p95_ratio = _safe_ratio(pred_abs_p95, target_abs_p95)
    warnings: list[str] = []
    if std_ratio is not None and (std_ratio > 100.0 or std_ratio < 0.01):
        warnings.append("prediction/target std ratio is extreme")
    if p95_ratio is not None and (p95_ratio > 100.0 or p95_ratio < 0.01):
        warnings.append("prediction/target p95 ratio is extreme")
    if target_abs_p95 is not None and target_abs_p95 < 0.01 and pred_abs_p95 is not None and pred_abs_p95 > 1.0:
        warnings.append("target looks fractional while predictions look large-scale")
    return {
        "scope": scope,
        **dict(keys),
        "row_count": int(len(group)),
        "prediction_mean": _mean_or_none(prediction),
        "prediction_std": pred_std,
        "prediction_abs_p50": _quantile_or_none(prediction.abs(), 0.50),
        "prediction_abs_p95": pred_abs_p95,
        "prediction_abs_p99": _quantile_or_none(prediction.abs(), 0.99),
        "raw_prediction_abs_p95": _quantile_or_none(raw_prediction.abs(), 0.95),
        "target_mean": _mean_or_none(target),
        "target_std": target_std,
        "target_abs_p50": _quantile_or_none(target.abs(), 0.50),
        "target_abs_p95": target_abs_p95,
        "target_abs_p99": _quantile_or_none(target.abs(), 0.99),
        "prediction_to_target_std_ratio": std_ratio,
        "prediction_to_target_abs_p95_ratio": p95_ratio,
        "prediction_target_correlation": _correlation_or_none(prediction, target),
        "prediction_target_sign_agreement": _sign_agreement_or_none(prediction, target),
        "scale_warnings": "; ".join(warnings),
    }


def _return_target_scale_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"model_id", "target_name", "market", "fold_id", "y_true", "y_pred_raw", "y_pred_calibrated"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        return pd.DataFrame([{"unavailable_reason": "missing columns: " + ",".join(missing)}])
    frame = predictions[
        predictions["model_id"].eq(RETURN_MODEL) & predictions["target_name"].eq(RETURN_TARGET)
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            [{"unavailable_reason": f"missing {RETURN_MODEL} / {RETURN_TARGET} rows"}]
        )
    specs = [
        ("overall", []),
        ("market", ["market"]),
        ("fold", ["fold_id"]),
        ("market_fold", ["market", "fold_id"]),
    ]
    records: list[dict[str, Any]] = []
    for scope, group_cols in specs:
        if not group_cols:
            records.append(_scale_record(frame, scope, {}))
            continue
        for keys, group in frame.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            records.append(_scale_record(group, scope, dict(zip(group_cols, keys))))
    return pd.DataFrame(records).sort_values(["scope", "row_count"], ascending=[True, False])


def _threshold_stability_assessment(
    threshold: pd.DataFrame,
    *,
    min_total_trades: int,
    min_positive_markets: int,
    min_positive_folds: int,
) -> dict[str, Any]:
    overall = threshold[threshold["scope"].eq("overall")].iloc[0]
    markets = threshold[threshold["scope"].eq("market")]
    folds = threshold[threshold["scope"].eq("fold")]
    positive_markets = markets[markets["net_return_dollars"].astype(float).gt(0.0)]
    positive_folds = folds[folds["net_return_dollars"].astype(float).gt(0.0)]
    total_trades = int(overall["trade_count"])
    stable = (
        total_trades >= min_total_trades
        and len(positive_markets) >= min_positive_markets
        and len(positive_folds) >= min_positive_folds
        and float(overall["net_return_dollars"]) > 0.0
        and float(overall["gross_return_dollars"]) > 0.0
    )
    reasons: list[str] = []
    if total_trades < min_total_trades:
        reasons.append(f"only {total_trades} trades below minimum {min_total_trades}")
    if len(positive_markets) < min_positive_markets:
        reasons.append(
            f"only {len(positive_markets)} net-positive markets below minimum {min_positive_markets}"
        )
    if len(positive_folds) < min_positive_folds:
        reasons.append(f"only {len(positive_folds)} net-positive folds below minimum {min_positive_folds}")
    if float(overall["net_return_dollars"]) <= 0.0:
        reasons.append("overall net is not positive")
    if float(overall["gross_return_dollars"]) <= 0.0:
        reasons.append("overall gross is not positive")
    return {
        "stable_threshold_region": stable,
        "total_trade_count": total_trades,
        "net_return_dollars": float(overall["net_return_dollars"]),
        "gross_return_dollars": float(overall["gross_return_dollars"]),
        "positive_market_count": int(len(positive_markets)),
        "market_count": int(len(markets)),
        "positive_fold_count": int(len(positive_folds)),
        "fold_count": int(len(folds)),
        "min_total_trades": min_total_trades,
        "min_positive_markets": min_positive_markets,
        "min_positive_folds": min_positive_folds,
        "instability_reasons": reasons,
    }


def _scale_assessment(scale: pd.DataFrame) -> dict[str, Any]:
    if "unavailable_reason" in scale.columns:
        return {"return_target_scale_status": "unavailable", "details": scale.iloc[0].to_dict()}
    overall = scale[scale["scope"].eq("overall")].iloc[0]
    flagged = scale[scale["scale_warnings"].fillna("").astype(str).ne("")]
    return {
        "return_target_scale_status": "flagged" if not flagged.empty else "ok",
        "overall": overall.to_dict(),
        "flagged_scope_count": int(len(flagged)),
        "flagged_examples": flagged.head(10).to_dict(orient="records"),
    }


def _top_findings(
    *,
    threshold_assessment: Mapping[str, Any],
    scale_assessment: Mapping[str, Any],
    direction_margin_threshold: float,
    min_fade_success: float,
    max_trend_danger: float,
) -> list[str]:
    findings = [
        (
            "Threshold scenario "
            f"margin={direction_margin_threshold:.2f}, fade>={min_fade_success:.2f}, "
            f"trend<{max_trend_danger:.2f} produced "
            f"{threshold_assessment['total_trade_count']} trades and net "
            f"{threshold_assessment['net_return_dollars']:.2f}."
        )
    ]
    if threshold_assessment["stable_threshold_region"]:
        findings.append("Threshold region passed the configured stability screen.")
    else:
        findings.append(
            "Threshold region failed stability screen: "
            + "; ".join(threshold_assessment["instability_reasons"])
        )
    if scale_assessment["return_target_scale_status"] == "flagged":
        overall = scale_assessment["overall"]
        findings.append(
            "Return model scale is flagged: std ratio "
            f"{float(overall['prediction_to_target_std_ratio']):.2f}, "
            f"p95 ratio {float(overall['prediction_to_target_abs_p95_ratio']):.2f}."
        )
    elif scale_assessment["return_target_scale_status"] == "unavailable":
        findings.append("Return target scale audit was unavailable.")
    else:
        findings.append("Return target scale audit did not flag an extreme ratio.")
    if not threshold_assessment["stable_threshold_region"]:
        findings.append("Decision: stop policy tuning and audit labels/features before further WFA.")
    else:
        findings.append("Decision: keep threshold as hypothesis only and test untouched holdout later.")
    findings.append("Return-magnitude logic should remain untrusted until ridge_return_v1 scale is explained.")
    return findings[:5]


def build_threshold_and_target_sanity(
    *,
    predictions_path: Path,
    costs_config: Path,
    output_root: Path,
    run: str,
    direction_margin_threshold: float,
    min_fade_success: float,
    max_trend_danger: float,
    min_total_trades: int,
    min_positive_markets: int,
    min_positive_folds: int,
) -> dict[str, Any]:
    if not predictions_path.exists():
        raise SystemExit(f"prediction parquet missing: {_relative_path(predictions_path)}")
    predictions = pd.read_parquet(predictions_path)
    policy_frame, failures, warnings = build_policy_frame(
        predictions,
        costs_config,
        PolicyConfig(
            long_short_margin=0.05,
            min_fade_success=0.50,
            max_trend_danger=0.50,
        ),
    )
    if failures:
        raise SystemExit("; ".join(failures))

    scenario = _scenario_frame(
        policy_frame,
        direction_margin_threshold=direction_margin_threshold,
        min_fade_success=min_fade_success,
        max_trend_danger=max_trend_danger,
    )
    threshold = _threshold_stability_rows(scenario)
    scale = _return_target_scale_audit(predictions)
    threshold_assessment = _threshold_stability_assessment(
        threshold,
        min_total_trades=min_total_trades,
        min_positive_markets=min_positive_markets,
        min_positive_folds=min_positive_folds,
    )
    scale_status = _scale_assessment(scale)
    next_action = (
        "freeze_threshold_as_hypothesis_for_untouched_holdout"
        if threshold_assessment["stable_threshold_region"]
        else "stop_policy_work_and_audit_labels_features"
    )
    if scale_status["return_target_scale_status"] == "flagged":
        return_action = "fix_or_explain_ridge_return_scale_before_return_magnitude_logic"
    else:
        return_action = "return_scale_not_blocking_from_this_audit"
    paths = _output_paths(output_root, run)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["threshold_stability"], threshold)
    _write_csv(paths["return_target_scale"], scale)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(policy_frame)),
        "threshold_scenario": {
            "direction_margin_threshold": direction_margin_threshold,
            "min_fade_success": min_fade_success,
            "max_trend_danger": max_trend_danger,
        },
        "threshold_stability": threshold_assessment,
        "return_target_scale": scale_status,
        "next_action": next_action,
        "return_target_action": return_action,
        "top_findings": _top_findings(
            threshold_assessment=threshold_assessment,
            scale_assessment=scale_status,
            direction_margin_threshold=direction_margin_threshold,
            min_fade_success=min_fade_success,
            max_trend_danger=max_trend_danger,
        ),
        "warnings": warnings,
        "outputs": {key: _relative_path(path) for key, path in paths.items()},
    }
    _write_json(paths["summary"], summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS.as_posix())
    parser.add_argument("--costs-config", default=DEFAULT_COSTS_CONFIG.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--direction-margin-threshold", type=float, default=0.30)
    parser.add_argument("--min-fade-success", type=float, default=0.50)
    parser.add_argument("--max-trend-danger", type=float, default=0.50)
    parser.add_argument("--min-total-trades", type=int, default=100)
    parser.add_argument("--min-positive-markets", type=int, default=2)
    parser.add_argument("--min-positive-folds", type=int, default=4)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_threshold_and_target_sanity(
        predictions_path=Path(args.predictions),
        costs_config=Path(args.costs_config),
        output_root=Path(args.output_root),
        run=args.run,
        direction_margin_threshold=args.direction_margin_threshold,
        min_fade_success=args.min_fade_success,
        max_trend_danger=args.max_trend_danger,
        min_total_trades=args.min_total_trades,
        min_positive_markets=args.min_positive_markets,
        min_positive_folds=args.min_positive_folds,
    )
    stability = summary["threshold_stability"]
    print(
        "PASS threshold and target sanity: "
        f"trades={stability['total_trade_count']} "
        f"net_dollars={stability['net_return_dollars']} "
        f"stable={stability['stable_threshold_region']} "
        f"next_action={summary['next_action']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
