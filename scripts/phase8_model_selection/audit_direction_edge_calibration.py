#!/usr/bin/env python3
"""Audit direction probability calibration and edge construction."""

from __future__ import annotations

import argparse
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
from scripts.phase8_model_selection.audit_trade_failure_drilldown import _empty_unavailable
from scripts.phase8_model_selection.evaluate_predictions import (
    DEFAULT_COSTS_CONFIG,
    PolicyConfig,
    build_policy_frame,
)


OUTPUT_SUFFIXES = {
    "summary": "direction_edge_calibration_summary.json",
    "class_calibration": "direction_class_calibration.csv",
    "confidence_bins": "direction_confidence_bins.csv",
    "edge_scenarios": "direction_edge_scenarios.csv",
    "scenario_market_side": "direction_edge_scenario_market_side.csv",
    "flat_suppression": "direction_flat_suppression.csv",
    "readme": "direction_edge_calibration_readme.md",
}


def _output_paths(output_root: Path, run: str) -> dict[str, Path]:
    return {key: output_root / f"{run}_{suffix}" for key, suffix in OUTPUT_SUFFIXES.items()}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _sum_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _direction_accuracy(position: pd.Series, actual: pd.Series) -> float | None:
    aligned = pd.DataFrame({"position": position, "actual": actual}).dropna()
    aligned = aligned[aligned["position"].ne(0) & aligned["actual"].isin([-1, 0, 1])]
    if aligned.empty:
        return None
    return float(aligned["position"].eq(aligned["actual"]).mean())


def _argmax_direction(frame: pd.DataFrame) -> pd.Series:
    probs = np.vstack(
        [
            _numeric(frame, "p_short").fillna(-np.inf).to_numpy(),
            _numeric(frame, "p_flat").fillna(-np.inf).to_numpy(),
            _numeric(frame, "p_long").fillna(-np.inf).to_numpy(),
        ]
    ).T
    return pd.Series(np.array([-1, 0, 1])[np.argmax(probs, axis=1)], index=frame.index)


def _realized_direction(frame: pd.DataFrame) -> pd.Series:
    move = _numeric(frame, "execution_close") - _numeric(frame, "execution_open")
    return pd.Series(np.sign(move.to_numpy(dtype=float)), index=frame.index, dtype=float)


def _max_direction_probability(frame: pd.DataFrame) -> pd.Series:
    return pd.concat([_numeric(frame, "p_long"), _numeric(frame, "p_short")], axis=1).max(axis=1)


def _side_probability(frame: pd.DataFrame, position: pd.Series) -> pd.Series:
    p_long = _numeric(frame, "p_long")
    p_short = _numeric(frame, "p_short")
    return pd.Series(np.where(position.eq(1), p_long, np.where(position.eq(-1), p_short, np.nan)), index=frame.index)


def _bucket(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels).astype("string").fillna("missing")


def _attach_fields(policy_frame: pd.DataFrame) -> pd.DataFrame:
    frame = policy_frame.copy()
    frame["target_direction"] = _numeric(frame, "observed_direction_target")
    frame["realized_direction"] = _realized_direction(frame)
    frame["argmax_direction"] = _argmax_direction(frame)
    frame["argmax_side"] = frame["argmax_direction"].map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    frame["argmax_probability"] = pd.concat(
        [_numeric(frame, "p_short"), _numeric(frame, "p_flat"), _numeric(frame, "p_long")],
        axis=1,
    ).max(axis=1)
    frame["max_direction_probability"] = _max_direction_probability(frame)
    frame["direction_margin_abs"] = _numeric(frame, "direction_margin").abs()
    frame["direction_vs_flat_edge"] = frame["max_direction_probability"] - _numeric(frame, "p_flat")
    frame["side_probability_current"] = _side_probability(frame, frame["base_position"])
    frame["side_probability_traded"] = _side_probability(frame, frame["position"])
    frame["argmax_probability_bucket"] = _bucket(
        frame["argmax_probability"],
        [-np.inf, 0.34, 0.40, 0.50, 0.60, 0.70, 0.85, np.inf],
        ["<=0.34", "0.34-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.85", ">0.85"],
    )
    frame["direction_margin_bucket"] = _bucket(
        frame["direction_margin_abs"],
        [-np.inf, 0.05, 0.10, 0.20, 0.40, np.inf],
        ["<=0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">0.40"],
    )
    frame["flat_probability_bucket"] = _bucket(
        _numeric(frame, "p_flat"),
        [-np.inf, 0.25, 0.35, 0.45, 0.55, 0.70, np.inf],
        ["<=0.25", "0.25-0.35", "0.35-0.45", "0.45-0.55", "0.55-0.70", ">0.70"],
    )
    frame["direction_vs_flat_bucket"] = _bucket(
        frame["direction_vs_flat_edge"],
        [-np.inf, -0.20, -0.10, 0.0, 0.05, 0.10, 0.20, np.inf],
        ["<=-0.20", "-0.20--0.10", "-0.10-0.00", "0.00-0.05", "0.05-0.10", "0.10-0.20", ">0.20"],
    )
    return frame


def _scenario_position(
    frame: pd.DataFrame,
    *,
    mode: str,
    margin_threshold: float,
    flat_margin: float | None,
    max_flat_probability: float | None,
) -> pd.Series:
    margin = _numeric(frame, "direction_margin")
    p_long = _numeric(frame, "p_long")
    p_short = _numeric(frame, "p_short")
    p_flat = _numeric(frame, "p_flat")
    position = pd.Series(0, index=frame.index, dtype=int)
    if mode == "current_margin":
        position.loc[margin.ge(margin_threshold)] = 1
        position.loc[margin.le(-margin_threshold)] = -1
    elif mode == "direction_beats_flat":
        flat_edge = 0.0 if flat_margin is None else flat_margin
        position.loc[p_long.sub(p_short).ge(margin_threshold) & p_long.sub(p_flat).ge(flat_edge)] = 1
        position.loc[p_short.sub(p_long).ge(margin_threshold) & p_short.sub(p_flat).ge(flat_edge)] = -1
    elif mode == "argmax_nonflat":
        argmax = _argmax_direction(frame)
        position.loc[argmax.eq(1) & margin.ge(margin_threshold)] = 1
        position.loc[argmax.eq(-1) & margin.le(-margin_threshold)] = -1
    else:
        raise ValueError(f"unknown edge mode: {mode}")
    if max_flat_probability is not None:
        position = position.where(p_flat.le(max_flat_probability), 0)
    gate_ok = frame["fade_allowed"].fillna(False) & ~frame["trend_danger_block"].fillna(True)
    return position.where(gate_ok, 0).astype(int)


def _scenario_record(
    frame: pd.DataFrame,
    *,
    mode: str,
    margin_threshold: float,
    flat_margin: float | None,
    max_flat_probability: float | None,
    group_keys: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    position = _scenario_position(
        frame,
        mode=mode,
        margin_threshold=margin_threshold,
        flat_margin=flat_margin,
        max_flat_probability=max_flat_probability,
    )
    gross = position * _numeric(frame, "price_move").fillna(0.0) * _numeric(frame, "point_value").fillna(0.0)
    cost = pd.Series(
        np.where(position.ne(0), _numeric(frame, "round_turn_cost_dollars").fillna(0.0), 0.0),
        index=frame.index,
    )
    net = gross - cost
    traded = position.ne(0)
    record: dict[str, Any] = {
        "scope": "scenario" if group_keys is None else "scenario_group",
        **dict(group_keys or {}),
        "edge_mode": mode,
        "direction_margin_threshold": margin_threshold,
        "flat_margin_threshold": flat_margin,
        "max_flat_probability": max_flat_probability,
        "row_count": int(len(frame)),
        "trade_count": int(traded.sum()),
        "long_count": int(position.eq(1).sum()),
        "short_count": int(position.eq(-1).sum()),
        "gross_return_dollars": _sum_float(gross),
        "cost_dollars": _sum_float(cost),
        "net_return_dollars": _sum_float(net),
        "target_direction_accuracy": _direction_accuracy(position[traded], frame.loc[traded, "target_direction"]),
        "realized_direction_accuracy": _direction_accuracy(position[traded], frame.loc[traded, "realized_direction"]),
        "avg_p_flat_traded": _mean_or_none(_numeric(frame.loc[traded], "p_flat")) if bool(traded.any()) else None,
        "avg_direction_margin_abs_traded": _mean_or_none(frame.loc[traded, "direction_margin_abs"]) if bool(traded.any()) else None,
        "avg_direction_vs_flat_edge_traded": _mean_or_none(frame.loc[traded, "direction_vs_flat_edge"]) if bool(traded.any()) else None,
    }
    return record


def _edge_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    specs: list[tuple[str, float, float | None, float | None]] = []
    for margin_threshold in (0.05, 0.10, 0.15, 0.20, 0.30):
        specs.append(("current_margin", margin_threshold, None, None))
        for max_flat in (0.35, 0.45, 0.55):
            specs.append(("current_margin", margin_threshold, None, max_flat))
        for flat_margin in (0.00, 0.05, 0.10, 0.20):
            specs.append(("direction_beats_flat", margin_threshold, flat_margin, None))
        specs.append(("argmax_nonflat", margin_threshold, None, None))
    records = [
        _scenario_record(
            frame,
            mode=mode,
            margin_threshold=margin_threshold,
            flat_margin=flat_margin,
            max_flat_probability=max_flat,
        )
        for mode, margin_threshold, flat_margin, max_flat in specs
    ]
    return pd.DataFrame(records).sort_values(
        ["net_return_dollars", "trade_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _scenario_market_side(frame: pd.DataFrame, best: Mapping[str, Any]) -> pd.DataFrame:
    if not best:
        return _empty_unavailable("no best scenario available")
    position = _scenario_position(
        frame,
        mode=str(best["edge_mode"]),
        margin_threshold=float(best["direction_margin_threshold"]),
        flat_margin=None if pd.isna(best.get("flat_margin_threshold")) else float(best["flat_margin_threshold"]),
        max_flat_probability=None if pd.isna(best.get("max_flat_probability")) else float(best["max_flat_probability"]),
    )
    work = frame.copy()
    work["scenario_position"] = position
    work["scenario_side"] = position.map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    records = [
        _scenario_record(
            group,
            mode=str(best["edge_mode"]),
            margin_threshold=float(best["direction_margin_threshold"]),
            flat_margin=None if pd.isna(best.get("flat_margin_threshold")) else float(best["flat_margin_threshold"]),
            max_flat_probability=None if pd.isna(best.get("max_flat_probability")) else float(best["max_flat_probability"]),
            group_keys={"market": market, "scenario_side": side},
        )
        for (market, side), group in work[work["scenario_position"].ne(0)].groupby(["market", "scenario_side"], dropna=False)
    ]
    if not records:
        return _empty_unavailable("best scenario has no trades")
    return pd.DataFrame(records).sort_values("net_return_dollars").reset_index(drop=True)


def _class_calibration(direction_rows: pd.DataFrame) -> pd.DataFrame:
    frame = direction_rows.copy()
    frame["target_direction"] = _numeric(frame, "y_true")
    frame["argmax_direction"] = _argmax_direction(frame)
    frame["argmax_probability"] = pd.concat(
        [_numeric(frame, "p_short"), _numeric(frame, "p_flat"), _numeric(frame, "p_long")],
        axis=1,
    ).max(axis=1)
    frame["argmax_probability_bucket"] = _bucket(
        frame["argmax_probability"],
        [-np.inf, 0.34, 0.40, 0.50, 0.60, 0.70, 0.85, np.inf],
        ["<=0.34", "0.34-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.85", ">0.85"],
    )
    records: list[dict[str, Any]] = []
    for (predicted, bucket), group in frame.groupby(["argmax_direction", "argmax_probability_bucket"], dropna=False):
        actual = _numeric(group, "target_direction")
        records.append(
            {
                "predicted_class": int(predicted) if pd.notna(predicted) else None,
                "argmax_probability_bucket": bucket,
                "row_count": int(len(group)),
                "avg_argmax_probability": _mean_or_none(group["argmax_probability"]),
                "empirical_predicted_class_rate": float(actual.eq(predicted).mean()) if len(group) else None,
                "empirical_long_rate": float(actual.eq(1).mean()) if len(group) else None,
                "empirical_flat_rate": float(actual.eq(0).mean()) if len(group) else None,
                "empirical_short_rate": float(actual.eq(-1).mean()) if len(group) else None,
                "avg_p_long": _mean_or_none(group["p_long"]),
                "avg_p_flat": _mean_or_none(group["p_flat"]),
                "avg_p_short": _mean_or_none(group["p_short"]),
            }
        )
    return pd.DataFrame(records).sort_values(["predicted_class", "argmax_probability_bucket"]).reset_index(drop=True)


def _confidence_bins(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["direction_margin_bucket", "flat_probability_bucket", "argmax_side"]
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        margin_bucket, flat_bucket, argmax_side = keys
        records.append(
            {
                "direction_margin_bucket": margin_bucket,
                "flat_probability_bucket": flat_bucket,
                "argmax_side": argmax_side,
                "row_count": int(len(group)),
                "base_signal_count": int(group["base_position"].ne(0).sum()),
                "trade_count": int(group["position"].ne(0).sum()),
                "argmax_accuracy": _direction_accuracy(group["argmax_direction"], group["target_direction"]),
                "base_signal_accuracy": _direction_accuracy(group["base_position"], group["target_direction"]),
                "traded_accuracy": _direction_accuracy(
                    group.loc[group["position"].ne(0), "position"],
                    group.loc[group["position"].ne(0), "target_direction"],
                ),
                "avg_p_long": _mean_or_none(group["p_long"]),
                "avg_p_flat": _mean_or_none(group["p_flat"]),
                "avg_p_short": _mean_or_none(group["p_short"]),
                "avg_direction_vs_flat_edge": _mean_or_none(group["direction_vs_flat_edge"]),
                "net_return_dollars": _sum_float(group["net_dollars"]),
            }
        )
    return pd.DataFrame(records).sort_values("net_return_dollars").reset_index(drop=True)


def _flat_suppression(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for max_flat in (0.25, 0.35, 0.45, 0.55, 0.70):
        subset = frame[frame["p_flat"].le(max_flat)]
        current_trades = subset[subset["position"].ne(0)]
        records.append(
            {
                "max_flat_probability": max_flat,
                "eligible_row_count": int(len(subset)),
                "current_trade_count": int(len(current_trades)),
                "current_net_dollars": _sum_float(current_trades["net_dollars"]),
                "current_direction_accuracy": _direction_accuracy(
                    current_trades["position"], current_trades["target_direction"]
                ),
                "current_realized_accuracy": _direction_accuracy(
                    current_trades["position"], current_trades["realized_direction"]
                ),
                "avg_direction_margin_abs": _mean_or_none(current_trades["direction_margin_abs"])
                if not current_trades.empty
                else None,
                "avg_direction_vs_flat_edge": _mean_or_none(current_trades["direction_vs_flat_edge"])
                if not current_trades.empty
                else None,
            }
        )
    return pd.DataFrame(records)


def _best_positive_scenario(edge_scenarios: pd.DataFrame) -> Mapping[str, Any] | None:
    positive = edge_scenarios[
        edge_scenarios["gross_return_dollars"].gt(0.0)
        & edge_scenarios["net_return_dollars"].gt(0.0)
        & edge_scenarios["trade_count"].ge(100)
    ]
    if positive.empty:
        return None
    return positive.sort_values("net_return_dollars", ascending=False).iloc[0].to_dict()


def _decision(edge_scenarios: pd.DataFrame, overall_current: Mapping[str, Any], class_calibration: pd.DataFrame) -> str:
    best = _best_positive_scenario(edge_scenarios)
    current_acc = overall_current.get("target_direction_accuracy")
    if best is not None:
        return "flat_aware_edge_candidate_found_needs_oos_stability_check"
    argmax_rows = class_calibration[class_calibration["predicted_class"].isin([-1, 1])]
    empirical = _mean_or_none(argmax_rows["empirical_predicted_class_rate"]) if not argmax_rows.empty else None
    if current_acc is not None and float(current_acc) < 0.50 and empirical is not None and empirical >= 0.50:
        return "edge_threshold_miscalibrated_relative_to_class_probabilities"
    return "direction_probabilities_not_tradeable_without_new_edge_model"


def _top_findings(
    *,
    edge_scenarios: pd.DataFrame,
    current: Mapping[str, Any],
    flat_suppression: pd.DataFrame,
    best: Mapping[str, Any] | None,
) -> list[str]:
    findings = [
        "Current edge trades "
        f"{int(current.get('trade_count') or 0)} rows with net "
        f"{float(current.get('net_return_dollars') or 0.0):.2f} and target accuracy "
        f"{float(current.get('target_direction_accuracy') or 0.0):.4f}.",
    ]
    if best is None:
        top = edge_scenarios.iloc[0]
        findings.append(
            "No tested flat-aware scenario reached positive net with at least 100 trades; "
            f"best net was {float(top['net_return_dollars']):.2f}."
        )
    else:
        findings.append(
            f"Best positive flat-aware scenario used {best['edge_mode']} with "
            f"{int(best['trade_count'])} trades and net {float(best['net_return_dollars']):.2f}."
        )
    if not flat_suppression.empty:
        top_flat = flat_suppression.sort_values("current_net_dollars", ascending=False).iloc[0]
        findings.append(
            f"Best simple p_flat cap was {float(top_flat['max_flat_probability']):.2f}, "
            f"current-trade net {float(top_flat['current_net_dollars']):.2f}."
        )
    top_scenario = edge_scenarios.iloc[0]
    findings.append(
        f"Best tested scenario overall was {top_scenario['edge_mode']} "
        f"margin {float(top_scenario['direction_margin_threshold']):.2f}, "
        f"net {float(top_scenario['net_return_dollars']):.2f}."
    )
    findings.append("This audit is diagnostic only; no policy thresholds were changed.")
    return findings[:5]


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    findings = "\n".join(f"- {item}" for item in summary["top_findings"])
    files = "\n".join(
        f"- `{value}`" for key, value in sorted(summary["outputs"].items()) if key != "readme"
    )
    text = f"""# Phase 8 Direction Edge Calibration

Run: `{summary['run']}`

This diagnostic audits direction probabilities and alternative edge
construction rules, including flat-probability suppression. It is read-only and
does not change model outputs, calibration, thresholds, or policy behavior.

## Top Findings

{findings}

## Files

{files}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_direction_edge_calibration(
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
    frame = _attach_fields(policy_frame)
    direction_rows = predictions[predictions["target_name"].eq("target_sign_with_deadzone")].copy()
    if direction_rows.empty:
        raise SystemExit("missing target_sign_with_deadzone prediction rows")

    class_calibration = _class_calibration(direction_rows)
    confidence_bins = _confidence_bins(frame)
    edge_scenarios = _edge_scenarios(frame)
    best = _best_positive_scenario(edge_scenarios)
    scenario_market_side = _scenario_market_side(frame, edge_scenarios.iloc[0].to_dict())
    flat_suppression = _flat_suppression(frame)
    current = _scenario_record(
        frame,
        mode="current_margin",
        margin_threshold=policy.long_short_margin,
        flat_margin=None,
        max_flat_probability=None,
    )
    decision = _decision(edge_scenarios, current, class_calibration)

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, run)
    _write_csv(paths["class_calibration"], class_calibration)
    _write_csv(paths["confidence_bins"], confidence_bins)
    _write_csv(paths["edge_scenarios"], edge_scenarios)
    _write_csv(paths["scenario_market_side"], scenario_market_side)
    _write_csv(paths["flat_suppression"], flat_suppression)
    outputs = {key: _relative_path(path) for key, path in paths.items()}
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(frame)),
        "current_edge": current,
        "best_edge_scenarios": edge_scenarios.head(10).to_dict(orient="records"),
        "best_positive_scenario_min_100_trades": best,
        "flat_suppression": flat_suppression.to_dict(orient="records"),
        "decision": decision,
        "top_findings": _top_findings(
            edge_scenarios=edge_scenarios,
            current=current,
            flat_suppression=flat_suppression,
            best=best,
        ),
        "warnings": warnings,
        "outputs": outputs,
    }
    _write_json(paths["summary"], summary)
    _write_readme(paths["readme"], summary)
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
    summary = build_direction_edge_calibration(
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
    current = summary["current_edge"]
    print(
        "PASS direction edge calibration: "
        f"rows={summary['policy_row_count']} "
        f"current_trades={current['trade_count']} "
        f"current_net={current['net_return_dollars']} "
        f"decision={summary['decision']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
