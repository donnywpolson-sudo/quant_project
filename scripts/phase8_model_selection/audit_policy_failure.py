#!/usr/bin/env python3
"""Write compact failure-breakdown diagnostics for Phase 8 policy results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.phase8_model_selection.evaluate_predictions import (
    DEFAULT_COSTS_CONFIG,
    PolicyConfig,
    _direction_accuracy,
    _mean_float,
    _policy_summary,
    _score_column_for_target,
    _std_float,
    build_policy_frame,
)


DEFAULT_RUN = "tier1_locked_baseline"
DEFAULT_PREDICTIONS = Path("data/predictions/tier1_locked_baseline/oos_predictions.parquet")
DEFAULT_OUTPUT_ROOT = Path("reports/phase8_failure_breakdown")

POSITION_LABELS = {
    -1: "short",
    0: "flat",
    1: "long",
}


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _group_summary(
    policy_frame: pd.DataFrame,
    *,
    scope: str,
    group_cols: list[str],
    sort_by_net: bool = True,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in policy_frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))
        records.append(_policy_summary(group, scope, key_values))
    frame = pd.DataFrame(records)
    if frame.empty or not sort_by_net:
        return frame
    return frame.sort_values("net_return_dollars", ascending=True).reset_index(drop=True)


def _direction_summary(policy_frame: pd.DataFrame) -> pd.DataFrame:
    frame = policy_frame.copy()
    frame["position_label"] = frame["position"].map(POSITION_LABELS).fillna("unknown")
    return _group_summary(
        frame,
        scope="final_position",
        group_cols=["position", "position_label"],
        sort_by_net=True,
    )


def _cost_components(summary: Mapping[str, Any]) -> pd.DataFrame:
    cost = float(summary.get("cost_dollars") or 0.0)
    slippage = float(summary.get("slippage_cost_dollars") or 0.0)
    commission = float(summary.get("commission_cost_dollars") or 0.0)
    gross = float(summary.get("gross_return_dollars") or 0.0)
    return pd.DataFrame(
        [
            {
                "gross_return_dollars": gross,
                "slippage_cost_dollars": slippage,
                "commission_cost_dollars": commission,
                "cost_dollars": cost,
                "net_return_dollars": float(summary.get("net_return_dollars") or 0.0),
                "cost_drag_to_abs_gross": cost / abs(gross) if abs(gross) > 0.0 else None,
                "slippage_share_of_cost": slippage / cost if cost > 0.0 else None,
                "commission_share_of_cost": commission / cost if cost > 0.0 else None,
            }
        ]
    )


def _prediction_metric_record(keys: tuple[Any, ...], group: pd.DataFrame) -> dict[str, Any]:
    model_id, model_family, target_name = keys
    y_true = pd.to_numeric(group["y_true"], errors="coerce")
    score = _score_column_for_target(group, str(target_name))
    record: dict[str, Any] = {
        "model_id": model_id,
        "model_family": model_family,
        "target_name": target_name,
        "row_count": int(len(group)),
        "market_count": int(group["market"].nunique(dropna=True)),
        "fold_count": int(group["fold_id"].nunique(dropna=True)),
        "prediction_type": group["prediction_type"].dropna().astype(str).iloc[0]
        if group["prediction_type"].notna().any()
        else None,
        "prediction_mean": _mean_float(score),
        "prediction_std": _std_float(score),
        "target_mean": _mean_float(y_true),
        "target_std": _std_float(y_true),
    }
    aligned = pd.DataFrame({"y_true": y_true, "score": score}).dropna()
    if not aligned.empty:
        errors = aligned["score"] - aligned["y_true"]
        record["mse"] = float(np.mean(np.square(errors)))
        record["mae"] = float(np.mean(np.abs(errors)))
        target_values = set(aligned["y_true"].unique().tolist())
        if target_values.issubset({0, 1}) and aligned["score"].between(0.0, 1.0).all():
            record["brier_score"] = float(np.mean(np.square(errors)))
    if target_name == "target_sign_with_deadzone":
        record["direction_accuracy"] = _direction_accuracy(group)
    return record


def _model_target_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_id", "model_family", "target_name"]
    records = [
        _prediction_metric_record(keys, group)
        for keys, group in predictions.groupby(group_cols, dropna=False)
    ]
    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)


def build_failure_breakdown(
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

    overall = _policy_summary(policy_frame, "overall", {})
    by_market = _group_summary(policy_frame, scope="market", group_cols=["market"])
    by_fold = _group_summary(policy_frame, scope="fold", group_cols=["fold_id"])
    by_market_fold = _group_summary(
        policy_frame,
        scope="market_fold",
        group_cols=["market", "fold_id"],
    )
    by_direction = _direction_summary(policy_frame)
    by_reason = _group_summary(
        policy_frame,
        scope="policy_reason",
        group_cols=["policy_reason"],
    )
    model_target = _model_target_summary(predictions)
    cost_components = _cost_components(overall)

    output_root.mkdir(parents=True, exist_ok=True)
    files = {
        "by_market": output_root / f"{run}_by_market.csv",
        "by_fold": output_root / f"{run}_by_fold.csv",
        "by_market_fold": output_root / f"{run}_by_market_fold.csv",
        "by_direction": output_root / f"{run}_by_direction.csv",
        "by_policy_reason": output_root / f"{run}_by_policy_reason.csv",
        "model_target": output_root / f"{run}_model_target_summary.csv",
        "cost_components": output_root / f"{run}_cost_components.csv",
        "summary": output_root / f"{run}_summary.json",
    }
    _write_csv(files["by_market"], by_market)
    _write_csv(files["by_fold"], by_fold)
    _write_csv(files["by_market_fold"], by_market_fold)
    _write_csv(files["by_direction"], by_direction)
    _write_csv(files["by_policy_reason"], by_reason)
    _write_csv(files["model_target"], model_target)
    _write_csv(files["cost_components"], cost_components)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "output_root": _relative_path(output_root),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(policy_frame)),
        "trade_count": int(policy_frame["trade_count"].sum()),
        "overall": overall,
        "worst_markets_by_net": by_market.head(10).to_dict(orient="records"),
        "worst_folds_by_net": by_fold.head(10).to_dict(orient="records"),
        "cost_components": cost_components.iloc[0].to_dict(),
        "warnings": warnings,
        "outputs": {key: _relative_path(path) for key, path in files.items()},
    }
    _write_json(files["summary"], payload)
    return payload


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
    report = build_failure_breakdown(
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
    overall = report["overall"]
    print(
        "PASS policy failure breakdown: "
        f"rows={report['policy_row_count']} trades={report['trade_count']} "
        f"net_dollars={overall['net_return_dollars']} "
        f"outputs={report['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
