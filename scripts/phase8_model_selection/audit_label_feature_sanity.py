#!/usr/bin/env python3
"""Audit Phase 8 predictions against the underlying feature/label matrix."""

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


DEFAULT_FEATURE_ROOT = Path("data/feature_matrices/baseline")
RETURN_TARGET = "target_ret_15m"
RETURN_MODEL = "ridge_return_v1"
FEATURE_MATCH_TOLERANCE = 1e-12

TARGET_COLUMNS = [
    "target_ret_15m",
    "target_ret_ticks_15m",
    "target_net_ticks_after_est_cost",
    "target_gross_dollars_15m",
    "target_sign_15m",
    "target_sign_with_deadzone",
    "target_tradeable_after_cost",
    "target_valid",
    "target_entry_price",
    "target_exit_price",
    "target_entry_ts",
    "target_exit_ts",
]
BASE_FEATURE_COLUMNS = [
    "ts",
    "market",
    "year",
    "close",
    "causal_valid",
    "feature_input_valid",
    "feature_row_valid",
    "training_row_valid",
]


def _output_paths(output_root: Path, run: str) -> dict[str, Path]:
    return {
        "summary": output_root / f"{run}_label_feature_sanity_summary.json",
        "target_alignment": output_root / f"{run}_target_alignment.csv",
        "label_balance": output_root / f"{run}_label_balance_by_market_fold.csv",
        "feature_quality": output_root / f"{run}_feature_quality_by_scope.csv",
        "feature_shift": output_root / f"{run}_feature_shift_top.csv",
        "return_scale": output_root / f"{run}_return_training_scale_check.csv",
        "readme": output_root / f"{run}_label_feature_sanity_readme.md",
    }


def _read_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _read_schema(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _std_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=0)) if not values.empty else None


def _quantile_or_none(series: pd.Series, q: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else None


def _match_rate(left: pd.Series, right: pd.Series, *, tolerance: float | None = None) -> float | None:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    if aligned.empty:
        return None
    if tolerance is None:
        return float(aligned["left"].eq(aligned["right"]).mean())
    return float((aligned["left"] - aligned["right"]).abs().le(tolerance).mean())


def _realized_return(frame: pd.DataFrame) -> pd.Series:
    entry = _numeric(frame, "execution_open")
    exit_ = _numeric(frame, "execution_close")
    return (exit_ / entry) - 1.0


def _realized_direction(frame: pd.DataFrame) -> pd.Series:
    move = _numeric(frame, "execution_close") - _numeric(frame, "execution_open")
    return np.sign(move).astype(float)


def _policy_base(predictions: pd.DataFrame, costs_config: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    policy = PolicyConfig(
        long_short_margin=0.05,
        min_fade_success=0.50,
        max_trend_danger=0.50,
    )
    policy_frame, failures, warnings = build_policy_frame(predictions, costs_config, policy)
    if failures:
        raise SystemExit("; ".join(failures))
    policy_frame["timestamp"] = pd.to_datetime(policy_frame["timestamp"], utc=True, errors="coerce")
    policy_frame["side"] = policy_frame["position"].map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    return policy_frame, failures, warnings


def _read_feature_subset(
    *,
    feature_root: Path,
    market: str,
    year: int,
    timestamps: set[pd.Timestamp],
    columns: list[str],
    unavailable: list[str],
) -> pd.DataFrame:
    path = feature_root / market / f"{year}.parquet"
    if not path.exists():
        unavailable.append(f"feature matrix missing: {_relative_path(path)}")
        return pd.DataFrame()
    available = _read_schema(path)
    missing_keys = sorted({"ts", "market", "year"} - available)
    if missing_keys:
        raise SystemExit(f"feature matrix missing required keys {missing_keys}: {_relative_path(path)}")
    read_columns = [column for column in columns if column in available]
    missing_optional = sorted(set(columns) - available - {"ts", "market", "year"})
    if missing_optional:
        unavailable.append(
            f"{market} {year} missing optional feature/target columns: {','.join(missing_optional[:20])}"
        )
    frame = pd.read_parquet(path, columns=read_columns)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame[frame["ts"].isin(timestamps)].copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame.rename(columns={"ts": "timestamp"})
    return frame


def _load_matched_feature_rows(
    policy_frame: pd.DataFrame,
    *,
    feature_root: Path,
    feature_cols: list[str],
    unavailable: list[str],
) -> pd.DataFrame:
    requested_columns = sorted(set(BASE_FEATURE_COLUMNS + TARGET_COLUMNS + feature_cols))
    frames: list[pd.DataFrame] = []
    for (market, year), group in policy_frame.groupby(["market", "year"], dropna=False):
        if pd.isna(market) or pd.isna(year):
            continue
        timestamps = set(pd.to_datetime(group["timestamp"], utc=True, errors="coerce").dropna())
        if not timestamps:
            continue
        frames.append(
            _read_feature_subset(
                feature_root=feature_root,
                market=str(market),
                year=int(year),
                timestamps=timestamps,
                columns=requested_columns,
                unavailable=unavailable,
            )
        )
    if not frames:
        return policy_frame.copy()
    features = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if features.empty:
        out = policy_frame.copy()
        out["feature_matrix_matched"] = False
        return out
    features["feature_matrix_matched"] = True
    merged = policy_frame.merge(
        features,
        on=["market", "year", "timestamp"],
        how="left",
        suffixes=("", "_feature"),
    )
    merged["feature_matrix_matched"] = merged["feature_matrix_matched"].fillna(False).astype(bool)
    return merged


def _alignment_record(frame: pd.DataFrame, scope: str, keys: Mapping[str, Any]) -> dict[str, Any]:
    observed_return = _numeric(frame, "observed_return_target")
    feature_return = _numeric(frame, "target_ret_15m")
    realized_return = _realized_return(frame)
    observed_direction = _numeric(frame, "observed_direction_target")
    feature_direction = _numeric(frame, "target_sign_with_deadzone")
    realized_direction = _realized_direction(frame)
    feature_target_valid_col = "target_valid_feature" if "target_valid_feature" in frame else "target_valid"
    pred_feature_abs = (observed_return - feature_return).abs()
    feature_exec_abs = (feature_return - realized_return).abs()
    return {
        "scope": scope,
        **dict(keys),
        "row_count": int(len(frame)),
        "feature_matrix_matched_rows": int(frame.get("feature_matrix_matched", pd.Series(False, index=frame.index)).sum()),
        "trade_count": int(frame["trade_count"].sum()) if "trade_count" in frame else 0,
        "observed_return_mean": _mean_or_none(observed_return),
        "feature_target_ret_mean": _mean_or_none(feature_return),
        "realized_return_from_execution_mean": _mean_or_none(realized_return),
        "observed_return_std": _std_or_none(observed_return),
        "feature_target_ret_std": _std_or_none(feature_return),
        "realized_return_from_execution_std": _std_or_none(realized_return),
        "mean_abs_observed_vs_feature_return_diff": _mean_or_none(pred_feature_abs),
        "max_abs_observed_vs_feature_return_diff": _quantile_or_none(pred_feature_abs, 1.0),
        "mean_abs_feature_vs_execution_return_diff": _mean_or_none(feature_exec_abs),
        "max_abs_feature_vs_execution_return_diff": _quantile_or_none(feature_exec_abs, 1.0),
        "observed_feature_return_match_rate": _match_rate(
            observed_return,
            feature_return,
            tolerance=FEATURE_MATCH_TOLERANCE,
        ),
        "observed_feature_direction_match_rate": _match_rate(observed_direction, feature_direction),
        "observed_realized_sign_match_rate_nonzero": _match_rate(
            observed_direction[observed_direction.ne(0)],
            realized_direction[observed_direction.ne(0)],
        ),
        "feature_target_valid_rate": float(frame[feature_target_valid_col].astype("boolean").mean())
        if feature_target_valid_col in frame
        else None,
        "feature_input_valid_rate": float(frame["feature_input_valid"].astype("boolean").mean())
        if "feature_input_valid" in frame
        else None,
    }


def _target_alignment(merged: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("overall", []),
        ("market", ["market"]),
        ("fold", ["fold_id"]),
        ("market_fold", ["market", "fold_id"]),
    ]
    records: list[dict[str, Any]] = []
    for scope, cols in specs:
        if not cols:
            records.append(_alignment_record(merged, scope, {}))
            continue
        for keys, group in merged.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            records.append(_alignment_record(group, scope, dict(zip(cols, keys))))
    return pd.DataFrame(records).sort_values(["scope", "row_count"], ascending=[True, False])


def _label_balance(merged: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (market, fold_id), group in merged.groupby(["market", "fold_id"], dropna=False):
        label = _numeric(group, "target_sign_with_deadzone")
        traded = group[group["trade_count"].eq(1)]
        traded_label = _numeric(traded, "target_sign_with_deadzone")
        records.append(
            {
                "market": market,
                "fold_id": fold_id,
                "row_count": int(len(group)),
                "trade_count": int(len(traded)),
                "label_long_count": int(label.eq(1).sum()),
                "label_short_count": int(label.eq(-1).sum()),
                "label_flat_count": int(label.eq(0).sum()),
                "label_long_rate": float(label.eq(1).mean()) if len(group) else None,
                "label_short_rate": float(label.eq(-1).mean()) if len(group) else None,
                "label_flat_rate": float(label.eq(0).mean()) if len(group) else None,
                "traded_label_long_count": int(traded_label.eq(1).sum()),
                "traded_label_short_count": int(traded_label.eq(-1).sum()),
                "traded_label_flat_count": int(traded_label.eq(0).sum()),
                "position_long_count": int(group["position"].eq(1).sum()),
                "position_short_count": int(group["position"].eq(-1).sum()),
                "gross_return_dollars": float(group["gross_dollars"].sum()),
                "cost_dollars": float(group["cost_dollars"].sum()),
                "net_return_dollars": float(group["net_dollars"].sum()),
            }
        )
    return pd.DataFrame(records).sort_values("net_return_dollars").reset_index(drop=True)


def _feature_stats(merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    traded = merged[merged["trade_count"].eq(1)]
    for column in feature_cols:
        if column not in merged.columns:
            continue
        all_values = pd.to_numeric(merged[column], errors="coerce")
        traded_values = pd.to_numeric(traded[column], errors="coerce") if not traded.empty else pd.Series(dtype=float)
        all_mean = _mean_or_none(all_values)
        all_std = _std_or_none(all_values)
        traded_mean = _mean_or_none(traded_values)
        standardized_shift = (
            abs(float(traded_mean) - float(all_mean)) / float(all_std)
            if all_mean is not None and traded_mean is not None and all_std not in (None, 0.0)
            else None
        )
        all_z = ((all_values - all_mean) / all_std).abs() if all_mean is not None and all_std not in (None, 0.0) else pd.Series(np.nan, index=merged.index)
        traded_z = ((traded_values - all_mean) / all_std).abs() if all_mean is not None and all_std not in (None, 0.0) else pd.Series(np.nan, index=traded.index)
        records.append(
            {
                "feature": column,
                "all_row_count": int(len(merged)),
                "traded_row_count": int(len(traded)),
                "all_missing_rate": float(all_values.isna().mean()) if len(all_values) else None,
                "traded_missing_rate": float(traded_values.isna().mean()) if len(traded_values) else None,
                "missing_rate_delta_traded_minus_all": (
                    float(traded_values.isna().mean() - all_values.isna().mean()) if len(traded_values) else None
                ),
                "all_mean": all_mean,
                "traded_mean": traded_mean,
                "all_std": all_std,
                "traded_std": _std_or_none(traded_values),
                "standardized_mean_shift_abs": standardized_shift,
                "all_extreme_abs_z_gt_5_rate": float(all_z.gt(5.0).mean()) if all_z.notna().any() else None,
                "traded_extreme_abs_z_gt_5_rate": float(traded_z.gt(5.0).mean()) if traded_z.notna().any() else None,
            }
        )
    if not records:
        return _empty_unavailable("no feature columns available for shift diagnostics")
    return pd.DataFrame(records).sort_values(
        ["standardized_mean_shift_abs", "missing_rate_delta_traded_minus_all"],
        ascending=[False, False],
        na_position="last",
    )


def _feature_quality_by_scope(feature_stats: pd.DataFrame) -> pd.DataFrame:
    if "unavailable_reason" in feature_stats.columns:
        return feature_stats
    records = [
        {
            "scope": "features_overall",
            "feature_count": int(len(feature_stats)),
            "mean_all_missing_rate": _mean_or_none(feature_stats["all_missing_rate"]),
            "mean_traded_missing_rate": _mean_or_none(feature_stats["traded_missing_rate"]),
            "max_standardized_mean_shift_abs": _quantile_or_none(
                feature_stats["standardized_mean_shift_abs"],
                1.0,
            ),
            "features_with_traded_missing_rate_gt_0": int(
                pd.to_numeric(feature_stats["traded_missing_rate"], errors="coerce").gt(0).sum()
            ),
            "features_with_traded_extreme_rate_gt_0": int(
                pd.to_numeric(feature_stats["traded_extreme_abs_z_gt_5_rate"], errors="coerce").gt(0).sum()
            ),
        }
    ]
    return pd.DataFrame(records)


def _return_training_scale(predictions: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    rows = predictions[
        predictions["model_id"].eq(RETURN_MODEL) & predictions["target_name"].eq(RETURN_TARGET)
    ].copy()
    if rows.empty:
        return _empty_unavailable(f"missing {RETURN_MODEL} / {RETURN_TARGET} prediction rows")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    feature_target = merged[["market", "year", "timestamp", "fold_id", "target_ret_15m"]].copy()
    joined = rows.merge(
        feature_target,
        on=["market", "year", "timestamp", "fold_id"],
        how="left",
        suffixes=("", "_feature"),
    )
    records: list[dict[str, Any]] = []
    for scope, cols in (("overall", []), ("market", ["market"]), ("fold", ["fold_id"])):
        groups = [((), joined)] if not cols else joined.groupby(cols, dropna=False)
        for keys, group in groups:
            if cols and not isinstance(keys, tuple):
                keys = (keys,)
            key_values = dict(zip(cols, keys)) if cols else {}
            y_true = _numeric(group, "y_true")
            y_pred = _numeric(group, "y_pred_calibrated")
            feature_y = _numeric(group, "target_ret_15m")
            y_true_std = _std_or_none(y_true)
            y_pred_std = _std_or_none(y_pred)
            feature_std = _std_or_none(feature_y)
            diff = (y_true - feature_y).abs()
            records.append(
                {
                    "scope": scope,
                    **key_values,
                    "row_count": int(len(group)),
                    "phase7_prediction_target_name": RETURN_TARGET,
                    "phase8_y_true_mean": _mean_or_none(y_true),
                    "feature_target_ret_mean": _mean_or_none(feature_y),
                    "phase8_y_true_std": y_true_std,
                    "feature_target_ret_std": feature_std,
                    "prediction_std": y_pred_std,
                    "prediction_to_y_true_std_ratio": (
                        abs(float(y_pred_std)) / abs(float(y_true_std))
                        if y_pred_std is not None and y_true_std not in (None, 0.0)
                        else None
                    ),
                    "phase8_y_true_feature_match_rate": _match_rate(
                        y_true,
                        feature_y,
                        tolerance=FEATURE_MATCH_TOLERANCE,
                    ),
                    "mean_abs_y_true_feature_diff": _mean_or_none(diff),
                    "max_abs_y_true_feature_diff": _quantile_or_none(diff, 1.0),
                    "same_target_units_reported": _match_rate(
                        y_true,
                        feature_y,
                        tolerance=FEATURE_MATCH_TOLERANCE,
                    )
                    == 1.0,
                }
            )
    return pd.DataFrame(records)


def _top_findings(
    *,
    alignment: pd.DataFrame,
    label_balance: pd.DataFrame,
    feature_shift: pd.DataFrame,
    return_scale: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    overall = alignment[alignment["scope"].eq("overall")].iloc[0]
    findings.append(
        "Feature target_ret_15m and Phase 8 y_true match rate is "
        f"{float(overall['observed_feature_return_match_rate'] or 0.0):.6f}."
    )
    findings.append(
        "Feature target_ret_15m vs execution-price return max abs diff is "
        f"{float(overall['max_abs_feature_vs_execution_return_diff'] or 0.0):.6g}."
    )
    worst = label_balance.iloc[0]
    findings.append(
        f"Worst market/fold by net is {worst['market']} {worst['fold_id']} at "
        f"{float(worst['net_return_dollars']):.2f}."
    )
    if "unavailable_reason" not in feature_shift.columns and not feature_shift.empty:
        top_feature = feature_shift.iloc[0]
        findings.append(
            f"Largest traded-vs-all feature shift is {top_feature['feature']} "
            f"with z-shift {float(top_feature['standardized_mean_shift_abs'] or 0.0):.2f}."
        )
    if "unavailable_reason" not in return_scale.columns:
        global_scale = return_scale[return_scale["scope"].eq("overall")].iloc[0]
        ratio = float(global_scale["prediction_to_y_true_std_ratio"] or 0.0)
        scale_phrase = "but" if ratio >= 100.0 else "and"
        findings.append(
            f"Phase 7 regression target units match Phase 8 y_true, {scale_phrase} "
            f"prediction/std ratio is {ratio:.2f}."
        )
    return findings[:5]


def _label_feature_decision(
    *,
    matched_feature_rows: int,
    policy_row_count: int,
    return_scale: pd.DataFrame,
) -> str:
    if matched_feature_rows != policy_row_count:
        return "feature_join_incomplete_audit_feature_availability_first"
    if "unavailable_reason" in return_scale.columns:
        return "targets_align_but_return_scale_unavailable"
    if "scope" not in return_scale.columns or not return_scale["scope"].eq("overall").any():
        return "targets_align_but_return_scale_unavailable"
    overall = return_scale[return_scale["scope"].eq("overall")].iloc[0]
    ratio = overall.get("prediction_to_y_true_std_ratio")
    if ratio is not None and float(ratio) >= 100.0:
        return "targets_align_but_return_prediction_scale_needs_audit"
    return "targets_align_return_scale_not_flagged_review_policy_signal_quality"


def _write_readme(path: Path, summary: Mapping[str, Any], outputs: Mapping[str, str]) -> None:
    findings = "\n".join(f"- {item}" for item in summary["top_findings"])
    files = "\n".join(f"- `{value}`" for key, value in sorted(outputs.items()) if key != "readme")
    text = f"""# Phase 8 Label/Feature Sanity

Run: `{summary['run']}`

This diagnostic joins saved Phase 8 policy rows back to the baseline feature
matrices by market/year/timestamp. It is read-only and does not change labels,
features, WFA splits, predictions, models, or policy behavior.

## Top Findings

{findings}

## Files

{files}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_label_feature_sanity(
    *,
    predictions_path: Path,
    costs_config: Path,
    feature_root: Path,
    output_root: Path,
    run: str,
    max_shift_features: int,
) -> dict[str, Any]:
    if not predictions_path.exists():
        raise SystemExit(f"prediction parquet missing: {_relative_path(predictions_path)}")
    predictions = pd.read_parquet(predictions_path)
    policy_frame, _, warnings = _policy_base(predictions, costs_config)
    feature_cols = _read_json_list(feature_root / "feature_cols.json")
    unavailable: list[str] = []
    if not feature_cols:
        unavailable.append(f"feature registry missing or empty: {_relative_path(feature_root / 'feature_cols.json')}")
    merged = _load_matched_feature_rows(
        policy_frame,
        feature_root=feature_root,
        feature_cols=feature_cols,
        unavailable=unavailable,
    )
    alignment = _target_alignment(merged)
    label_balance = _label_balance(merged)
    feature_stats = _feature_stats(merged, feature_cols)
    feature_quality = _feature_quality_by_scope(feature_stats)
    feature_shift = (
        feature_stats.head(max_shift_features).reset_index(drop=True)
        if "unavailable_reason" not in feature_stats.columns
        else feature_stats
    )
    return_scale = _return_training_scale(predictions, merged)
    outputs = _output_paths(output_root, run)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(outputs["target_alignment"], alignment)
    _write_csv(outputs["label_balance"], label_balance)
    _write_csv(outputs["feature_quality"], feature_quality)
    _write_csv(outputs["feature_shift"], feature_shift)
    _write_csv(outputs["return_scale"], return_scale)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "feature_root": _relative_path(feature_root),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(policy_frame)),
        "matched_feature_row_count": int(merged["feature_matrix_matched"].sum())
        if "feature_matrix_matched" in merged
        else 0,
        "target_alignment_overall": alignment[alignment["scope"].eq("overall")].iloc[0].to_dict(),
        "worst_label_balance_rows": label_balance.head(10).to_dict(orient="records"),
        "feature_quality": feature_quality.to_dict(orient="records"),
        "top_feature_shifts": feature_shift.head(10).to_dict(orient="records"),
        "return_training_scale_overall": return_scale[return_scale["scope"].eq("overall")].iloc[0].to_dict()
        if "scope" in return_scale.columns and return_scale["scope"].eq("overall").any()
        else return_scale.iloc[0].to_dict(),
        "top_findings": _top_findings(
            alignment=alignment,
            label_balance=label_balance,
            feature_shift=feature_shift,
            return_scale=return_scale,
        ),
        "decision": _label_feature_decision(
            matched_feature_rows=int(merged["feature_matrix_matched"].sum())
            if "feature_matrix_matched" in merged
            else 0,
            policy_row_count=len(policy_frame),
            return_scale=return_scale,
        ),
        "unavailable_diagnostics": unavailable,
        "warnings": warnings,
        "outputs": {key: _relative_path(path) for key, path in outputs.items()},
    }
    _write_json(outputs["summary"], summary)
    _write_readme(outputs["readme"], summary, summary["outputs"])
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS.as_posix())
    parser.add_argument("--costs-config", default=DEFAULT_COSTS_CONFIG.as_posix())
    parser.add_argument("--feature-root", default=DEFAULT_FEATURE_ROOT.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--max-shift-features", type=int, default=40)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_label_feature_sanity(
        predictions_path=Path(args.predictions),
        costs_config=Path(args.costs_config),
        feature_root=Path(args.feature_root),
        output_root=Path(args.output_root),
        run=args.run,
        max_shift_features=args.max_shift_features,
    )
    align = summary["target_alignment_overall"]
    print(
        "PASS label-feature sanity: "
        f"policy_rows={summary['policy_row_count']} "
        f"matched_features={summary['matched_feature_row_count']} "
        f"target_match={align['observed_feature_return_match_rate']} "
        f"decision={summary['decision']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
