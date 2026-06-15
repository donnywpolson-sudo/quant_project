#!/usr/bin/env python3
"""Audit Phase 7 return-model prediction scale and outlier provenance."""

from __future__ import annotations

import argparse
import inspect
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


DEFAULT_REPORTS_ROOT = Path("reports/wfa")
DEFAULT_FEATURE_ROOT = Path("data/feature_matrices/baseline")
DEFAULT_SPLIT_PLAN = Path("reports/wfa/split_plan.json")
DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")
RETURN_MODEL = "ridge_return_v1"
RETURN_TARGET = "target_ret_15m"
DEFAULT_ABS_OUTLIER_THRESHOLD = 0.01
DEFAULT_RATIO_WARN_THRESHOLD = 100.0


def _output_paths(output_root: Path, run: str) -> dict[str, Path]:
    return {
        "summary": output_root / f"{run}_return_model_scale_summary.json",
        "scale_by_scope": output_root / f"{run}_return_model_scale_by_scope.csv",
        "raw_calibrated": output_root / f"{run}_return_model_raw_calibrated_check.csv",
        "outliers": output_root / f"{run}_return_model_outliers.csv",
        "wfa_reconciliation": output_root / f"{run}_return_model_wfa_reconciliation.csv",
        "feature_contributions": output_root / f"{run}_return_model_feature_contributions.csv",
        "readme": output_root / f"{run}_return_model_scale_readme.md",
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _std(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=0)) if not values.empty else None


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _quantile(series: pd.Series, q: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else None


def _max_abs(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").abs().dropna()
    return float(values.max()) if not values.empty else None


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(aligned) < 2:
        return None
    return float(aligned["left"].corr(aligned["right"]))


def _match_rate(left: pd.Series, right: pd.Series, *, tolerance: float = 1e-12) -> float | None:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    if aligned.empty:
        return None
    return float((aligned["left"] - aligned["right"]).abs().le(tolerance).mean())


def _return_rows(predictions: pd.DataFrame, model_id: str, target_name: str) -> pd.DataFrame:
    rows = predictions[
        predictions["model_id"].eq(model_id) & predictions["target_name"].eq(target_name)
    ].copy()
    if rows.empty:
        raise SystemExit(f"missing prediction rows for {model_id} / {target_name}")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    for column in ("y_true", "y_pred_raw", "y_pred_calibrated"):
        rows[column] = _numeric(rows, column)
    return rows


def _scale_record(
    group: pd.DataFrame,
    *,
    scope: str,
    keys: Mapping[str, Any],
    abs_outlier_threshold: float,
) -> dict[str, Any]:
    y_true = _numeric(group, "y_true")
    raw = _numeric(group, "y_pred_raw")
    calibrated = _numeric(group, "y_pred_calibrated")
    target_std = _std(y_true)
    calibrated_std = _std(calibrated)
    abs_pred = calibrated.abs()
    return {
        "scope": scope,
        **dict(keys),
        "row_count": int(len(group)),
        "target_mean": _mean(y_true),
        "target_std": target_std,
        "target_abs_p95": _quantile(y_true.abs(), 0.95),
        "raw_prediction_mean": _mean(raw),
        "raw_prediction_std": _std(raw),
        "raw_prediction_abs_p95": _quantile(raw.abs(), 0.95),
        "raw_prediction_abs_p99": _quantile(raw.abs(), 0.99),
        "raw_prediction_abs_max": _max_abs(raw),
        "calibrated_prediction_mean": _mean(calibrated),
        "calibrated_prediction_std": calibrated_std,
        "calibrated_prediction_abs_p95": _quantile(abs_pred, 0.95),
        "calibrated_prediction_abs_p99": _quantile(abs_pred, 0.99),
        "calibrated_prediction_abs_p999": _quantile(abs_pred, 0.999),
        "calibrated_prediction_abs_max": _max_abs(calibrated),
        "prediction_to_target_std_ratio": (
            abs(float(calibrated_std)) / abs(float(target_std))
            if calibrated_std is not None and target_std not in (None, 0.0)
            else None
        ),
        "prediction_target_correlation": _corr(calibrated, y_true),
        "raw_calibrated_match_rate": _match_rate(raw, calibrated),
        "abs_prediction_outlier_count": int(abs_pred.gt(abs_outlier_threshold).sum()),
        "abs_prediction_outlier_rate": (
            float(abs_pred.gt(abs_outlier_threshold).mean()) if len(abs_pred) else None
        ),
    }


def _scale_by_scope(rows: pd.DataFrame, *, abs_outlier_threshold: float) -> pd.DataFrame:
    specs = [
        ("overall", []),
        ("market", ["market"]),
        ("fold", ["fold_id"]),
        ("market_fold", ["market", "fold_id"]),
    ]
    records: list[dict[str, Any]] = []
    for scope, group_cols in specs:
        if not group_cols:
            records.append(
                _scale_record(
                    rows,
                    scope=scope,
                    keys={},
                    abs_outlier_threshold=abs_outlier_threshold,
                )
            )
            continue
        for keys, group in rows.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            records.append(
                _scale_record(
                    group,
                    scope=scope,
                    keys=dict(zip(group_cols, keys)),
                    abs_outlier_threshold=abs_outlier_threshold,
                )
            )
    return pd.DataFrame(records).sort_values(
        ["scope", "prediction_to_target_std_ratio", "calibrated_prediction_abs_max"],
        ascending=[True, False, False],
        na_position="last",
    )


def _raw_calibrated_check(rows: pd.DataFrame) -> pd.DataFrame:
    raw = _numeric(rows, "y_pred_raw")
    calibrated = _numeric(rows, "y_pred_calibrated")
    diff = (raw - calibrated).abs()
    return pd.DataFrame(
        [
            {
                "row_count": int(len(rows)),
                "raw_calibrated_match_rate": _match_rate(raw, calibrated),
                "raw_calibrated_max_abs_diff": _max_abs(diff),
                "raw_calibrated_mean_abs_diff": _mean(diff),
                "calibration_ids": ",".join(sorted(rows["calibration_id"].dropna().astype(str).unique()))
                if "calibration_id" in rows
                else "",
                "raw_calibrated_identical": bool(diff.fillna(0.0).le(1e-12).all()),
            }
        ]
    )


def _outlier_rows(rows: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    columns = [
        "market",
        "year",
        "fold_id",
        "timestamp",
        "model_id",
        "target_name",
        "y_true",
        "y_pred_raw",
        "y_pred_calibrated",
        "calibration_id",
        "execution_open",
        "execution_close",
    ]
    available = [column for column in columns if column in rows.columns]
    out = rows.assign(abs_prediction=rows["y_pred_calibrated"].abs())
    out = out.sort_values("abs_prediction", ascending=False).head(max_rows)
    out["prediction_minus_target"] = out["y_pred_calibrated"] - out["y_true"]
    return out[[*available, "abs_prediction", "prediction_minus_target"]].reset_index(drop=True)


def _report_paths(reports_root: Path, run: str) -> list[Path]:
    candidates = sorted(reports_root.glob(f"{run}*_wfa_report.json"))
    exact = reports_root / f"{run}_wfa_report.json"
    if exact.exists() and exact not in candidates:
        candidates.append(exact)
    return sorted(candidates)


def _load_wfa_diagnostics(
    *,
    reports_root: Path,
    run: str,
    model_id: str,
    target_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in _report_paths(reports_root, run):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        diagnostics = report.get("diagnostics", [])
        if not isinstance(diagnostics, list):
            continue
        for item in diagnostics:
            if not isinstance(item, Mapping):
                continue
            if item.get("model_id") != model_id or item.get("target_name") != target_name:
                continue
            records.append(
                {
                    "report_path": _relative_path(path),
                    "market": item.get("market"),
                    "fold_id": item.get("fold_id"),
                    "status": item.get("status"),
                    "fit_estimator": item.get("fit_estimator"),
                    "train_rows": item.get("train_rows"),
                    "test_rows": item.get("test_rows"),
                    "fit_ts_min": item.get("fit_ts_min"),
                    "fit_ts_max": item.get("fit_ts_max"),
                    "score_ts_min": item.get("score_ts_min"),
                    "score_ts_max": item.get("score_ts_max"),
                    "phase7_report_prediction_std": item.get("prediction_std"),
                    "warning_text": "; ".join(str(x) for x in item.get("warnings", []))
                    if isinstance(item.get("warnings"), list)
                    else item.get("warnings"),
                }
            )
    if not records:
        return pd.DataFrame(
            [{"unavailable_reason": f"no matching WFA diagnostics under {_relative_path(reports_root)}"}]
        )
    return pd.DataFrame(records)


def _wfa_reconciliation(scale: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if "unavailable_reason" in diagnostics.columns:
        return diagnostics
    fold_scale = scale[scale["scope"].eq("market_fold")].copy()
    keep_cols = [
        "market",
        "fold_id",
        "row_count",
        "target_std",
        "calibrated_prediction_std",
        "calibrated_prediction_abs_max",
        "prediction_to_target_std_ratio",
        "abs_prediction_outlier_count",
    ]
    merged = diagnostics.merge(fold_scale[keep_cols], on=["market", "fold_id"], how="left")
    reported = pd.to_numeric(merged["phase7_report_prediction_std"], errors="coerce")
    actual = pd.to_numeric(merged["calibrated_prediction_std"], errors="coerce")
    merged["phase7_report_vs_saved_prediction_std_abs_diff"] = (reported - actual).abs()
    merged["phase7_report_matches_saved_predictions"] = merged[
        "phase7_report_vs_saved_prediction_std_abs_diff"
    ].le(1e-10)
    return merged.sort_values(
        ["calibrated_prediction_abs_max", "prediction_to_target_std_ratio"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _source_location(func: Any) -> dict[str, Any]:
    path = Path(inspect.getsourcefile(func) or "")
    try:
        _, line = inspect.getsourcelines(func)
    except OSError:
        line = None
    return {
        "path": _relative_path(path),
        "function": getattr(func, "__name__", "<unknown>"),
        "start_line": line,
    }


def _empty_unavailable(reason: str) -> pd.DataFrame:
    return pd.DataFrame([{"unavailable_reason": reason}])


def _load_split_fold(split_plan: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(split_plan.read_text(encoding="utf-8"))
    folds = payload.get("folds", [])
    if not isinstance(folds, list):
        raise ValueError("split plan folds are missing")
    matches = [fold for fold in folds if isinstance(fold, Mapping) and fold.get("fold_id") == fold_id]
    if not matches:
        raise ValueError(f"fold not found in split plan: {fold_id}")
    return dict(payload), dict(matches[0])


def _replay_top_outlier(
    *,
    outliers: pd.DataFrame,
    feature_root: Path,
    split_plan: Path,
    models_config: Path,
    model_id: str,
    top_contributions: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if outliers.empty:
        return _empty_unavailable("no outlier rows"), {}
    if not split_plan.exists():
        return _empty_unavailable(f"split plan missing: {_relative_path(split_plan)}"), {}
    if not models_config.exists():
        return _empty_unavailable(f"models config missing: {_relative_path(models_config)}"), {}
    try:
        import scripts.phase4_features.build_baseline_features as phase4
        import scripts.phase7_wfa.run_wfa as wfa

        row = outliers.iloc[0]
        split_payload, fold = _load_split_fold(split_plan, str(row["fold_id"]))
        feature_cols, _ = wfa.load_feature_cols(feature_root)
        specs, _ = wfa.load_model_specs(models_config)
        spec = next(spec for spec in specs if spec.model_id == model_id)
        years = [int(year) for year in split_payload.get("years", [])]
        frame, failures, _ = wfa._load_market_frame(
            str(row["market"]),
            years,
            feature_root,
            wfa._required_source_columns(feature_cols, specs),
        )
        if failures or frame is None:
            return _empty_unavailable("; ".join(failures) or "feature frame unavailable"), {}
        target = wfa._target_series(frame, spec)
        train_mask, test_mask = wfa._fold_masks(frame, fold, target)
        train = frame.loc[train_mask].copy()
        test = frame.loc[test_mask].copy()
        if train.empty or test.empty:
            return _empty_unavailable("empty train or test rows during replay"), {}
        estimator, actual_estimator = wfa._build_estimator(spec, target.loc[train_mask])
        estimator.fit(train[feature_cols], target.loc[train_mask])
        timestamp = pd.Timestamp(row["timestamp"])
        candidate = test[pd.to_datetime(test["ts"], utc=True, errors="coerce").eq(timestamp)]
        if candidate.empty:
            return _empty_unavailable("top outlier timestamp not found in feature matrix"), {}
        selected = candidate.iloc[[0]]
        replayed_prediction = float(estimator.predict(selected[feature_cols])[0])
        if not all(name in estimator.named_steps for name in ("imputer", "scaler", "model")):
            return _empty_unavailable("replayed estimator is not an imputer/scaler/model pipeline"), {
                "replayed_prediction": replayed_prediction,
                "fit_estimator": actual_estimator,
            }
        imputer = estimator.named_steps["imputer"]
        scaler = estimator.named_steps["scaler"]
        model = estimator.named_steps["model"]
        imputed = imputer.transform(selected[feature_cols])
        scaled = scaler.transform(imputed)
        contributions = scaled[0] * model.coef_
        train_numeric = train[feature_cols].apply(
            lambda column: pd.to_numeric(column, errors="coerce").astype(float)
        )
        selected_values = selected[feature_cols].iloc[0]
        records: list[dict[str, Any]] = []
        for idx, feature in enumerate(feature_cols):
            train_col = train_numeric[feature]
            value = selected_values[feature]
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").astype(float).iloc[0]
            train_min = _quantile(train_col, 0.0)
            train_max = _quantile(train_col, 1.0)
            records.append(
                {
                    "market": row["market"],
                    "fold_id": row["fold_id"],
                    "timestamp": row["timestamp"],
                    "feature": feature,
                    "feature_value": numeric_value,
                    "imputed_value": float(imputed[0][idx]),
                    "train_mean": float(scaler.mean_[idx]),
                    "train_scale": float(scaler.scale_[idx]),
                    "train_min": train_min,
                    "train_max": train_max,
                    "standardized_value": float(scaled[0][idx]),
                    "ridge_coefficient": float(model.coef_[idx]),
                    "prediction_contribution": float(contributions[idx]),
                    "abs_prediction_contribution": float(abs(contributions[idx])),
                    "outside_train_min_max": (
                        bool(numeric_value < train_min or numeric_value > train_max)
                        if pd.notna(numeric_value) and train_min is not None and train_max is not None
                        else None
                    ),
                }
            )
        frame_out = pd.DataFrame(records).sort_values(
            "abs_prediction_contribution",
            ascending=False,
        ).head(top_contributions)
        metadata = {
            "fit_estimator": actual_estimator,
            "replayed_prediction": replayed_prediction,
            "saved_prediction": float(row["y_pred_calibrated"]),
            "replay_abs_diff": abs(replayed_prediction - float(row["y_pred_calibrated"])),
            "intercept": float(model.intercept_),
            "top_outlier_market": row["market"],
            "top_outlier_fold_id": row["fold_id"],
            "top_outlier_timestamp": str(row["timestamp"]),
            "phase7_locations": [
                _source_location(wfa.run_wfa),
                _source_location(wfa._build_estimator),
                _source_location(wfa._prediction_frame),
            ],
            "phase4_candidate_feature_location": _source_location(phase4.shock_decay_features),
        }
        return frame_out.reset_index(drop=True), metadata
    except Exception as exc:
        return _empty_unavailable(f"replay failed: {exc}"), {}


def _decision(
    *,
    scale: pd.DataFrame,
    raw_calibrated: pd.DataFrame,
    contributions: pd.DataFrame,
    replay_metadata: Mapping[str, Any],
    ratio_warn_threshold: float,
) -> str:
    overall = scale[scale["scope"].eq("overall")].iloc[0]
    ratio = overall.get("prediction_to_target_std_ratio")
    raw_same = bool(raw_calibrated.iloc[0].get("raw_calibrated_identical"))
    if ratio is not None and float(ratio) >= ratio_warn_threshold:
        replay_diff = replay_metadata.get("replay_abs_diff")
        saved_prediction = replay_metadata.get("saved_prediction")
        if replay_diff is not None and saved_prediction is not None:
            tolerance = max(1e-8, abs(float(saved_prediction)) * 1e-6)
            if float(replay_diff) > tolerance:
                return "saved_predictions_stale_after_feature_rebuild_regenerate_phase7_for_affected_scope"
        if "unavailable_reason" not in contributions.columns and not contributions.empty:
            return "extreme_feature_value_drives_unbounded_phase7_ridge_prediction"
        if raw_same:
            return "uncalibrated_phase7_return_predictions_have_extreme_scale"
        return "calibrated_phase7_return_predictions_have_extreme_scale"
    return "return_prediction_scale_not_flagged"


def _top_findings(
    *,
    scale: pd.DataFrame,
    raw_calibrated: pd.DataFrame,
    outliers: pd.DataFrame,
    reconciliation: pd.DataFrame,
    contributions: pd.DataFrame,
    replay_metadata: Mapping[str, Any],
) -> list[str]:
    findings: list[str] = []
    overall = scale[scale["scope"].eq("overall")].iloc[0]
    findings.append(
        "Return prediction/std ratio is "
        f"{float(overall['prediction_to_target_std_ratio'] or 0.0):.2f}."
    )
    findings.append(
        "Raw and calibrated return predictions match rate is "
        f"{float(raw_calibrated.iloc[0]['raw_calibrated_match_rate'] or 0.0):.6f}."
    )
    if not outliers.empty:
        top = outliers.iloc[0]
        findings.append(
            f"Largest return prediction is {top['market']} {top['fold_id']} "
            f"{top['timestamp']} at {float(top['y_pred_calibrated']):.6g}."
        )
    if "unavailable_reason" not in reconciliation.columns and not reconciliation.empty:
        first = reconciliation.iloc[0]
        findings.append(
            "Phase 7 WFA report prediction_std matches saved predictions for worst fold: "
            f"{bool(first.get('phase7_report_matches_saved_predictions'))}."
        )
    if replay_metadata.get("replay_abs_diff") is not None:
        findings.append(
            "Current feature replay prediction is "
            f"{float(replay_metadata.get('replayed_prediction') or 0.0):.6g}; "
            f"saved-vs-replay abs diff is {float(replay_metadata['replay_abs_diff']):.6g}."
        )
    if "unavailable_reason" not in contributions.columns and not contributions.empty:
        top_feature = contributions.iloc[0]
        findings.append(
            f"Top replay contribution is {top_feature['feature']} at "
            f"{float(top_feature['prediction_contribution']):.6g}."
        )
    return findings[:5]


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    findings = "\n".join(f"- {item}" for item in summary["top_findings"])
    files = "\n".join(
        f"- `{value}`" for key, value in sorted(summary["outputs"].items()) if key != "readme"
    )
    text = f"""# Phase 7 Return Model Scale Audit

Run: `{summary['run']}`

This diagnostic is read-only. It audits saved Phase 7 return predictions,
raw-vs-calibrated score handling, WFA diagnostic consistency, and the largest
outlier's replayed feature contribution.

## Top Findings

{findings}

## Files

{files}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_return_model_scale_audit(
    *,
    predictions_path: Path,
    reports_root: Path,
    feature_root: Path,
    split_plan: Path,
    models_config: Path,
    output_root: Path,
    run: str,
    model_id: str,
    target_name: str,
    abs_outlier_threshold: float,
    ratio_warn_threshold: float,
    max_outliers: int,
    top_contributions: int,
) -> dict[str, Any]:
    if not predictions_path.exists():
        raise SystemExit(f"prediction parquet missing: {_relative_path(predictions_path)}")
    predictions = pd.read_parquet(predictions_path)
    rows = _return_rows(predictions, model_id, target_name)
    scale = _scale_by_scope(rows, abs_outlier_threshold=abs_outlier_threshold)
    raw_calibrated = _raw_calibrated_check(rows)
    outliers = _outlier_rows(rows, max_rows=max_outliers)
    diagnostics = _load_wfa_diagnostics(
        reports_root=reports_root,
        run=run,
        model_id=model_id,
        target_name=target_name,
    )
    reconciliation = _wfa_reconciliation(scale, diagnostics)
    contributions, replay_metadata = _replay_top_outlier(
        outliers=outliers,
        feature_root=feature_root,
        split_plan=split_plan,
        models_config=models_config,
        model_id=model_id,
        top_contributions=top_contributions,
    )

    outputs = _output_paths(output_root, run)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(outputs["scale_by_scope"], scale)
    _write_csv(outputs["raw_calibrated"], raw_calibrated)
    _write_csv(outputs["outliers"], outliers)
    _write_csv(outputs["wfa_reconciliation"], reconciliation)
    _write_csv(outputs["feature_contributions"], contributions)

    decision = _decision(
        scale=scale,
        raw_calibrated=raw_calibrated,
        contributions=contributions,
        replay_metadata=replay_metadata,
        ratio_warn_threshold=ratio_warn_threshold,
    )
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "reports_root": _relative_path(reports_root),
        "feature_root": _relative_path(feature_root),
        "model_id": model_id,
        "target_name": target_name,
        "return_row_count": int(len(rows)),
        "overall_scale": scale[scale["scope"].eq("overall")].iloc[0].to_dict(),
        "raw_calibrated_check": raw_calibrated.iloc[0].to_dict(),
        "largest_outliers": outliers.head(10).to_dict(orient="records"),
        "worst_wfa_reconciliation_rows": reconciliation.head(10).to_dict(orient="records"),
        "top_feature_contributions": contributions.head(10).to_dict(orient="records"),
        "replay_metadata": replay_metadata,
        "decision": decision,
        "top_findings": _top_findings(
            scale=scale,
            raw_calibrated=raw_calibrated,
            outliers=outliers,
            reconciliation=reconciliation,
            contributions=contributions,
            replay_metadata=replay_metadata,
        ),
        "outputs": {key: _relative_path(path) for key, path in outputs.items()},
    }
    _write_json(outputs["summary"], summary)
    _write_readme(outputs["readme"], summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS.as_posix())
    parser.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT.as_posix())
    parser.add_argument("--feature-root", default=DEFAULT_FEATURE_ROOT.as_posix())
    parser.add_argument("--split-plan", default=DEFAULT_SPLIT_PLAN.as_posix())
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--model-id", default=RETURN_MODEL)
    parser.add_argument("--target-name", default=RETURN_TARGET)
    parser.add_argument("--abs-outlier-threshold", type=float, default=DEFAULT_ABS_OUTLIER_THRESHOLD)
    parser.add_argument("--ratio-warn-threshold", type=float, default=DEFAULT_RATIO_WARN_THRESHOLD)
    parser.add_argument("--max-outliers", type=int, default=50)
    parser.add_argument("--top-contributions", type=int, default=40)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_return_model_scale_audit(
        predictions_path=Path(args.predictions),
        reports_root=Path(args.reports_root),
        feature_root=Path(args.feature_root),
        split_plan=Path(args.split_plan),
        models_config=Path(args.models_config),
        output_root=Path(args.output_root),
        run=args.run,
        model_id=args.model_id,
        target_name=args.target_name,
        abs_outlier_threshold=args.abs_outlier_threshold,
        ratio_warn_threshold=args.ratio_warn_threshold,
        max_outliers=args.max_outliers,
        top_contributions=args.top_contributions,
    )
    scale = summary["overall_scale"]
    print(
        "PASS return model scale audit: "
        f"rows={summary['return_row_count']} "
        f"ratio={scale['prediction_to_target_std_ratio']} "
        f"decision={summary['decision']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
