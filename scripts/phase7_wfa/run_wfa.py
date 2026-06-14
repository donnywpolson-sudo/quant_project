#!/usr/bin/env python3
"""Run simple train-only baseline models on existing WFA split plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PROFILE = "tier_1"
DEFAULT_MATRIX = "baseline"
DEFAULT_RUN = "baseline"
DEFAULT_INPUT_ROOT = Path("data/feature_matrices/baseline")
DEFAULT_SPLIT_PLAN = Path("reports/wfa/split_plan.json")
DEFAULT_PREDICTIONS_ROOT = Path("data/predictions")
DEFAULT_REPORTS_ROOT = Path("reports/wfa")
DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")
NO_CALIBRATION_ID = "no_calibration"
PHASE_7A_STAGE = "phase_7a_linear_controls"
CLASSIFIER_COLLAPSE_STD_EPS = 1e-9
PREDICTION_COLUMNS = [
    "market",
    "year",
    "fold_id",
    "timestamp",
    "session_id",
    "session_segment_id",
    "split_group",
    "model_id",
    "model_family",
    "target_name",
    "prediction_type",
    "y_true",
    "y_pred_raw",
    "y_pred_calibrated",
    "p_long",
    "p_short",
    "p_flat",
    "p_fade_success",
    "p_trend_danger",
    "calibration_id",
    "model_config_hash",
    "feature_config_hash",
    "execution_open",
    "execution_close",
    "target_valid",
    "causal_valid",
    "close",
    "target_entry_ts",
    "target_exit_ts",
    "minutes_until_session_close",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    stage: str
    family: str
    task: str
    target: str
    config_hash: str


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hash_or_missing(path: Path) -> str:
    return _file_sha256(path) if path.exists() else "MISSING"


def _file_hash_map(paths: Iterable[Path]) -> dict[str, str]:
    return {_relative_path(path): _file_hash_or_missing(path) for path in paths}


def _stale_prediction_output_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "stale_output_path_exists": False,
            "stale_output_path": None,
            "stale_output_file_hash": None,
            "stale_output_mtime_utc": None,
            "stale_output_row_count": None,
            "stale_output_split_groups": [],
        }

    info: dict[str, Any] = {
        "stale_output_path_exists": True,
        "stale_output_path": _relative_path(path),
        "stale_output_file_hash": _file_sha256(path),
        "stale_output_mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "stale_output_row_count": None,
        "stale_output_split_groups": [],
    }
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        info["stale_output_row_count"] = int(parquet_file.metadata.num_rows)
        if "split_group" in parquet_file.schema.names:
            table = pq.read_table(path, columns=["split_group"])
            groups = table.column("split_group").to_pylist()
            info["stale_output_split_groups"] = sorted(
                {str(value) for value in groups if value is not None}
            )
    except Exception as exc:
        info["stale_output_read_error"] = str(exc)
    return info


def prediction_artifact_evidence_failures(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if int(manifest.get("failure_count") or 0) > 0:
        failures.append("manifest failure_count is nonzero")
    if int(manifest.get("prediction_count") or 0) <= 0:
        failures.append("manifest prediction_count is zero")
    output_hashes = manifest.get("output_file_hashes", {})
    if isinstance(output_hashes, Mapping) and any(value == "NOT_WRITTEN" for value in output_hashes.values()):
        failures.append("manifest output hash is NOT_WRITTEN")
    if manifest.get("stale_output_path_exists") is True:
        failures.append("stale prediction output exists from a previous run")
    return failures


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def load_feature_cols(input_root: Path, feature_cols_path: Path | None = None) -> tuple[list[str], Path]:
    path = feature_cols_path or (input_root / "feature_cols.json")
    if not path.exists():
        raise SystemExit(f"feature column registry missing: {_relative_path(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise SystemExit(f"invalid feature column registry: {_relative_path(path)}")
    if any(item.startswith("target_") for item in payload):
        raise SystemExit("feature column registry contains target columns")
    return list(payload), path


def load_model_specs(models_config: Path) -> tuple[list[ModelSpec], dict[str, Any]]:
    config = _read_yaml(models_config)
    policy = config.get("policy", {})
    if not isinstance(policy, Mapping):
        raise SystemExit("models policy mapping missing")
    if policy.get("random_splits_allowed") is not False:
        raise SystemExit("random train/test splits must be disabled")
    if policy.get("hyperparameter_tuning_allowed_initially") is not False:
        raise SystemExit("initial hyperparameter tuning must be disabled")
    if policy.get("final_holdout_tuning_allowed") is not False:
        raise SystemExit("final holdout tuning must be disabled")

    models = config.get("models", {})
    if not isinstance(models, Mapping):
        raise SystemExit("models mapping missing")
    specs: list[ModelSpec] = []
    for model_id, raw_model in models.items():
        if not isinstance(model_id, str) or not isinstance(raw_model, Mapping):
            continue
        if raw_model.get("enabled") is not True:
            continue
        if raw_model.get("requires_optional_dependency") is True:
            continue
        if raw_model.get("stage") != PHASE_7A_STAGE:
            continue
        family = str(raw_model.get("family", ""))
        task = str(raw_model.get("task", ""))
        if family not in {"ridge_regression", "logistic_regression"}:
            raise SystemExit(f"unsupported initial model family for {model_id}: {family}")
        if task not in {"regression", "classification"}:
            raise SystemExit(f"unsupported task for {model_id}: {task}")
        target = str(raw_model.get("target", ""))
        if not target:
            raise SystemExit(f"missing target for {model_id}")
        specs.append(
            ModelSpec(
                model_id=model_id,
                stage=str(raw_model["stage"]),
                family=family,
                task=task,
                target=target,
                config_hash=_stable_hash({"model_id": model_id, **dict(raw_model)}),
            )
        )
    if not specs:
        raise SystemExit("no enabled Phase 7A baseline models found")
    return specs, config


def _required_source_columns(feature_cols: list[str], model_specs: list[ModelSpec]) -> list[str]:
    columns = set(feature_cols)
    columns.update(
        {
            "ts",
            "market",
            "year",
            "session_id",
            "session_segment_id",
            "causal_valid",
            "target_valid",
            "feature_input_valid",
            "training_row_valid",
            "close",
            "target_entry_ts",
            "target_exit_ts",
            "target_entry_price",
            "target_exit_price",
            "minutes_until_session_close",
        }
    )
    for spec in model_specs:
        columns.add(spec.target)
    return sorted(columns)


def _read_schema(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


def _load_market_frame(
    market: str,
    years: Iterable[int],
    input_root: Path,
    columns: list[str],
) -> tuple[pd.DataFrame | None, list[str], list[Path]]:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    paths: list[Path] = []
    for year in sorted(set(int(item) for item in years)):
        path = input_root / market / f"{year}.parquet"
        paths.append(path)
        if not path.exists():
            failures.append(f"missing feature matrix: {_relative_path(path)}")
            continue
        available = _read_schema(path)
        missing = [column for column in columns if column not in available]
        required_missing = [
            column
            for column in missing
            if column in {"ts", "market", "year", "session_segment_id", "target_valid", "causal_valid"}
            or column.startswith("target_")
        ]
        if required_missing:
            failures.append(
                f"feature matrix missing required columns {required_missing}: {_relative_path(path)}"
            )
            continue
        read_columns = [column for column in columns if column in available]
        frame = pd.read_parquet(path, columns=read_columns)
        for column in missing:
            frame[column] = np.nan
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        frames.append(frame)
    if not frames:
        return None, failures, paths
    out = pd.concat(frames, ignore_index=True).sort_values("ts", kind="mergesort")
    return out.reset_index(drop=True), failures, paths


def _valid_bool(df: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[column].fillna(default).astype(bool)


def _target_series(df: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    if spec.target not in df.columns:
        return pd.Series(np.nan, index=df.index)
    series = df[spec.target]
    if spec.task == "classification":
        if series.dtype == bool:
            return series.astype(int)
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _build_estimator(spec: ModelSpec, y_train: pd.Series) -> tuple[Any, str]:
    if spec.task == "regression":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            "ridge_regression",
        )

    unique = pd.Series(y_train).dropna().unique()
    if len(unique) < 2:
        return DummyClassifier(strategy="prior"), "dummy_class_prior"
    return (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=200)),
            ]
        ),
        "logistic_regression",
    )


def _positive_probability(classes: np.ndarray, probabilities: np.ndarray, positive: object) -> np.ndarray:
    out = np.full(len(probabilities), np.nan)
    for idx, value in enumerate(classes):
        if value == positive:
            out = probabilities[:, idx]
            break
    return out


def _classification_predictions(
    estimator: Any,
    spec: ModelSpec,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probabilities = estimator.predict_proba(x_test)
    classes = np.asarray(estimator.classes_)
    p_long = _positive_probability(classes, probabilities, 1)
    p_short = _positive_probability(classes, probabilities, -1)
    p_flat = _positive_probability(classes, probabilities, 0)
    p_true = _positive_probability(classes, probabilities, 1)

    columns = {
        "p_long": np.full(len(x_test), np.nan),
        "p_short": np.full(len(x_test), np.nan),
        "p_flat": np.full(len(x_test), np.nan),
        "p_fade_success": np.full(len(x_test), np.nan),
        "p_trend_danger": np.full(len(x_test), np.nan),
    }
    if spec.target == "target_sign_with_deadzone":
        columns["p_long"] = p_long
        columns["p_short"] = p_short
        columns["p_flat"] = p_flat
        raw = np.nan_to_num(p_long, nan=0.0) - np.nan_to_num(p_short, nan=0.0)
    elif "fade_success" in spec.target:
        columns["p_fade_success"] = p_true
        raw = p_true
    elif "trend_danger" in spec.target:
        columns["p_trend_danger"] = p_true
        raw = p_true
    else:
        raw = p_true
    return raw, columns


def _finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    return float(np.std(finite))


def classifier_collapse_failure(
    *,
    spec: ModelSpec,
    actual_estimator: str,
    raw_pred: np.ndarray,
    probability_cols: dict[str, np.ndarray],
) -> str | None:
    if spec.task != "classification" or actual_estimator == "dummy_class_prior":
        return None
    relevant = raw_pred
    if spec.target == "target_sign_with_deadzone":
        relevant = np.nan_to_num(probability_cols["p_long"], nan=0.0) - np.nan_to_num(
            probability_cols["p_short"], nan=0.0
        )
    elif "fade_success" in spec.target:
        relevant = probability_cols["p_fade_success"]
    elif "trend_danger" in spec.target:
        relevant = probability_cols["p_trend_danger"]
    if _finite_std(np.asarray(relevant, dtype=float)) <= CLASSIFIER_COLLAPSE_STD_EPS:
        return f"{spec.model_id}: classifier probabilities collapsed to near-constant class-prior values"
    return None


def _prediction_frame(
    test: pd.DataFrame,
    spec: ModelSpec,
    fold: Mapping[str, Any],
    y_true: pd.Series,
    raw_pred: np.ndarray,
    probability_cols: dict[str, np.ndarray] | None,
    feature_config_hash: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "market": test["market"].astype(str).to_numpy(),
            "year": pd.to_numeric(test["year"], errors="coerce").astype("Int64").to_numpy(),
            "fold_id": str(fold["fold_id"]),
            "timestamp": test["ts"].to_numpy(),
            "session_id": test.get("session_id", pd.Series(pd.NA, index=test.index)).to_numpy(),
            "session_segment_id": test["session_segment_id"].astype(str).to_numpy(),
            "split_group": str(fold.get("split_group", "")),
            "model_id": spec.model_id,
            "model_family": spec.family,
            "target_name": spec.target,
            "prediction_type": "regression" if spec.task == "regression" else "classification_probability",
            "y_true": y_true.to_numpy(),
            "y_pred_raw": raw_pred,
            "y_pred_calibrated": raw_pred,
            "calibration_id": NO_CALIBRATION_ID,
            "model_config_hash": spec.config_hash,
            "feature_config_hash": feature_config_hash,
            "execution_open": pd.to_numeric(test["target_entry_price"], errors="coerce").to_numpy(),
            "execution_close": pd.to_numeric(test["target_exit_price"], errors="coerce").to_numpy(),
            "target_valid": _valid_bool(test, "target_valid", False).to_numpy(),
            "causal_valid": _valid_bool(test, "causal_valid", False).to_numpy(),
            "close": pd.to_numeric(test["close"], errors="coerce").to_numpy(),
            "target_entry_ts": pd.to_datetime(
                test["target_entry_ts"], utc=True, errors="coerce"
            ).to_numpy(),
            "target_exit_ts": pd.to_datetime(
                test["target_exit_ts"], utc=True, errors="coerce"
            ).to_numpy(),
            "minutes_until_session_close": pd.to_numeric(
                test["minutes_until_session_close"], errors="coerce"
            ).to_numpy(),
        }
    )
    for column in ("p_long", "p_short", "p_flat", "p_fade_success", "p_trend_danger"):
        if probability_cols is None:
            out[column] = np.nan
        else:
            out[column] = probability_cols[column]
    return out[PREDICTION_COLUMNS]


def _fold_masks(frame: pd.DataFrame, fold: Mapping[str, Any], y: pd.Series) -> tuple[pd.Series, pd.Series]:
    ts = frame["ts"]
    train_start = _utc(fold["train_start"])
    purged_train_end = _utc(fold["purged_train_end"])
    test_start = _utc(fold["test_start"])
    test_end = _utc(fold["test_end"])

    train_valid = (
        _valid_bool(frame, "training_row_valid", False)
        & _valid_bool(frame, "causal_valid", False)
        & _valid_bool(frame, "target_valid", False)
        & y.notna()
    )
    test_valid = (
        _valid_bool(frame, "feature_input_valid", True)
        & _valid_bool(frame, "causal_valid", False)
        & _valid_bool(frame, "target_valid", False)
        & y.notna()
    )
    train_mask = (ts >= train_start) & (ts <= purged_train_end) & train_valid
    test_mask = (ts >= test_start) & (ts <= test_end) & test_valid
    return train_mask, test_mask


def run_wfa(
    *,
    profile: str,
    matrix: str,
    run: str,
    input_root: Path,
    split_plan: Path,
    predictions_root: Path,
    reports_root: Path,
    models_config: Path,
    feature_cols_path: Path | None = None,
    max_folds: int | None = None,
) -> dict[str, Any]:
    if matrix != "baseline":
        raise SystemExit("only baseline matrix is supported in the initial WFA runner")
    feature_cols, resolved_feature_cols_path = load_feature_cols(input_root, feature_cols_path)
    model_specs, model_config = load_model_specs(models_config)
    split_manifest = _read_json(split_plan)
    folds = split_manifest.get("folds", [])
    if not isinstance(folds, list) or not folds:
        raise SystemExit(f"split plan has no folds: {_relative_path(split_plan)}")
    failures: list[str] = []
    selectable_folds: list[Mapping[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    for fold in folds:
        if not isinstance(fold, Mapping):
            failures.append("invalid fold entry")
            continue
        if "selection_allowed" not in fold:
            failures.append(f"{fold.get('fold_id', '<unknown>')}: missing selection_allowed")
            continue
        split_group = str(fold.get("split_group", ""))
        if fold.get("selection_allowed") is True and split_group == "research":
            selectable_folds.append(fold)
        elif fold.get("selection_allowed") is True:
            failures.append(
                f"{fold.get('fold_id', '<unknown>')}: non-research split_group {split_group!r} "
                "cannot be selection_allowed"
            )
            skipped_folds.append(
                {
                    "fold_id": str(fold.get("fold_id", "<unknown>")),
                    "market": str(fold.get("market", "")),
                    "split_group": split_group,
                    "reason": "selection_allowed true on non-research split",
                }
            )
        else:
            skipped_folds.append(
                {
                    "fold_id": str(fold.get("fold_id", "<unknown>")),
                    "market": str(fold.get("market", "")),
                    "split_group": split_group,
                    "reason": "selection_allowed is false",
                }
            )
    if not selectable_folds:
        failures.append("no selectable research folds in split plan")
    if max_folds is not None:
        selectable_folds = selectable_folds[:max_folds]

    years_by_market: dict[str, set[int]] = {}
    for fold in selectable_folds:
        market = str(fold["market"])
        years_by_market.setdefault(market, set()).update(int(year) for year in split_manifest["years"])

    source_columns = _required_source_columns(feature_cols, model_specs)
    frames: dict[str, pd.DataFrame] = {}
    input_paths: list[Path] = [split_plan, resolved_feature_cols_path, models_config]
    for market, years in years_by_market.items():
        frame, market_failures, paths = _load_market_frame(market, years, input_root, source_columns)
        input_paths.extend(paths)
        failures.extend(market_failures)
        if frame is not None:
            frames[market] = frame

    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for fold in selectable_folds:
        market = str(fold["market"])
        frame = frames.get(market)
        if frame is None:
            continue
        for spec in model_specs:
            target = _target_series(frame, spec)
            train_mask, test_mask = _fold_masks(frame, fold, target)
            train = frame.loc[train_mask]
            test = frame.loc[test_mask]
            detail: dict[str, Any] = {
                "fold_id": str(fold["fold_id"]),
                "market": market,
                "split_group": str(fold.get("split_group", "")),
                "model_id": spec.model_id,
                "model_family": spec.family,
                "target_name": spec.target,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "fit_ts_min": train["ts"].min().isoformat() if not train.empty else None,
                "fit_ts_max": train["ts"].max().isoformat() if not train.empty else None,
                "score_ts_min": test["ts"].min().isoformat() if not test.empty else None,
                "score_ts_max": test["ts"].max().isoformat() if not test.empty else None,
                "fit_estimator": None,
                "status": "PASS",
                "warnings": [],
            }
            if train.empty or test.empty:
                detail["status"] = "SKIP"
                detail["warnings"].append("empty train or test rows")
                diagnostics.append(detail)
                continue
            if train["ts"].max() >= test["ts"].min():
                detail["status"] = "FAIL"
                failures.append(f"{detail['fold_id']} {spec.model_id}: train/test timestamp overlap")
                diagnostics.append(detail)
                continue

            x_train = train[feature_cols]
            x_test = test[feature_cols]
            y_train = _target_series(train, spec)
            y_test = _target_series(test, spec)
            y_train_non_null = pd.Series(y_train).dropna()
            detail["y_train_unique"] = int(y_train_non_null.nunique())
            if spec.task == "classification":
                detail["y_train_class_counts"] = {
                    str(key): int(value)
                    for key, value in y_train_non_null.value_counts().sort_index().items()
                }
            estimator, actual_estimator = _build_estimator(spec, y_train)
            detail["fit_estimator"] = actual_estimator
            detail["dummy_fallback_used"] = actual_estimator == "dummy_class_prior"
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x_train, y_train)
            detail["warnings"].extend(str(item.message).splitlines()[0] for item in caught_warnings)
            if caught_warnings:
                detail["status"] = "FAIL"
                failures.append(
                    f"{detail['fold_id']} {spec.model_id}: estimator emitted convergence warning"
                )
                diagnostics.append(detail)
                continue
            detail["train_feature_means_sample"] = {
                column: float(pd.to_numeric(x_train[column], errors="coerce").mean())
                for column in feature_cols[:5]
            }

            if spec.task == "regression":
                raw_pred = estimator.predict(x_test)
                probability_cols = None
                detail["prediction_std"] = _finite_std(np.asarray(raw_pred, dtype=float))
            else:
                raw_pred, probability_cols = _classification_predictions(estimator, spec, x_test)
                collapse_failure = classifier_collapse_failure(
                    spec=spec,
                    actual_estimator=actual_estimator,
                    raw_pred=np.asarray(raw_pred, dtype=float),
                    probability_cols=probability_cols,
                )
                detail["prediction_std"] = _finite_std(np.asarray(raw_pred, dtype=float))
                detail["probability_std_by_column"] = {
                    column: _finite_std(np.asarray(values, dtype=float))
                    for column, values in probability_cols.items()
                }
                if collapse_failure is not None:
                    detail["status"] = "FAIL"
                    detail["warnings"].append(collapse_failure)
                    failures.append(f"{detail['fold_id']} {collapse_failure}")
                    diagnostics.append(detail)
                    continue
            predictions.append(
                _prediction_frame(
                    test,
                    spec,
                    fold,
                    y_test,
                    np.asarray(raw_pred, dtype=float),
                    probability_cols,
                    _file_hash_or_missing(resolved_feature_cols_path),
                )
            )
            detail["prediction_rows"] = int(len(test))
            diagnostics.append(detail)

    output_path = predictions_root / run / "oos_predictions.parquet"
    prediction_count = 0
    duplicate_count = 0
    if predictions:
        output = pd.concat(predictions, ignore_index=True)
        duplicate_count = int(
            output.duplicated(
                subset=["market", "timestamp", "fold_id", "model_id", "target_name"]
            ).sum()
        )
        if duplicate_count:
            failures.append(f"duplicate prediction rows: {duplicate_count}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(f"{output_path.name}.tmp")
        output.to_parquet(tmp_path, index=False)
        tmp_path.replace(output_path)
        prediction_count = int(len(output))
    else:
        failures.append("no prediction rows generated")
    output_hashes = (
        _file_hash_map([output_path])
        if prediction_count > 0
        else {_relative_path(output_path): "NOT_WRITTEN"}
    )
    stale_output = _stale_prediction_output_info(output_path) if prediction_count == 0 else {
        "stale_output_path_exists": False,
        "stale_output_path": None,
        "stale_output_file_hash": None,
        "stale_output_mtime_utc": None,
        "stale_output_row_count": None,
        "stale_output_split_groups": [],
    }
    if stale_output["stale_output_path_exists"]:
        failures.append(
            f"stale prediction output exists from a previous run: {stale_output['stale_output_path']}"
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_path": _relative_path(Path(__file__)),
        "script_hash": _file_sha256(Path(__file__)),
        "profile": profile,
        "matrix": matrix,
        "run": run,
        "models": [spec.__dict__ for spec in model_specs],
        "feature_count": len(feature_cols),
        "fold_count": len(selectable_folds),
        "skipped_fold_count": len(skipped_folds),
        "skipped_folds": skipped_folds,
        "prediction_count": prediction_count,
        "duplicate_prediction_count": duplicate_count,
        "warning_count": sum(len(item["warnings"]) for item in diagnostics),
        "failure_count": len(failures),
        "failures": failures,
        "diagnostics": diagnostics,
        **stale_output,
    }
    manifest = {
        **{key: report[key] for key in ("generated_at", "git_commit", "script_path", "script_hash")},
        "profile": profile,
        "matrix": matrix,
        "run": run,
        "model_config_hash": _stable_hash(model_config),
        "feature_config_hash": _file_hash_or_missing(resolved_feature_cols_path),
        "split_plan_path": _relative_path(split_plan),
        "input_root": _relative_path(input_root),
        "output_root": _relative_path(predictions_root),
        "predictions_root": _relative_path(predictions_root),
        "reports_root": _relative_path(reports_root),
        "prediction_path": _relative_path(output_path),
        "input_file_hashes": _file_hash_map(input_paths),
        "output_file_hashes": output_hashes,
        "required_columns": PREDICTION_COLUMNS,
        "model_ids": [spec.model_id for spec in model_specs],
        "target_names": [spec.target for spec in model_specs],
        "fold_count": len(selectable_folds),
        "skipped_fold_count": len(skipped_folds),
        "skipped_folds": skipped_folds,
        "prediction_count": prediction_count,
        "duplicate_prediction_count": duplicate_count,
        "warning_count": report["warning_count"],
        "failure_count": len(failures),
        "failures": failures,
        **stale_output,
    }
    artifact_evidence_failures = prediction_artifact_evidence_failures(manifest)
    manifest["artifact_evidence_ready"] = not artifact_evidence_failures
    manifest["artifact_evidence_failures"] = artifact_evidence_failures
    report["artifact_evidence_ready"] = manifest["artifact_evidence_ready"]
    report["artifact_evidence_failures"] = artifact_evidence_failures
    _write_json(reports_root / f"{run}_wfa_report.json", report)
    _write_json(reports_root / f"{run}_predictions_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--matrix", default=DEFAULT_MATRIX)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT.as_posix())
    parser.add_argument("--split-plan", default=DEFAULT_SPLIT_PLAN.as_posix())
    parser.add_argument("--predictions-root", default=DEFAULT_PREDICTIONS_ROOT.as_posix())
    parser.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT.as_posix())
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG.as_posix())
    parser.add_argument("--feature-cols", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = run_wfa(
        profile=args.profile,
        matrix=args.matrix,
        run=args.run,
        input_root=Path(args.input_root),
        split_plan=Path(args.split_plan),
        predictions_root=Path(args.predictions_root),
        reports_root=Path(args.reports_root),
        models_config=Path(args.models_config),
        feature_cols_path=Path(args.feature_cols) if args.feature_cols else None,
        max_folds=args.max_folds,
    )
    status = "FAIL" if manifest["failure_count"] else "PASS"
    print(
        f"{status} WFA baseline: predictions={manifest['prediction_count']} "
        f"models={len(manifest['model_ids'])} folds={manifest['fold_count']} "
        f"failures={manifest['failure_count']}"
    )
    return 1 if manifest["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
