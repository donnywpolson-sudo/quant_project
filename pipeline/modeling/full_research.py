from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from pipeline.features.discovery import select_features_train_only
from pipeline.features.engine import load_or_build_feature_target_matrix
from pipeline.features.preprocessing import fit_apply_train_scaler
from pipeline.execution.cost_model import attach_execution_cost_model
from pipeline.walkforward.walkforward import apply_walkforward_contract


def run_full_research_modeling(
    df_or_path: pl.DataFrame | str,
    feature_cols: list[str],
    target_col: str,
    train_start: Any,
    train_end: Any,
    test_start: Any,
    test_end: Any,
    context: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    cfg = context["config"]
    df, safe_features, registry = load_or_build_feature_target_matrix(df_or_path, feature_cols, target_col, context)
    train, test = apply_walkforward_contract(
        df,
        train_start,
        train_end,
        test_start,
        test_end,
        target_horizon_bars=int(getattr(cfg.target, "target_15m_horizon", 0)),
        embargo_bars=int(getattr(cfg.walkforward, "embargo_bars", 0)),
        purge_target_overlap=bool(getattr(cfg.walkforward, "purge_target_overlap", True)),
        entry_lag_bars=int(getattr(cfg.execution, "entry_lag_bars", 1)),
    )
    train = train.drop_nulls([target_col])
    test = test.drop_nulls([target_col])
    if train.is_empty() or test.is_empty():
        raise RuntimeError("FULL_RESEARCH MODELING FAIL: empty train/test after walkforward contract")
    selected, selector_artifact = select_features_train_only(
        train,
        test,
        safe_features,
        target_col,
        {**context, "train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end},
    )
    if not selected:
        raise RuntimeError("FULL_RESEARCH MODELING FAIL: no selected train-safe features")
    train_s, test_s, scaler_artifact = fit_apply_train_scaler(
        train,
        test,
        selected,
        {**context, "train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end},
    )
    beta, intercept = _fit_ridge(train_s, selected, target_col, float(cfg.walkforward.ridge_params.get("alpha", 1.0)))
    pred = _predict(test_s, selected, beta, intercept)
    result = _attach_execution(test_s, pred, target_col, cfg, feature_set_id=_feature_set_id(selected), symbol=context.get("symbol"))
    artifacts = {
        "feature_registry": registry,
        "selector_path": selector_artifact["path"],
        "scaler_path": scaler_artifact["path"],
        "selected_features": selected,
        "feature_set_id": _feature_set_id(selected),
    }
    return result, artifacts


def _fit_ridge(train: pl.DataFrame, features: list[str], target_col: str, alpha: float) -> tuple[np.ndarray, float]:
    x = train.select(features).to_numpy()
    y = train[target_col].cast(pl.Float64).to_numpy()
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.shape[0] < 2:
        raise RuntimeError("FULL_RESEARCH MODELING FAIL: insufficient finite train rows")
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    reg = np.eye(x_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    return coef[1:], float(coef[0])


def _predict(test: pl.DataFrame, features: list[str], beta: np.ndarray, intercept: float) -> np.ndarray:
    x = test.select(features).to_numpy()
    return x @ beta + intercept


def _attach_execution(df: pl.DataFrame, pred: np.ndarray, target_col: str, cfg: Any, feature_set_id: str, symbol: str | None = None) -> pl.DataFrame:
    out = df.with_columns(pl.Series("prediction", pred))
    out = out.with_columns(
        (1.0 / (1.0 + (-pl.col("prediction")).exp())).alias("prediction_prob"),
        pl.when(pl.col("prediction") > float(cfg.execution.prediction_entry_threshold)).then(1)
        .when(pl.col("prediction") < -float(cfg.execution.prediction_entry_threshold)).then(-1)
        .otherwise(0).alias("raw_signal"),
        pl.lit(float(cfg.execution.prediction_entry_threshold)).alias("signal_entry_threshold"),
    )
    return attach_execution_cost_model(out, target_col=target_col, config=cfg, symbol=symbol, feature_set_id=feature_set_id)


def _feature_set_id(features: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()[:16]
