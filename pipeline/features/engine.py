from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


METADATA_PREFIXES = ("target_", "future_", "label_", "roll_", "continuous_")
METADATA_COLS = {
    "ts_event",
    "date",
    "session",
    "session_id",
    "session_date",
    "symbol",
    "market",
    "session_timezone",
    "session_calendar_accuracy",
    "rtype",
    "publisher_id",
    "instrument_id",
    "prediction_time",
    "earliest_execution_time",
    "execution_time",
    "non_model_metadata_columns",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_or_build_feature_target_matrix(
    df_or_path: pl.DataFrame | str | Path,
    feature_cols: list[str] | None,
    target_col: str,
    context: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, list[str], dict[str, Any]]:
    df = pl.read_parquet(df_or_path) if isinstance(df_or_path, (str, Path)) else df_or_path.clone()
    if "ts_event" in df.columns:
        df = df.sort("ts_event")
    df = _ensure_target(df, target_col)
    df = _add_baseline_features(df)
    safe = _safe_numeric_features(df, target_col)
    if feature_cols:
        safe = [c for c in feature_cols if c in safe]
    registry = {
        "target_col": target_col,
        "feature_cols": safe,
        "metadata_cols": [c for c in df.columns if c not in safe and c != target_col],
    }
    return df, safe, registry


def _ensure_target(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    if target_col in df.columns:
        return df
    price = "open" if "open" in df.columns else "close"
    if price not in df.columns:
        raise ValueError(f"cannot derive {target_col}: missing open/close")
    return df.with_columns(((pl.col(price).shift(-16) / pl.col(price).shift(-1)).log()).alias(target_col))


def _add_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    if "close" in df.columns:
        exprs += [
            pl.col("close").pct_change().fill_null(0).alias("ret_1"),
            (pl.col("close") / pl.col("close").shift(5) - 1).fill_null(0).alias("ret_5"),
        ]
    if {"high", "low", "close"}.issubset(df.columns):
        exprs.append(((pl.col("high") - pl.col("low")) / pl.col("close")).fill_null(0).alias("range_frac"))
    if "volume" in df.columns:
        exprs.append(pl.col("volume").cast(pl.Float64).pct_change().fill_null(0).alias("volume_chg"))
    return df.with_columns(exprs) if exprs else df


def _safe_numeric_features(df: pl.DataFrame, target_col: str) -> list[str]:
    out = []
    for col, dtype in zip(df.columns, df.dtypes):
        if col == target_col or col in METADATA_COLS:
            continue
        if col.startswith(METADATA_PREFIXES):
            continue
        if dtype.is_numeric():
            out.append(col)
    return out
