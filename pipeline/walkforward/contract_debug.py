from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.walkforward.walkforward import _coerce_boundary


DEBUG_CSV = Path("reports/validation/wfa_contract_debug.csv")
DEBUG_JSON = Path("reports/validation/wfa_contract_debug.json")

DEBUG_COLUMNS = [
    "symbol",
    "split",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "data_root",
    "input_files",
    "data_min_ts",
    "data_max_ts",
    "rows_loaded",
    "rows_for_symbol",
    "train_rows_before_contract",
    "test_rows_before_contract",
    "train_rows_after_date_filter",
    "test_rows_after_date_filter",
    "train_rows_after_target_valid",
    "test_rows_after_target_valid",
    "train_rows_after_target_nonnull",
    "test_rows_after_target_nonnull",
    "train_rows_after_feature_finite",
    "test_rows_after_feature_finite",
    "train_rows_after_purge",
    "test_rows_after_purge",
    "target_col",
    "timestamp_col",
    "missing_required_columns",
    "reason",
]


def build_wfa_contract_debug_row(
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    train_start: Any,
    train_end: Any,
    test_start: Any,
    test_end: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    ts_col = str(context.get("timestamp_col") or "ts_event")
    symbol = context.get("symbol")
    missing = [c for c in [ts_col, target_col] if c not in df.columns]

    base = df
    if symbol and "symbol" in base.columns:
        base = base.filter(pl.col("symbol").cast(pl.Utf8) == str(symbol))

    row: dict[str, Any] = {
        "symbol": symbol or "",
        "split": context.get("split_id") or context.get("split") or "",
        "train_start": _stringify(train_start),
        "train_end": _stringify(train_end),
        "test_start": _stringify(test_start),
        "test_end": _stringify(test_end),
        "data_root": context.get("data_root") or "",
        "input_files": _stringify(context.get("input_files") or []),
        "data_min_ts": "",
        "data_max_ts": "",
        "rows_loaded": df.height,
        "rows_for_symbol": base.height,
        "train_rows_before_contract": base.height,
        "test_rows_before_contract": base.height,
        "target_col": target_col,
        "timestamp_col": ts_col,
        "missing_required_columns": ",".join(missing),
    }
    for c in DEBUG_COLUMNS:
        row.setdefault(c, 0 if c.startswith(("train_rows_", "test_rows_")) else "")

    if missing:
        row["reason"] = f"missing required columns: {','.join(missing)}"
        return row

    if base.height:
        row["data_min_ts"] = _stringify(base[ts_col].min())
        row["data_max_ts"] = _stringify(base[ts_col].max())

    work = base.sort(ts_col).with_row_index("_wfa_row")
    train_date = _filter_window(work, ts_col, train_start, train_end)
    test_date = _filter_window(work, ts_col, test_start, test_end)
    row["train_rows_after_date_filter"] = train_date.height
    row["test_rows_after_date_filter"] = test_date.height

    train_valid = _apply_target_valid(train_date)
    test_valid = _apply_target_valid(test_date)
    row["train_rows_after_target_valid"] = train_valid.height
    row["test_rows_after_target_valid"] = test_valid.height

    train_target = _drop_target_nulls(train_valid, target_col)
    test_target = _drop_target_nulls(test_valid, target_col)
    row["train_rows_after_target_nonnull"] = train_target.height
    row["test_rows_after_target_nonnull"] = test_target.height

    train_finite = _filter_feature_finite(train_target, feature_cols)
    test_finite = _filter_feature_finite(test_target, feature_cols)
    row["train_rows_after_feature_finite"] = train_finite.height
    row["test_rows_after_feature_finite"] = test_finite.height

    train_purge, test_purge = _apply_purge(
        train_finite,
        test_finite,
        context=context,
    )
    row["train_rows_after_purge"] = train_purge.height
    row["test_rows_after_purge"] = test_purge.height
    row["reason"] = _infer_reason(row)
    return row


def write_wfa_contract_debug_row(row: dict[str, Any]) -> None:
    DEBUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = DEBUG_CSV.exists()
    with DEBUG_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DEBUG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: _stringify(row.get(k, "")) for k in DEBUG_COLUMNS})

    rows: list[dict[str, Any]]
    if DEBUG_JSON.exists():
        try:
            rows = json.loads(DEBUG_JSON.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                rows = []
        except Exception:
            rows = []
    else:
        rows = []
    rows.append({k: _jsonable(row.get(k, "")) for k in DEBUG_COLUMNS})
    DEBUG_JSON.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def diagnose_and_write_wfa_contract_failure(*args: Any, **kwargs: Any) -> dict[str, Any]:
    row = build_wfa_contract_debug_row(*args, **kwargs)
    write_wfa_contract_debug_row(row)
    return row


def _filter_window(df: pl.DataFrame, ts_col: str, start: Any, end: Any) -> pl.DataFrame:
    out = df
    dtype = df[ts_col].dtype
    if start is not None:
        out = out.filter(pl.col(ts_col) >= pl.lit(_coerce_boundary(start, dtype)).cast(dtype))
    if end is not None:
        out = out.filter(pl.col(ts_col) < pl.lit(_coerce_boundary(end, dtype)).cast(dtype))
    return out


def _apply_target_valid(df: pl.DataFrame) -> pl.DataFrame:
    if "target_valid" not in df.columns:
        return df
    return df.filter(pl.col("target_valid").fill_null(False).cast(pl.Boolean))


def _drop_target_nulls(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    if target_col not in df.columns:
        return df.head(0)
    return df.drop_nulls([target_col])


def _filter_feature_finite(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    cols = [c for c in feature_cols if c in df.columns and df[c].dtype.is_numeric()]
    if not cols or df.is_empty():
        return df
    mask = pl.all_horizontal([pl.col(c).cast(pl.Float64).is_finite().fill_null(False) for c in cols])
    return df.filter(mask)


def _apply_purge(df_train: pl.DataFrame, df_test: pl.DataFrame, *, context: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    cfg = context.get("config")
    if cfg is None:
        return df_train, df_test
    train = df_train
    test = df_test
    embargo = int(getattr(getattr(cfg, "walkforward", object()), "embargo_bars", 0))
    purge_overlap = bool(getattr(getattr(cfg, "walkforward", object()), "purge_target_overlap", True))
    horizon = int(getattr(getattr(cfg, "target", object()), "target_15m_horizon", 0))
    lag = int(getattr(getattr(cfg, "execution", object()), "entry_lag_bars", 1))
    if embargo > 0 and train.height and test.height:
        boundary = int(test["_wfa_row"].min())
        train = train.filter(pl.col("_wfa_row") < boundary - embargo)
    if purge_overlap and test.height:
        last_allowed = int(test["_wfa_row"].max()) - horizon - lag
        test = test.filter(pl.col("_wfa_row") <= last_allowed)
    return train, test


def _infer_reason(row: dict[str, Any]) -> str:
    if row.get("missing_required_columns"):
        return f"missing required columns: {row['missing_required_columns']}"
    if int(row["train_rows_after_date_filter"]) == 0:
        return "train window outside feature matrix coverage"
    if int(row["test_rows_after_date_filter"]) == 0:
        return "test window outside feature matrix coverage"
    if int(row["train_rows_after_target_valid"]) == 0 or int(row["test_rows_after_target_valid"]) == 0:
        return "target_valid removed all rows"
    if int(row["train_rows_after_target_nonnull"]) == 0 or int(row["test_rows_after_target_nonnull"]) == 0:
        return "target column all null"
    if int(row["train_rows_after_feature_finite"]) == 0 or int(row["test_rows_after_feature_finite"]) == 0:
        return "feature finite filter removed all rows"
    if int(row["train_rows_after_purge"]) == 0 and int(row["train_rows_after_feature_finite"]) > 0:
        return "purge removed all train rows"
    if int(row["test_rows_after_purge"]) == 0 and int(row["test_rows_after_feature_finite"]) > 0:
        return "purge removed all test rows"
    if int(row["train_rows_after_purge"]) == 0 or int(row["test_rows_after_purge"]) == 0:
        return "empty train/test after walkforward contract"
    return "PASS"


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "|".join(_stringify(v) for v in value)
    if isinstance(value, datetime):
        return value.isoformat()
    return "" if value is None else str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _stringify(value)
