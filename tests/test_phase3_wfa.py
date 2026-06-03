from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

import scripts.build_phase3_wfa as wfa
from scripts.build_phase3_wfa import (
    build_time_folds,
    build_wfa,
    fit_predict_fold,
    load_registries,
)


def _write_matrix(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    features = ["f1", "f2"]
    targets = ["target_ret_15m", "target_valid"]
    metadata = ["market", "prediction_ts", "bar_end_ts", "session_segment_id"]
    excluded = ["open", "high", "low", "close", "volume"]
    for name, payload in [
        ("feature_cols.json", features),
        ("target_cols.json", targets),
        ("metadata_cols.json", metadata),
        ("excluded_cols.json", excluded),
    ]:
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    rows = []
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(180):
        rows.append(
            {
                "market": "ES",
                "prediction_ts": ts + timedelta(days=i),
                "bar_end_ts": ts + timedelta(days=i),
                "session_segment_id": i // 10,
                "f1": float(i),
                "f2": None if i % 17 == 0 else float(i % 7),
                "target_ret_15m": float(i % 11) / 100.0,
                "target_valid": True,
                "open": 1.0,
                "close": 1.0,
            }
        )
    out = root / "ES" / "2025.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out)


def test_folds_are_time_ordered_nonoverlapping_and_purged(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _write_matrix(root)
    regs = load_registries(root)
    df = pl.read_parquet(root / "ES" / "2025.parquet").filter(pl.col("target_valid") == True)
    folds, skipped = build_time_folds(
        df,
        train_days=60,
        test_days=20,
        step_days=20,
        purge_bars=15,
        embargo_bars=15,
        min_train_rows=10,
        min_test_rows=1,
    )
    assert folds
    assert not skipped or all(r["status"] == "skipped_insufficient_rows" for r in skipped)
    for fold in folds:
        assert fold["train_start"] < fold["train_end"] <= fold["test_start"] < fold["test_end"]
        assert fold["purged_train_rows"] == 15
        assert fold["embargo_status"] == "informational_not_enforced"
    pred, _ = fit_predict_fold(df.with_row_index("row_id"), folds[0], regs["feature_cols"])
    assert pred["prediction_ts"].min() >= folds[0]["test_start"]
    assert pred["prediction_ts"].max() < folds[0]["test_end"]


def test_only_feature_cols_are_used_and_targets_metadata_excluded(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _write_matrix(root)
    regs = load_registries(root)
    assert regs["feature_cols"] == ["f1", "f2"]
    forbidden = set(regs["target_cols"]) | set(regs["metadata_cols"]) | set(regs["excluded_cols"])
    assert set(regs["feature_cols"]).isdisjoint(forbidden)


def test_wfa_outputs_are_written_and_predictions_are_oos(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    out = tmp_path / "reports" / "wfa"
    _write_matrix(root)
    manifest = build_wfa(
        root=root,
        out=out,
        markets={"ES"},
        years={2025},
        train_days=60,
        test_days=20,
        step_days=20,
        purge_bars=15,
        embargo_bars=15,
        min_train_rows=10,
        min_test_rows=1,
    )
    assert manifest["fold_count"] > 0
    assert manifest["oos_prediction_rows"] > 0
    for rel in ["split_plan.csv", "fold_summary.csv", "oos_predictions.parquet", "manifest.json"]:
        assert (out / rel).exists()
    preds = pl.read_parquet(out / "oos_predictions.parquet")
    folds = pl.read_csv(out / "fold_summary.csv").with_columns(
        [
            pl.col("test_start").str.to_datetime(time_zone="UTC"),
            pl.col("test_end").str.to_datetime(time_zone="UTC"),
        ]
    )
    for fold in folds.iter_rows(named=True):
        part = preds.filter(pl.col("fold_id") == fold["fold_id"])
        assert part.height > 0
        assert part["prediction_ts"].min() >= fold["test_start"]
        assert part["prediction_ts"].max() < fold["test_end"]


def test_train_only_pipeline_fit_and_test_only_predict(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "matrix"
    _write_matrix(root)
    regs = load_registries(root)
    df = pl.read_parquet(root / "ES" / "2025.parquet").filter(pl.col("target_valid") == True).with_row_index("row_id")
    folds, _ = build_time_folds(
        df,
        train_days=60,
        test_days=20,
        step_days=20,
        purge_bars=15,
        embargo_bars=15,
        min_train_rows=10,
        min_test_rows=1,
    )
    calls = {}

    class RecordingPipeline:
        def __init__(self, steps):
            self.steps = steps

        def fit(self, x, y):
            calls["fit_rows"] = len(x)
            calls["fit_y_rows"] = len(y)
            return self

        def predict(self, x):
            calls["predict_rows"] = len(x)
            return np.zeros(len(x), dtype=np.float32)

    monkeypatch.setattr(wfa, "Pipeline", RecordingPipeline)
    pred, summary = fit_predict_fold(df, folds[0], regs["feature_cols"])
    assert calls == {"fit_rows": 45, "fit_y_rows": 45, "predict_rows": 20}
    assert pred.height == 20
    assert summary["train_only_fit"] is True


def test_build_wfa_groups_year_files_by_market_for_default_window(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _write_matrix(root)
    rows = []
    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for i in range(430):
        rows.append(
            {
                "market": "ES",
                "prediction_ts": ts + timedelta(days=i),
                "bar_end_ts": ts + timedelta(days=i),
                "session_segment_id": i,
                "f1": float(i),
                "f2": float(i % 3),
                "target_ret_15m": float(i % 5),
                "target_valid": True,
                "open": 1.0,
                "close": 1.0,
            }
        )
    df = pl.DataFrame(rows)
    (root / "ES" / "2023.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("prediction_ts").dt.year() == 2023).write_parquet(root / "ES" / "2023.parquet")
    df.filter(pl.col("prediction_ts").dt.year() == 2024).write_parquet(root / "ES" / "2024.parquet")
    manifest = build_wfa(
        root=root,
        out=tmp_path / "reports" / "wfa",
        markets={"ES"},
        years={2023, 2024},
        train_days=365,
        test_days=30,
        step_days=30,
        purge_bars=15,
        embargo_bars=15,
        min_train_rows=10,
        min_test_rows=1,
    )
    assert manifest["file_count"] == 2
    assert manifest["fold_count"] > 0
    folds = pl.read_csv(tmp_path / "reports" / "wfa" / "fold_summary.csv")
    assert folds["years"].unique().to_list() == ["2023,2024"]
