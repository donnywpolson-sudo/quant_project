from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

import scripts.build_phase6_final_wfa as final_wfa
from scripts.build_phase6_final_wfa import build_final_wfa, fit_predict_fold, load_final_registries
from scripts.build_phase3_wfa import build_time_folds


def _write_expanded(root: Path) -> tuple[Path, Path]:
    expanded = root / "data" / "feature_matrices" / "expanded"
    frozen = root / "data" / "frozen_features" / "phase5_v1"
    (expanded / "ES").mkdir(parents=True, exist_ok=True)
    frozen.mkdir(parents=True, exist_ok=True)
    expanded_features = ["f1", "f2", "x_unused"]
    frozen_features = ["f2", "f1"]
    for name, payload in [
        ("feature_cols.json", expanded_features),
        ("target_cols.json", ["target_ret_15m", "target_valid"]),
        ("metadata_cols.json", ["market", "prediction_ts", "bar_end_ts", "session_segment_id"]),
        ("excluded_cols.json", ["open", "high", "low", "close", "volume"]),
    ]:
        (expanded / name).write_text(json.dumps(payload), encoding="utf-8")
    (frozen / "feature_cols.json").write_text(json.dumps(frozen_features), encoding="utf-8")

    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(180):
        rows.append(
            {
                "market": "ES",
                "prediction_ts": ts + timedelta(days=i),
                "bar_end_ts": ts + timedelta(days=i),
                "session_segment_id": i // 10,
                "f1": float(i),
                "f2": None if i % 13 == 0 else float(i % 5),
                "x_unused": float(i % 7),
                "target_ret_15m": float(i % 11) / 100.0,
                "target_valid": True,
                "open": 1.0,
                "close": 1.0,
            }
        )
    pl.DataFrame(rows).write_parquet(expanded / "ES" / "2025.parquet")
    return expanded, frozen / "feature_cols.json"


def test_final_wfa_uses_exact_frozen_features_and_outputs_not_baseline_wfa(tmp_path: Path) -> None:
    expanded, frozen_cols = _write_expanded(tmp_path)
    out = tmp_path / "reports" / "final_wfa"
    manifest = build_final_wfa(
        expanded_root=expanded,
        frozen_feature_cols_path=frozen_cols,
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
    assert manifest["output_root"].endswith("final_wfa")
    assert not manifest["output_root"].endswith("reports/wfa")
    assert manifest["feature_count"] == 2
    assert manifest["x_columns_equal_frozen_feature_cols"] is True
    assert manifest["frozen_feature_cols_subset_of_expanded"] is True
    for rel in ["split_plan.csv", "fold_summary.csv", "oos_predictions.parquet", "manifest.json"]:
        assert (out / rel).exists()
    preds = pl.read_parquet(out / "oos_predictions.parquet")
    folds = pl.read_csv(out / "fold_summary.csv").with_columns(
        [pl.col("test_start").str.to_datetime(time_zone="UTC"), pl.col("test_end").str.to_datetime(time_zone="UTC")]
    )
    for fold in folds.iter_rows(named=True):
        part = preds.filter(pl.col("fold_id") == fold["fold_id"])
        assert part["prediction_ts"].min() >= fold["test_start"]
        assert part["prediction_ts"].max() < fold["test_end"]


def test_frozen_features_exclude_target_metadata_excluded_and_subset_expanded(tmp_path: Path) -> None:
    expanded, frozen_cols = _write_expanded(tmp_path)
    regs = load_final_registries(expanded, frozen_cols)
    assert regs["feature_cols"] == ["f2", "f1"]
    forbidden = set(regs["target_cols"]) | set(regs["metadata_cols"]) | set(regs["excluded_cols"])
    assert set(regs["feature_cols"]).isdisjoint(forbidden)
    assert set(regs["feature_cols"]).issubset(set(regs["expanded_feature_cols"]))


def test_purge_no_overlap_and_train_only_fit(monkeypatch, tmp_path: Path) -> None:
    expanded, frozen_cols = _write_expanded(tmp_path)
    regs = load_final_registries(expanded, frozen_cols)
    df = pl.read_parquet(expanded / "ES" / "2025.parquet").sort("prediction_ts").filter(pl.col("target_valid") == True)
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
            calls["fit_cols"] = x.shape[1]
            calls["fit_y_rows"] = len(y)
            return self

        def predict(self, x):
            calls["predict_rows"] = len(x)
            calls["predict_cols"] = x.shape[1]
            return np.zeros(len(x), dtype=np.float32)

    monkeypatch.setattr(final_wfa, "Pipeline", RecordingPipeline)
    pred, summary = fit_predict_fold(df, folds[0], regs["feature_cols"])
    assert calls == {"fit_rows": 45, "fit_cols": 2, "fit_y_rows": 45, "predict_rows": 20, "predict_cols": 2}
    assert summary["train_only_fit"] is True
    assert summary["feature_count"] == 2
    assert folds[0]["purged_train_rows"] == 15
    assert pred["prediction_ts"].min() >= folds[0]["test_start"]
    assert pred["prediction_ts"].max() < folds[0]["test_end"]
