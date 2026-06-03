from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from scripts.build_phase5_features import discover_features, write_expanded_artifacts


def _write_baseline(root: Path) -> tuple[Path, Path]:
    baseline = root / "data" / "feature_matrices" / "baseline"
    wfa = root / "reports" / "wfa"
    (baseline / "ES").mkdir(parents=True, exist_ok=True)
    wfa.mkdir(parents=True, exist_ok=True)
    features = ["f_train", "f_test", "ret_1", "bar_range", "volume_zscore_15", "rsi_14"]
    targets = ["target_ret_15m", "target_valid"]
    metadata = ["market", "prediction_ts", "session_segment_id", "bar_index_in_session", "bars_remaining_in_session"]
    excluded = ["open", "high", "low", "close", "volume"]
    for name, payload in [
        ("feature_cols.json", features),
        ("target_cols.json", targets),
        ("metadata_cols.json", metadata),
        ("excluded_cols.json", excluded),
    ]:
        (baseline / name).write_text(json.dumps(payload), encoding="utf-8")

    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(100):
        target = float((i % 7) - 3) / 100.0
        in_train = i < 60
        rows.append(
            {
                "market": "ES",
                "prediction_ts": ts + timedelta(minutes=i),
                "session_segment_id": 0 if i < 50 else 1,
                "bar_index_in_session": i if i < 50 else i - 50,
                "bars_remaining_in_session": 49 - i if i < 50 else 99 - i,
                "f_train": target if in_train else 0.0,
                "f_test": 0.0 if in_train else target * 100.0,
                "ret_1": target,
                "bar_range": abs(target),
                "volume_zscore_15": float(i % 5),
                "rsi_14": float(50 + i % 10),
                "target_ret_15m": target,
                "target_valid": True,
            }
        )
    pl.DataFrame(rows).write_parquet(baseline / "ES" / "2025.parquet")
    pl.DataFrame(
        [
            {
                "fold_id": 0,
                "market": "ES",
                "train_start": ts,
                "train_end": ts + timedelta(minutes=60),
                "train_max_prediction_ts": ts + timedelta(minutes=59),
                "test_start": ts + timedelta(minutes=60),
                "test_end": ts + timedelta(minutes=90),
            }
        ]
    ).write_csv(wfa / "fold_summary.csv")
    return baseline, wfa


def test_expanded_matrix_exists_and_registry_has_no_leakage(tmp_path: Path) -> None:
    baseline, _ = _write_baseline(tmp_path)
    expanded = tmp_path / "data" / "feature_matrices" / "expanded"
    manifest = write_expanded_artifacts(baseline_root=baseline, expanded_root=expanded, markets={"ES"}, years={2025})
    assert (expanded / "ES" / "2025.parquet").exists()
    assert manifest["expanded_feature_count"] > manifest["baseline_feature_count"]
    feature_cols = json.loads((expanded / "feature_cols.json").read_text(encoding="utf-8"))
    forbidden = set(json.loads((expanded / "target_cols.json").read_text())) | set(
        json.loads((expanded / "metadata_cols.json").read_text())
    ) | set(json.loads((expanded / "excluded_cols.json").read_text()))
    assert set(feature_cols).isdisjoint(forbidden)
    assert not any(c.startswith(("target_", "future_", "label_")) for c in feature_cols)


def test_discovery_is_train_fold_only_and_frozen_subset_is_valid(tmp_path: Path) -> None:
    baseline, wfa = _write_baseline(tmp_path)
    expanded = tmp_path / "data" / "feature_matrices" / "expanded"
    reports = tmp_path / "reports" / "feature_discovery"
    frozen = tmp_path / "data" / "frozen_features"
    write_expanded_artifacts(baseline_root=baseline, expanded_root=expanded, markets={"ES"}, years={2025})
    manifest = discover_features(
        expanded_root=expanded,
        wfa_root=wfa,
        report_root=reports,
        frozen_root=frozen,
        frozen_version="test_v1",
        markets={"ES"},
        years={2025},
        max_features=2,
        corr_threshold=0.95,
    )
    assert manifest["ranking_scope"] == "train_fold_only"
    assert manifest["no_oos_test_performance_selection"] is True
    assert (reports / "feature_scores.csv").exists()
    assert (reports / "fold_feature_scores.csv").exists()
    assert (reports / "market_feature_scores.csv").exists()
    assert (reports / "feature_stability.csv").exists()
    assert (reports / "feature_correlation_report.csv").exists()
    selected = json.loads((frozen / "test_v1" / "feature_cols.json").read_text(encoding="utf-8"))
    expanded_cols = set(json.loads((expanded / "feature_cols.json").read_text(encoding="utf-8")))
    assert set(selected).issubset(expanded_cols)
    assert "f_train" in selected
    assert "f_test" not in selected
    fold_scores = pl.read_csv(reports / "fold_feature_scores.csv")
    assert fold_scores["selection_data"].unique().to_list() == ["train_fold_only"]


def test_session_rolling_features_do_not_cross_session_segments(tmp_path: Path) -> None:
    baseline, _ = _write_baseline(tmp_path)
    expanded = tmp_path / "data" / "feature_matrices" / "expanded"
    write_expanded_artifacts(baseline_root=baseline, expanded_root=expanded, markets={"ES"}, years={2025})
    df = pl.read_parquet(expanded / "ES" / "2025.parquet")
    first_seg1 = df.filter(pl.col("session_segment_id") == 1).sort("prediction_ts").head(1)
    assert first_seg1["x_ret_1_lag_1"].to_list() == [None]
