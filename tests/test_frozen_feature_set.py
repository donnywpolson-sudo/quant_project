import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from pipeline.common.config import DataSectionConfig, RootConfig
from pipeline.features.frozen import create_frozen_feature_set, validate_frozen_feature_set
from pipeline.stage_status import build_pipeline_status


def _cfg(root: Path) -> RootConfig:
    cfg = RootConfig(
        symbols=["ES"],
        start_year=2025,
        end_year=2025,
        data=DataSectionConfig(root=str(root)),
    )
    cfg.discovery.max_selected_features = 2
    return cfg


def _write_expanded(root: Path) -> None:
    start = datetime(2025, 1, 1)
    df = pl.DataFrame(
        {
            "ts_event": [start + timedelta(days=i) for i in range(160)],
            "open": [100.0 + i for i in range(160)],
            "high": [101.0 + i for i in range(160)],
            "low": [99.0 + i for i in range(160)],
            "close": [100.5 + i for i in range(160)],
            "volume": [1000 + i for i in range(160)],
            "target_15m_ret": [float((i % 5) - 2) for i in range(160)],
            "target_valid": [True] * 160,
            "ret_lag_1": [float((i % 5) - 2) for i in range(160)],
            "roll_vol_5": [float(i % 7) for i in range(160)],
            "future_bad": [float(i) for i in range(160)],
        }
    )
    path = root / "ES" / "2025.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    baseline = Path("data/feature_matrices/baseline/ES/2025.parquet")
    baseline.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(baseline)
    Path("data/feature_matrices/baseline/column_registry.json").write_text(
        json.dumps({"feature_columns": ["ret_lag_1", "roll_vol_5"], "target_columns": ["target_15m_ret"]}),
        encoding="utf-8",
    )
    Path("reports/validation").mkdir(parents=True, exist_ok=True)
    Path("reports/validation/stage_21_feature_discovery_audit_report.json").write_text(
        json.dumps({"status": "PASS", "feature_cols": ["ret_lag_1", "roll_vol_5"]}),
        encoding="utf-8",
    )
    Path("reports/validation/stage_22_train_only_selection_audit_report.json").write_text(
        json.dumps({"status": "PASS", "selection_method": "train_abs_corr"}),
        encoding="utf-8",
    )


def _stage(rows, idx):
    return next(r for r in rows if str(r["stage_index"]) == str(idx))


def test_stage23_missing_reports_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/feature_matrices/expanded"
    _write_expanded(root)

    row = _stage(build_pipeline_status(_cfg(root), data_root="data/feature_matrices/baseline"), 23)

    assert row["status"] == "MISSING"
    assert "missing artifacts" in row["reason"] or "expected output missing" in row["reason"]


def test_valid_frozen_feature_set_reports_pass_and_final_eligible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/feature_matrices/expanded"
    _write_expanded(root)
    cfg = _cfg(root)

    result = create_frozen_feature_set(config=cfg, run_id="run_test", profile="tier_1_bare_minimum_alpha", source_feature_matrix_root=root)
    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert result["status"] == "PASS"
    assert _stage(rows, 23)["status"] == "PASS"
    assert _stage(rows, 24)["status"] == "MISSING"
    assert _stage(rows, 24)["upstream_stage_status"] == "PASS"


def test_selected_features_cannot_include_target_metadata_or_excluded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/feature_matrices/expanded"
    _write_expanded(root)
    out = Path("data/frozen_features/phase5_v1")
    out.mkdir(parents=True)
    (out / "feature_cols.json").write_text(json.dumps({"feature_cols": ["target_15m_ret", "open", "future_bad"]}), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"train_only": True, "leakage_check": "PASS", "target_col": "target_15m_ret"}), encoding="utf-8")
    (out / "selected_features.csv").write_text("feature\nfuture_bad\n", encoding="utf-8")
    (out / "rejected_features.csv").write_text("feature\nret_lag_1\n", encoding="utf-8")

    result = validate_frozen_feature_set(output_root=out, source_feature_matrix_root=root, config=_cfg(root))

    assert result["status"] == "FAIL"
    assert "forbidden" in result["reason"]


def test_manifest_must_be_train_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/feature_matrices/expanded"
    _write_expanded(root)
    out = Path("data/frozen_features/phase5_v1")
    out.mkdir(parents=True)
    (out / "feature_cols.json").write_text(json.dumps({"feature_cols": ["ret_lag_1"]}), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"train_only": False, "leakage_check": "PASS", "target_col": "target_15m_ret"}), encoding="utf-8")
    (out / "selected_features.csv").write_text("feature\nret_lag_1\n", encoding="utf-8")
    (out / "rejected_features.csv").write_text("feature\nroll_vol_5\n", encoding="utf-8")

    result = validate_frozen_feature_set(output_root=out, source_feature_matrix_root=root, config=_cfg(root))

    assert result["status"] == "FAIL"
    assert "train_only" in result["reason"]


def test_stages_24_27_skipped_when_stage23_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/feature_matrices/expanded"
    _write_expanded(root)
    rows = build_pipeline_status(_cfg(root), data_root="data/feature_matrices/baseline")

    assert _stage(rows, 23)["status"] == "MISSING"
    assert all(_stage(rows, i)["status"] == "SKIPPED" for i in range(24, 28))
