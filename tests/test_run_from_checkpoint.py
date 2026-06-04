import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from pipeline.common.config import PipelineConfig, RootConfig, TargetConfig
from pipeline.data_gate.manifest import build_data_manifest
from pipeline.labels.generate import label_root
from pipeline.orchestration.downstream import run_from_causally_gated_checkpoint


REPO = Path(__file__).resolve().parents[1]


def _causal_df(n=1200):
    start = datetime(2025, 1, 1, 9, 30)
    ts = pl.Series([start + timedelta(minutes=i) for i in range(n)])
    exec_ts = pl.Series([start + timedelta(minutes=i + 1) for i in range(n)])
    return pl.DataFrame({
        "ts_event": ts,
        "open": [5000.0 + i * 0.1 for i in range(n)],
        "high": [5000.2 + i * 0.1 for i in range(n)],
        "low": [4999.8 + i * 0.1 for i in range(n)],
        "close": [5000.05 + i * 0.1 for i in range(n)],
        "volume": [100 + i for i in range(n)],
        "session_id": ["s"] * n,
        "session_date": [str(d.date()) for d in ts],
        "market": ["ES"] * n,
        "session_timezone": ["America/Chicago"] * n,
        "session_calendar_accuracy": ["configured"] * n,
        "prediction_time": ts,
        "earliest_execution_time": exec_ts,
        "non_model_metadata_columns": [""] * n,
        "x": [float(i % 7) for i in range(n)],
    })


def _write_checkpoint(root: Path):
    p = root / "ES" / "2025.parquet"
    p.parent.mkdir(parents=True)
    _causal_df().write_parquet(p)
    build_data_manifest(root, stage="causally_gated_normalized")
    return p


def test_run_from_causally_gated_checkpoint_writes_downstream_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    _write_checkpoint(root)
    cfg = RootConfig(symbols=["ES"], start_year=2025, end_year=2025, target=TargetConfig(target_15m_horizon=1))
    result = run_from_causally_gated_checkpoint(root, cfg, {"run_id": "r1"})
    assert result["status"] == "PASS"
    assert (tmp_path / "data/labeled/ES/2025.parquet").exists()
    assert (tmp_path / "data/feature_matrices/baseline/ES/2025.parquet").exists()
    assert (tmp_path / "data/feature_matrices/baseline/column_registry.json").exists()
    assert (tmp_path / "output/checkpoint_run/oos_predictions.parquet").exists()
    assert list((tmp_path / "reports/metrics").glob("*_metrics_report.json"))
    assert list((tmp_path / "reports/acceptance").glob("*_acceptance_gate.json"))
    assert (tmp_path / "artifacts/run_manifests/checkpoint_downstream.json").exists()
    assert result["stage_plan"][0]["status"] == "SKIPPED_CHECKPOINT"


def test_run_py_from_causally_gated_checkpoint_does_not_require_validated(tmp_path):
    root = tmp_path / "causal"
    _write_checkpoint(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["CONFIG_ENV"] = "tier_0_smoke_pipeline"
    env["QUANT_MODELING_MODE"] = "minimal_compatible"
    env["QUANT_START_STAGE"] = "causally_gated_normalized"
    env["QUANT_DATA_ROOT"] = str(root)
    result = subprocess.run([sys.executable, str(REPO / "run.py"), "--from-stage", "causally_gated_normalized", "--data-root", str(root)], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "[CHECKPOINT START]" in result.stdout
    assert "DATA PREFLIGHT FAIL" not in result.stderr
    manifest = sorted((tmp_path / "artifacts/run_manifests").glob("*.json"))[-1]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["start_stage"] == "causally_gated_normalized"
    assert any(s["status"] == "SKIPPED_CHECKPOINT" for s in payload["stages"][:8])


def test_full_research_from_checkpoint_missing_dependency_fails_fast(monkeypatch, tmp_path):
    import importlib.util
    from pipeline.cli import run_modeling_pipeline

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "pipeline.features.engine" else real(name))
    cfg = RootConfig(pipeline=PipelineConfig(modeling_mode="full_research"))
    try:
        run_modeling_pipeline(_causal_df(), ["x"], "target_15m_ret", 0, 20, 20, 40, {"config": cfg})
    except RuntimeError as exc:
        assert "FULL_RESEARCH MODELING FAIL: missing pipeline.features.engine.load_or_build_feature_target_matrix" in str(exc)
    else:
        raise AssertionError("expected fail-fast")


def test_cached_labels_reused_then_invalidated_by_source_manifest(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    _write_checkpoint(root)
    cfg = RootConfig(symbols=["ES"], start_year=2025, end_year=2025, target=TargetConfig(target_15m_horizon=1))
    first = label_root(root, "data/labeled", cfg)
    assert first["files"][0]["status"] == "PASS"
    second = label_root(root, "data/labeled", cfg)
    assert second["files"][0]["status"] == "COMPLETED_CACHED"
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    raw["changed"] = True
    (root / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    third = label_root(root, "data/labeled", cfg)
    assert third["files"][0]["status"] == "PASS"
