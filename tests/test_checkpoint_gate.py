import polars as pl

from pipeline.common.config import RootConfig
from pipeline.data_gate.checkpoint import validate_checkpoint_stage
from pipeline.data_gate.manifest import build_data_manifest


def _write(root, df):
    p = root / "ES" / "2025.parquet"
    p.parent.mkdir(parents=True)
    df.write_parquet(p)
    build_data_manifest(root, stage="causally_gated_normalized")
    return p


def _valid():
    return pl.DataFrame({
        "ts_event": [1, 2],
        "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1],
        "session_id": ["s", "s"], "session_date": ["2025-01-01", "2025-01-01"], "market": ["ES", "ES"],
        "session_timezone": ["America/Chicago", "America/Chicago"], "session_calendar_accuracy": ["configured", "configured"],
        "prediction_time": [1, 2], "earliest_execution_time": [2, 3], "non_model_metadata_columns": ["", ""],
        "foo_available_at": [1, 2],
    })


def test_causally_gated_checkpoint_gate_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    _write(root, _valid())
    report = validate_checkpoint_stage("causally_gated_normalized", str(root), RootConfig(), ["ES"], 2025, 2025)
    assert report["status"] == "PASS"


def test_causally_gated_checkpoint_gate_fails_missing_prediction_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    _write(root, _valid().drop("prediction_time"))
    report = validate_checkpoint_stage("causally_gated_normalized", str(root), RootConfig(), ["ES"], 2025, 2025)
    assert report["status"] == "FAIL"
    assert "missing prediction_time" in "; ".join(report["failures"])


def test_causally_gated_checkpoint_gate_fails_future_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    _write(root, _valid().with_columns(pl.lit(1.0).alias("future_foo")))
    report = validate_checkpoint_stage("causally_gated_normalized", str(root), RootConfig(), ["ES"], 2025, 2025)
    assert report["status"] == "FAIL"
    assert "future_foo" in "; ".join(report["failures"])


def test_causally_gated_checkpoint_gate_fails_available_after_prediction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    _write(root, _valid().with_columns(pl.lit(99).alias("bar_available_at")))
    report = validate_checkpoint_stage("causally_gated_normalized", str(root), RootConfig(), ["ES"], 2025, 2025)
    assert report["status"] == "FAIL"
    assert "bar_available_at" in "; ".join(report["failures"])


def test_checkpoint_gate_missing_manifest_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    p = root / "ES" / "2025.parquet"
    p.parent.mkdir(parents=True)
    _valid().write_parquet(p)
    report = validate_checkpoint_stage("causally_gated_normalized", str(root), RootConfig(), ["ES"], 2025, 2025)
    assert report["status"] == "FAIL"
    assert "missing manifest" in "; ".join(report["failures"])


def test_checkpoint_gate_failure_includes_remediation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/causally_gated_normalized"
    p = root / "ES" / "2025.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"ts_event": [1], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]}).write_parquet(p)
    build_data_manifest(root, stage="causally_gated_normalized")
    report = validate_checkpoint_stage("causally_gated_normalized", str(root), RootConfig(), ["ES"], 2025, 2025)
    assert report["status"] == "FAIL"
    assert "python -m pipeline.data.adopt_checkpoint --stage validated" in report["remediation"]
