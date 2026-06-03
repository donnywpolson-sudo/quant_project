import polars as pl

from pipeline.data.adopt_checkpoint import adopt_checkpoint


def _causal_df():
    return pl.DataFrame({
        "ts_event": [1, 2],
        "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1],
        "session_id": ["s", "s"], "session_date": ["2025-01-01", "2025-01-01"], "market": ["ES", "ES"],
        "session_timezone": ["America/Chicago", "America/Chicago"], "session_calendar_accuracy": ["configured", "configured"],
        "prediction_time": [1, 2], "earliest_execution_time": [2, 3], "non_model_metadata_columns": ["", ""],
    })


def test_adopt_checkpoint_canonical_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src" / "ES" / "2025.parquet"
    src.parent.mkdir(parents=True)
    _causal_df().write_parquet(src)
    report = adopt_checkpoint("causally_gated_normalized", tmp_path / "src", "data/causally_gated_normalized", copy=True)
    assert report["status"] == "PASS"
    assert (tmp_path / "data/causally_gated_normalized/ES/2025.parquet").exists()
    assert (tmp_path / "data/causally_gated_normalized/manifest.json").exists()
    assert (tmp_path / "data/causally_gated_normalized/_manifest.csv").exists()


def test_adopt_checkpoint_flat_layout_infers_market_year(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src" / "my_ES_2025_causal.parquet"
    src.parent.mkdir()
    _causal_df().write_parquet(src)
    report = adopt_checkpoint("causally_gated_normalized", tmp_path / "src", "data/causally_gated_normalized", copy=True)
    assert report["status"] == "PASS"
    assert (tmp_path / "data/causally_gated_normalized/ES/2025.parquet").exists()


def test_adopt_checkpoint_dry_run_writes_no_parquet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src" / "ES" / "2025.parquet"
    src.parent.mkdir(parents=True)
    _causal_df().write_parquet(src)
    report = adopt_checkpoint("causally_gated_normalized", tmp_path / "src", "data/causally_gated_normalized", dry_run=True)
    assert report["status"] == "PASS"
    assert not (tmp_path / "data/causally_gated_normalized/ES/2025.parquet").exists()


def test_adopt_checkpoint_refuses_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src" / "ES" / "2025.parquet"
    dst = tmp_path / "data/causally_gated_normalized/ES/2025.parquet"
    src.parent.mkdir(parents=True)
    dst.parent.mkdir(parents=True)
    _causal_df().write_parquet(src)
    _causal_df().write_parquet(dst)
    report = adopt_checkpoint("causally_gated_normalized", tmp_path / "src", "data/causally_gated_normalized", copy=True)
    assert report["status"] == "FAIL"
    assert "target exists without --force" in report["errors"][0]


def test_adopt_checkpoint_missing_market_year_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src" / "unknown.parquet"
    src.parent.mkdir()
    _causal_df().write_parquet(src)
    report = adopt_checkpoint("causally_gated_normalized", tmp_path / "src", "data/causally_gated_normalized", copy=True)
    assert report["status"] == "FAIL"
    assert "cannot infer market/year" in report["errors"][0]
