import polars as pl

from pipeline.data.adopt_checkpoint import adopt_checkpoint


def _ohlcv():
    return pl.DataFrame({"ts_event": [1, 2], "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1]})


def _write(path, df):
    path.parent.mkdir(parents=True)
    df.write_parquet(path)


def test_auto_adopt_ohlcv_to_validated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "src/my_ES_2025.parquet", _ohlcv())
    report = adopt_checkpoint("auto", tmp_path / "src", target_root="data", copy=True)
    assert report["status"] == "PASS"
    assert report["stage"] == "validated"
    assert (tmp_path / "data/validated/ES/2025.parquet").exists()


def test_auto_adopt_session_to_session_normalized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "src/my_ES_2025.parquet", _ohlcv().with_columns(
        pl.lit("s").alias("session_id"),
        pl.lit("2025-01-01").alias("session_date"),
        pl.lit("ES").alias("market"),
        pl.lit("America/Chicago").alias("session_timezone"),
        pl.lit("configured").alias("session_calendar_accuracy"),
    ))
    report = adopt_checkpoint("auto", tmp_path / "src", target_root="data", copy=True)
    assert report["stage"] == "session_normalized"
    assert (tmp_path / "data/session_normalized/ES/2025.parquet").exists()


def test_auto_adopt_causal_to_causally_gated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _ohlcv().with_columns(
        pl.lit("s").alias("session_id"),
        pl.lit("2025-01-01").alias("session_date"),
        pl.lit("ES").alias("market"),
        pl.lit("America/Chicago").alias("session_timezone"),
        pl.lit("configured").alias("session_calendar_accuracy"),
        pl.col("ts_event").alias("prediction_time"),
        (pl.col("ts_event") + 1).alias("earliest_execution_time"),
        pl.lit("").alias("non_model_metadata_columns"),
    )
    _write(tmp_path / "src/my_ES_2025.parquet", df)
    report = adopt_checkpoint("auto", tmp_path / "src", target_root="data", copy=True)
    assert report["stage"] == "causally_gated_normalized"
    assert (tmp_path / "data/causally_gated_normalized/ES/2025.parquet").exists()
