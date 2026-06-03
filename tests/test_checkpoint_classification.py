import json

import polars as pl

from pipeline.data.classify_checkpoint import classify_checkpoint


def _write(path, df):
    path.parent.mkdir(parents=True)
    df.write_parquet(path)


def _ohlcv():
    return pl.DataFrame({"ts_event": [1, 2], "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1]})


def test_classify_ohlcv_only_as_validated_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "src/ES/2025.parquet", _ohlcv())
    report = classify_checkpoint(tmp_path / "src")
    assert report["inferred_stage"] in {"validated_candidate", "validated"}
    assert "session_id" in report["reason"]
    assert (tmp_path / "reports/validation/checkpoint_classification_report.json").exists()


def test_classify_session_normalized_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "src/ES/2025.parquet", _ohlcv().with_columns(
        pl.lit("s").alias("session_id"),
        pl.lit("2025-01-01").alias("session_date"),
        pl.lit("ES").alias("market"),
        pl.lit("America/Chicago").alias("session_timezone"),
        pl.lit("configured").alias("session_calendar_accuracy"),
    ))
    report = classify_checkpoint(tmp_path / "src")
    assert report["inferred_stage"] == "session_normalized_candidate"
    assert "causal columns missing" in report["reason"]


def test_classify_causally_gated(tmp_path, monkeypatch):
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
    _write(tmp_path / "src/ES/2025.parquet", df)
    report = classify_checkpoint(tmp_path / "src")
    assert report["inferred_stage"] == "causally_gated_normalized"


def test_mislabeled_causal_folder_not_classified_as_causal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "data/causally_gated_normalized/ES/2025.parquet", _ohlcv())
    report = classify_checkpoint(tmp_path / "data/causally_gated_normalized")
    assert report["inferred_stage"] != "causally_gated_normalized"
