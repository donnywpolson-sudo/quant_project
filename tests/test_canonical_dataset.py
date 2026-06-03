from datetime import datetime, timezone, timedelta

import polars as pl

from pipeline.common.config import config
from pipeline.ingest import ingest


def _canonical_df() -> pl.DataFrame:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=5 * i) for i in range(3)]
    return pl.DataFrame(
        {
            "ts_event": ts,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10, 11, 12],
            "session_id": ["2024-01-02"] * 3,
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
    )


def test_canonical_stream_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESAMPLE_FREQUENCIES", ["5m"], raising=False)
    calls = {"n": 0}

    def fake_load_all_streams_chunked(_data_glob):
        calls["n"] += 1
        return {"5m": _canonical_df()}

    monkeypatch.setattr(ingest, "load_all_streams_chunked", fake_load_all_streams_chunked)

    cache_base = tmp_path / "canonical_data_test.parquet"
    built = ingest.load_or_build_canonical_streams("unused.parquet", cache_path=cache_base)

    assert calls["n"] == 1
    assert built["5m"].height == 3
    assert (tmp_path / "canonical_data_test_5m.parquet").exists()

    def fail_if_called(_data_glob):
        raise AssertionError("raw loader should not run when canonical cache exists")

    monkeypatch.setattr(ingest, "load_all_streams_chunked", fail_if_called)
    loaded = ingest.load_or_build_canonical_streams("unused.parquet", cache_path=cache_base)

    assert loaded["5m"].height == 3


def test_aligned_continuous_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESAMPLE_FREQUENCIES", ["5m"], raising=False)
    monkeypatch.setattr(config, "MEMORY_LOG_ENABLED", False, raising=False)
    calls = {"n": 0}

    def fake_canonical(_data_glob, cache_path=None):
        calls["n"] += 1
        return {"5m": _canonical_df()}

    monkeypatch.setattr(ingest, "load_or_build_canonical_streams", fake_canonical)

    aligned_cache = tmp_path / "aligned_data_test.parquet"
    built = ingest.load_or_build_aligned_continuous_data(
        "data/L0_ohlcv_1m/ES/2024.parquet",
        aligned_cache_path=aligned_cache,
        canonical_cache_path=tmp_path / "canonical_data_test.parquet",
    )

    assert calls["n"] == 1
    assert aligned_cache.exists()
    assert "continuous_price" in built.columns
    assert "contract_multiplier" in built.columns
    assert built["contract_multiplier"].unique().to_list() == [50.0]

    def fail_if_called(_data_glob, cache_path=None):
        raise AssertionError("canonical builder should not run when aligned cache exists")

    monkeypatch.setattr(ingest, "load_or_build_canonical_streams", fail_if_called)
    loaded = ingest.load_or_build_aligned_continuous_data(
        "data/L0_ohlcv_1m/ES/2024.parquet",
        aligned_cache_path=aligned_cache,
        canonical_cache_path=tmp_path / "canonical_data_test.parquet",
    )

    assert loaded.height == built.height
    assert "continuous_price" in loaded.columns
