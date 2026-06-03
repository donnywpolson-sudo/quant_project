import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.features import engine
from pipeline.target.target import add_target_15m, add_target_daily_regime


def _feature_target_df() -> pl.DataFrame:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "ts_event": [start + timedelta(minutes=5 * i) for i in range(3)],
            "ret_1": [0.1, 0.2, 0.3],
            "target_15m_dir": [1, 0, 1],
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("ret_1").cast(pl.Float32),
        pl.col("target_15m_dir").cast(pl.Int8),
    )


def test_feature_target_matrix_cache_roundtrip(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_generate_features(_df_aligned):
        calls["n"] += 1
        return _feature_target_df()

    monkeypatch.setattr(engine, "generate_features", fake_generate_features)
    cache_path = tmp_path / "full_feature_matrix_test.parquet"

    built = engine.load_or_build_feature_target_matrix(
        pl.DataFrame({"dummy": [1]}),
        cache_path=cache_path,
        target_col="target_15m_dir",
    )

    assert calls["n"] == 1
    assert built.height == 3
    assert cache_path.exists()

    def fail_if_called(_df_aligned):
        raise AssertionError("feature builder should not run when cache exists")

    monkeypatch.setattr(engine, "generate_features", fail_if_called)
    loaded = engine.load_or_build_feature_target_matrix(
        pl.DataFrame({"dummy": [1]}),
        cache_path=cache_path,
        target_col="target_15m_dir",
    )

    assert loaded.height == 3


def test_feature_target_matrix_rejects_null_target():
    bad = _feature_target_df().with_columns(
        pl.Series("target_15m_dir", [1, None, 0], dtype=pl.Int8)
    )
    with pytest.raises(RuntimeError, match="null values in target_15m_dir"):
        engine.validate_feature_target_matrix(bad, target_col="target_15m_dir")


def test_target_15m_is_execution_aligned():
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    n = 20
    df = pl.DataFrame(
        {
            "ts_event": [start + timedelta(minutes=i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")))

    out = add_target_15m(df)

    expected = 116.0 / 101.0
    assert out["target_15m_ret"][0] == pytest.approx(math.log(expected))
    assert out["target_15m_dir"][0] == 1
    assert out["target_15m_trade_class"][0] == 1
    assert out["target_15m_ret"].null_count() == 16


def test_daily_regime_is_not_primary_label():
    df = _feature_target_df().with_columns(
        pl.Series("htf_daily_trend_slope_10", [-0.1, 0.0, 0.2], dtype=pl.Float32)
    )
    out = add_target_daily_regime(df)

    assert out["target_daily_regime"].to_list() == [-1, 0, 1]
