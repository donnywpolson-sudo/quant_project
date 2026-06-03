from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.features import discovery


def _df() -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "ts_event": [start + timedelta(days=i) for i in range(5)],
            "ret_1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target_15m_dir": [1, 0, 1, 0, 1],
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    )


def test_filter_feature_matrix_to_train_window():
    out = discovery.filter_feature_matrix_to_train_window(
        _df(),
        "2024-01-02T00:00:00+00:00",
        "2024-01-04T00:00:00+00:00",
    )

    assert out.height == 2
    assert str(out["ts_event"].min()) == "2024-01-02 00:00:00+00:00"
    assert str(out["ts_event"].max()) == "2024-01-03 00:00:00+00:00"


def test_train_window_requires_start_and_end():
    with pytest.raises(RuntimeError, match="provided together"):
        discovery.filter_feature_matrix_to_train_window(
            _df(),
            "2024-01-02T00:00:00+00:00",
            None,
        )


def test_train_only_discovery_refuses_missing_bounds(monkeypatch, tmp_path):
    called = {"n": 0}

    def fake_run(*_args, **_kwargs):
        called["n"] += 1

    monkeypatch.setattr(discovery, "run_feature_discovery", fake_run)

    with pytest.raises(RuntimeError, match="train_start/train_end are required"):
        discovery.run_train_only_feature_discovery(
            "features.parquet",
            str(tmp_path / "manifest.json"),
            "",
            "",
        )

    assert called["n"] == 0
