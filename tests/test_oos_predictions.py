from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.walkforward.walkforward import (
    build_oos_prediction_frame,
    validate_oos_prediction_frame,
)


def _result() -> pl.DataFrame:
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "ts_event": [start + timedelta(minutes=5 * i) for i in range(3)],
            "prediction_prob": [0.40, 0.50, 0.60],
            "raw_signal": [-1.0, 0.0, 1.0],
            "target_15m_dir": [0, 1, 1],
            "pnl": [1.0, -1.0, 0.5],
            "position": [0.0, -1.0, 0.0],
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("prediction_prob").cast(pl.Float32),
        pl.col("raw_signal").cast(pl.Float32),
    )


def test_build_oos_prediction_frame_projects_prediction_columns_only():
    out = build_oos_prediction_frame(_result(), target_col="target_15m_dir")

    assert out.columns == ["ts_event", "prediction_prob", "raw_signal", "target_15m_dir"]
    assert out.height == 3


def test_oos_prediction_rejects_prob_out_of_range():
    bad = _result().with_columns(pl.Series("prediction_prob", [0.2, 1.2, 0.4], dtype=pl.Float32))

    with pytest.raises(RuntimeError, match="outside \\[0, 1\\]"):
        validate_oos_prediction_frame(bad.select(["ts_event", "prediction_prob", "raw_signal"]))


def test_oos_prediction_rejects_invalid_raw_signal():
    bad = _result().with_columns(pl.Series("raw_signal", [-1.0, 0.5, 1.0], dtype=pl.Float32))

    with pytest.raises(RuntimeError, match="raw_signal outside"):
        validate_oos_prediction_frame(bad.select(["ts_event", "prediction_prob", "raw_signal"]))
