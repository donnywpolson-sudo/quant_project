from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.walkforward.walkforward import (
    validate_walkforward_oos_predictions,
    validate_walkforward_train_test,
)


def _df(start: datetime, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_event": [start + timedelta(minutes=5 * i) for i in range(n)],
            "ret_1": [float(i) for i in range(n)],
            "target_15m_dir": [i % 2 for i in range(n)],
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("ret_1").cast(pl.Float32),
        pl.col("target_15m_dir").cast(pl.Int8),
    )


def test_walkforward_train_test_validation_passes():
    train = _df(datetime(2024, 1, 1, tzinfo=timezone.utc), 20)
    test = _df(datetime(2024, 1, 2, tzinfo=timezone.utc), 5)

    out = validate_walkforward_train_test(train, test, ["ret_1"], "target_15m_dir")

    assert out["train_rows"] == 20
    assert out["test_rows"] == 5
    assert out["purged_train_rows"] > 0


def test_walkforward_rejects_overlap():
    train = _df(datetime(2024, 1, 1, tzinfo=timezone.utc), 20)
    test = _df(datetime(2024, 1, 1, 1, tzinfo=timezone.utc), 5)

    with pytest.raises(RuntimeError, match="train/test overlap"):
        validate_walkforward_train_test(train, test, ["ret_1"], "target_15m_dir")


def test_walkforward_rejects_missing_feature():
    train = _df(datetime(2024, 1, 1, tzinfo=timezone.utc), 20)
    test = _df(datetime(2024, 1, 2, tzinfo=timezone.utc), 5)

    with pytest.raises(RuntimeError, match="missing features"):
        validate_walkforward_train_test(train, test, ["feature_missing"], "target_15m_dir")


def test_oos_prediction_validation_rejects_outside_test_window():
    test = _df(datetime(2024, 1, 2, tzinfo=timezone.utc), 5)
    result = test.with_columns(pl.lit(0.5).cast(pl.Float32).alias("prediction_prob"))
    result = result.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(datetime(2024, 1, 1, tzinfo=timezone.utc)))
        .otherwise(pl.col("ts_event"))
        .alias("ts_event")
    )

    with pytest.raises(RuntimeError, match="outside test window"):
        validate_walkforward_oos_predictions(result, test)
