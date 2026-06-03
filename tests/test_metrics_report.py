from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.analytics.aggregate import build_metrics_report


def _bt_df() -> pl.DataFrame:
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "ts_event": [start + timedelta(minutes=5 * i) for i in range(5)],
            "pnl": [0.0, 10.0, -5.0, 3.0, -1.0],
            "gross_pnl": [0.0, 11.0, -4.0, 4.0, 0.0],
            "position": [0.0, 1.0, 1.0, 0.0, -1.0],
            "prediction_prob": [0.5, 0.6, 0.7, 0.4, 0.3],
            "ret_exec": [0.0, 0.01, -0.01, 0.005, -0.002],
            "benchmark_pnl": [0.0, 1.0, -1.0, 0.5, -0.2],
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("pnl").cast(pl.Float32),
        pl.col("gross_pnl").cast(pl.Float32),
        pl.col("position").cast(pl.Float32),
        pl.col("prediction_prob").cast(pl.Float32),
        pl.col("ret_exec").cast(pl.Float32),
        pl.col("benchmark_pnl").cast(pl.Float32),
    )


def test_metrics_report_contains_required_sections():
    report = build_metrics_report(_bt_df(), context={"split": 1})

    assert report["status"] == "OK"
    assert report["rows"] == 5
    assert "net" in report
    assert "gross" in report
    assert "prediction_distribution" in report
    assert "position_distribution" in report
    assert "diagnostics" in report
    assert report["context"]["split"] == 1


def test_metrics_report_rejects_missing_required_column():
    with pytest.raises(RuntimeError, match="missing columns"):
        build_metrics_report(_bt_df().drop("prediction_prob"))


def test_metrics_report_rejects_non_finite_pnl():
    bad = _bt_df().with_columns(pl.Series("pnl", [0.0, float("nan"), 1.0, 2.0, 3.0]))

    with pytest.raises(RuntimeError, match="non-finite pnl"):
        build_metrics_report(bad)
