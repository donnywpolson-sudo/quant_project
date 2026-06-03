import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from pipeline.data_gate.manifest import build_data_manifest


REPO = Path(__file__).resolve().parents[1]


def _base(n=200):
    ts = pl.datetime_range(pl.datetime(2025, 1, 1, 9, 30), pl.datetime(2025, 7, 19, 9, 30), "1d", eager=True)
    return pl.DataFrame({
        "ts_event": ts,
        "open": [100.0 + i * 0.1 for i in range(n)],
        "high": [100.2 + i * 0.1 for i in range(n)],
        "low": [99.8 + i * 0.1 for i in range(n)],
        "close": [100.05 + i * 0.1 for i in range(n)],
        "volume": [100 + i for i in range(n)],
        "x": [float(i % 7) for i in range(n)],
    })


def _run_auto(tmp_path, root):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["CONFIG_ENV"] = "tier_0_smoke_pipeline"
    return subprocess.run([sys.executable, str(REPO / "run.py"), "--from-stage", "auto", "--data-root", str(root)], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=80)


def _write(root, df, stage):
    p = root / "ES" / "2025.parquet"
    p.parent.mkdir(parents=True)
    df.write_parquet(p)
    build_data_manifest(root, stage=stage)


def test_run_py_auto_from_ohlcv_continues_from_validated(tmp_path):
    root = tmp_path / "validated_like"
    _write(root, _base(), "validated")
    result = _run_auto(tmp_path, root)
    assert result.returncode == 0, result.stderr
    assert "[CHECKPOINT AUTO-DETECT]" in result.stdout
    assert "inferred_stage=validated" in result.stdout
    assert "continuing_from_stage=5 SESSION NORMALIZATION" in result.stdout


def test_run_py_auto_from_session_continues_from_causal_gating(tmp_path):
    root = tmp_path / "session_like"
    _write(root, _base().with_columns(
        pl.lit("s").alias("session_id"),
        pl.lit("2025-01-01").alias("session_date"),
        pl.lit("ES").alias("market"),
        pl.lit("America/Chicago").alias("session_timezone"),
        pl.lit("configured").alias("session_calendar_accuracy"),
    ), "session_normalized")
    result = _run_auto(tmp_path, root)
    assert result.returncode == 0, result.stderr
    assert "inferred_stage=session_normalized" in result.stdout
    assert "continuing_from_stage=7 CAUSAL GATING" in result.stdout


def test_run_py_auto_from_causal_continues_from_labels(tmp_path):
    root = tmp_path / "causal_like"
    df = _base().with_columns(
        pl.lit("s").alias("session_id"),
        pl.lit("2025-01-01").alias("session_date"),
        pl.lit("ES").alias("market"),
        pl.lit("America/Chicago").alias("session_timezone"),
        pl.lit("configured").alias("session_calendar_accuracy"),
        pl.col("ts_event").alias("prediction_time"),
        (pl.col("ts_event") + pl.duration(minutes=1)).alias("earliest_execution_time"),
        pl.lit("").alias("non_model_metadata_columns"),
    )
    _write(root, df, "causally_gated_normalized")
    result = _run_auto(tmp_path, root)
    assert result.returncode == 0, result.stderr
    assert "inferred_stage=causally_gated_normalized" in result.stdout
    assert "continuing_from_stage=9 TARGET / LABEL GENERATION" in result.stdout
