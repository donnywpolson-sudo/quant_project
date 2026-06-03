import importlib.util
import argparse
import json

import polars as pl

from pipeline.cli import cmd_run, run_modeling_pipeline
from pipeline.common.config import PipelineConfig, RootConfig, TargetConfig


def _df(n=30):
    return pl.DataFrame(
        {
            "ts_event": list(range(n)),
            "open": [100.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [100 + i for i in range(n)],
            "x": [float(i % 5) for i in range(n)],
            "target_15m_ret": [0.01 if i % 2 else -0.01 for i in range(n)],
        }
    )


def test_full_research_mode_fails_fast_if_required_symbol_missing(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "pipeline.features.engine":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    cfg = RootConfig(pipeline=PipelineConfig(modeling_mode="full_research"))
    try:
        run_modeling_pipeline(_df(), ["x"], "target_15m_ret", 0, 10, 10, 20, {"config": cfg})
    except RuntimeError as exc:
        assert "FULL_RESEARCH MODELING FAIL: missing pipeline.features.engine.load_or_build_feature_target_matrix" in str(exc)
        assert "minimal_compatible" not in str(exc)
    else:
        raise AssertionError("expected full_research fail-fast error")


def test_minimal_compatible_mode_still_returns_required_columns():
    cfg = RootConfig()
    out = run_modeling_pipeline(_df(), ["x"], "target_15m_ret", 0, 10, 10, 20, {"config": cfg})
    for col in ["ts_event", "prediction", "prediction_prob", "raw_signal", "position", "ret_exec", "pnl", "equity_curve", "drawdown_pct"]:
        assert col in out.columns


def test_full_research_mode_runs_train_to_test_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = RootConfig(pipeline=PipelineConfig(modeling_mode="full_research"), target=TargetConfig(target_15m_horizon=1))
    ctx = {"config": cfg, "symbol": "ES", "run_id": "r1", "split_id": "1"}
    out = run_modeling_pipeline(_df(), ["x"], "target_15m_ret", 0, 15, 15, 30, ctx)
    for col in ["ts_event", "prediction", "prediction_prob", "raw_signal", "position", "ret_exec", "pnl", "gross_pnl", "costs", "equity_curve", "drawdown_pct"]:
        assert col in out.columns
    assert out["ts_event"].min() >= 15
    assert ctx["modeling_artifacts"]["selector_path"]
    assert ctx["modeling_artifacts"]["scaler_path"]


def test_full_research_cli_relaxes_train_test_schema_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = RootConfig(pipeline=PipelineConfig(modeling_mode="full_research"), target=TargetConfig(target_15m_horizon=1))
    monkeypatch.setattr("pipeline.cli._load_cfg", lambda: cfg)
    train = tmp_path / "train" / "ES" / "2025.parquet"
    test = tmp_path / "test" / "ES" / "2025.parquet"
    train.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    base = _df(120).with_columns(
        pl.col("ts_event").cast(pl.Int64),
        (pl.col("open") + 1.0).alias("high"),
        (pl.col("open") - 1.0).alias("low"),
    )
    train_df = base.filter(pl.col("ts_event") < 60).with_columns(pl.col("volume").cast(pl.Int64))
    test_df = base.filter(pl.col("ts_event") >= 60).with_columns(pl.col("volume").cast(pl.Float64))
    train_df.write_parquet(train)
    test_df.write_parquet(test)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"data": str(train), "selected_features": ["x"], "target_col": "target_15m_ret"}), encoding="utf-8")

    cmd_run(
        argparse.Namespace(
            data=str(test),
            manifest=str(manifest),
            out=str(tmp_path / "out"),
            train_start="0",
            train_end="60",
            start="60",
            end="120",
            from_stage=None,
            data_root=None,
        )
    )

    assert (tmp_path / "out" / "backtest_results.parquet").exists()
