import json

import polars as pl

from pipeline.cli import _feature_cols
from pipeline.common.config import RootConfig
from pipeline.features.engine import load_or_build_feature_target_matrix
from pipeline.features.registry import build_column_registry, write_column_registry


def test_column_registry_separates_feature_target_metadata(tmp_path):
    df = pl.DataFrame({"ts_event": [1], "x": [1.0], "target_15m_ret": [0.1], "roll_flag": [1], "settlement_available_at": [1]})
    reg = build_column_registry(df, "baseline")
    assert "x" in reg["feature_columns"]
    assert "target_15m_ret" in reg["target_columns"]
    assert "roll_flag" in reg["metadata_columns"]
    assert "target_15m_ret" in reg["forbidden_model_columns"]
    path = tmp_path / "column_registry.json"
    write_column_registry(df, path, "baseline")
    assert json.loads(path.read_text())["source_stage"] == "baseline"


def test_metadata_ids_timing_and_prices_cannot_be_model_features():
    df = pl.DataFrame(
        {
            "ts_event": [1, 2],
            "rtype": [32, 32],
            "publisher_id": [1, 1],
            "instrument_id": [100, 100],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
            "symbol": ["ES", "ES"],
            "session_id": ["s1", "s1"],
            "session_date": ["2025-01-02", "2025-01-02"],
            "market": ["ES", "ES"],
            "session_timezone": ["America/Chicago", "America/Chicago"],
            "session_calendar_accuracy": ["configured", "configured"],
            "prediction_time": [1, 2],
            "earliest_execution_time": [2, 3],
            "execution_time": [2, 3],
            "non_model_metadata_columns": ["rtype,publisher_id", "rtype,publisher_id"],
            "x_alpha": [0.1, -0.2],
            "target_15m_ret": [0.01, -0.01],
        }
    )
    forbidden = {
        "rtype", "publisher_id", "instrument_id", "open", "high", "low", "close", "volume",
        "ts_event", "session_id", "session_date", "market", "prediction_time",
        "earliest_execution_time", "execution_time",
    }

    reg = build_column_registry(df, "baseline")
    _, engine_features, _ = load_or_build_feature_target_matrix(df, None, "target_15m_ret", {})
    cli_features = _feature_cols(df, "target_15m_ret", RootConfig())

    assert "x_alpha" in reg["feature_columns"]
    assert "x_alpha" in engine_features
    assert "x_alpha" in cli_features
    assert forbidden.isdisjoint(reg["feature_columns"])
    assert forbidden.isdisjoint(engine_features)
    assert forbidden.isdisjoint(cli_features)
