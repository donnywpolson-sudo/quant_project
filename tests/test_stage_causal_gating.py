import polars as pl

from pipeline.causal.gate import causal_gate_df, causal_gate_root


def test_causal_gating_flags_metadata_and_writes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "data" / "session_normalized" / "ES" / "2024.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"ts_event": [1, 2], "settlement_available_at": [1, 3], "roll_flag": [0, 1], "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1]}).write_parquet(p)
    report = causal_gate_root("data/session_normalized", "data/causally_gated_normalized")
    out = pl.read_parquet(tmp_path / "data" / "causally_gated_normalized" / "ES" / "2024.parquet")
    assert report["status"] == "PASS"
    assert "non_model_metadata_columns" in out.columns
    assert "settlement_available_at_is_available" in out.columns
    assert (tmp_path / "reports" / "causal_gating" / "causal_gating_report.json").exists()


def test_causal_gating_preserves_raw_ohlcv_and_session_columns():
    df = pl.DataFrame(
        {
            "rtype": [1, 1],
            "publisher_id": [10, 10],
            "instrument_id": [100, 100],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
            "symbol": ["ES", "ES"],
            "ts_event": [1, 2],
            "session_id": ["s1", "s1"],
            "session_date": ["2025-01-02", "2025-01-02"],
            "market": ["ES", "ES"],
            "session_timezone": ["America/Chicago", "America/Chicago"],
            "session_calendar_accuracy": ["configured", "configured"],
        }
    )
    out = causal_gate_df(df)
    for col in df.columns:
        assert col in out.columns
    for col in ["open", "high", "low", "close", "volume", "prediction_time", "earliest_execution_time"]:
        assert col in out.columns
    assert out["earliest_execution_time"].dtype == out["ts_event"].dtype
