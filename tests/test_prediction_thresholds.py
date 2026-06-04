import polars as pl

from pipeline.common.config import ExecutionConfig, RootConfig
from pipeline.validation.prediction_thresholds import (
    build_threshold_candidate_economics,
    build_prediction_threshold_diagnostics,
    print_threshold_diagnostic_summary,
    write_prediction_threshold_diagnostics,
)


def test_current_threshold_larger_than_predictions_reports_near_zero_active_pct():
    cfg = RootConfig(execution=ExecutionConfig(prediction_entry_threshold=0.25))
    df = pl.DataFrame({"prediction": [-0.001, 0.0, 0.001]})
    row, _ = build_prediction_threshold_diagnostics(df, symbol="ES", split=1, config=cfg)
    assert row["active_pct_at_current_threshold"] == 0.0
    assert row["bars_above_current_long"] == 0
    assert row["bars_below_current_short"] == 0


def test_p99_candidate_threshold_activates_about_one_pct():
    preds = [i / 1_000_000 for i in range(1000)]
    df = pl.DataFrame({"prediction": preds})
    _, grid = build_prediction_threshold_diagnostics(df, symbol="ES", split=1, config=RootConfig())
    p99 = next(r for r in grid if r["threshold_type"] == "p99")
    assert 0.005 <= p99["active_bar_pct"] <= 0.015


def test_p995_candidate_threshold_activates_about_half_pct():
    preds = [i / 1_000_000 for i in range(1000)]
    df = pl.DataFrame({"prediction": preds})
    _, grid = build_prediction_threshold_diagnostics(df, symbol="ES", split=1, config=RootConfig())
    p995 = next(r for r in grid if r["threshold_type"] == "p995")
    assert 0.002 <= p995["active_bar_pct"] <= 0.008


def test_candidate_grid_includes_fixed_and_quantile_thresholds():
    df = pl.DataFrame({"prediction": [i / 1000 for i in range(-100, 101)]})
    _, grid = build_prediction_threshold_diagnostics(df, symbol="CL", split=7, config=RootConfig())
    types = {r["threshold_type"] for r in grid}
    assert {"p90", "p95", "p99", "p995", "p999", "fixed_0.001", "fixed_0.25"}.issubset(types)


def test_threshold_reports_and_summary_print(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    df = pl.DataFrame({"prediction": [i / 1000 for i in range(-100, 101)]})
    write_prediction_threshold_diagnostics(df, symbol="CL", split=1, config=RootConfig())
    print_threshold_diagnostic_summary(expected_splits=1, expected_run_id="manual")
    out = capsys.readouterr().out
    assert "[THRESHOLD DIAG] current threshold active splits=" in out
    assert "candidate threshold p99" in out
    assert (tmp_path / "reports" / "validation" / "prediction_threshold_diagnostics.csv").exists()
    assert (tmp_path / "reports" / "validation" / "threshold_candidate_grid.csv").exists()
    assert (tmp_path / "reports" / "validation" / "threshold_candidate_economics.csv").exists()


def test_threshold_candidate_economics_cost_drag_math():
    cfg = RootConfig(
        execution=ExecutionConfig(
            prediction_entry_threshold=0.25,
            commission_per_contract=1.0,
            exchange_fees_per_contract=0.0,
            slippage_ticks=0.0,
            spread_ticks=0.0,
        )
    )
    df = pl.DataFrame(
        {
            "prediction": [0.2, 0.0, -0.2],
            "target_15m_ret": [0.01, 0.01, 0.01],
            "open": [100.0, 100.0, 100.0],
        }
    )
    _, grid = build_prediction_threshold_diagnostics(df, symbol="TEST", split=1, config=cfg)
    candidate = [r for r in grid if r["threshold_type"] == "fixed_0.1"]
    econ = build_threshold_candidate_economics(df, candidate, symbol="TEST", split=1, config=cfg)[0]
    assert econ["turnover"] == 3.0
    assert econ["cost_drag"] == econ["gross_pnl"] - econ["net_pnl"]
    assert econ["cost_drag"] == 3.0
    assert econ["pnl_per_turnover"] == econ["net_pnl"] / econ["turnover"]


def test_threshold_candidate_economics_run_scoped_and_string_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANT_RUN_ID", "run_abc12345")
    df = pl.DataFrame(
        {
            "prediction": [0.2, -0.2],
            "target_15m_ret": [0.01, -0.01],
            "open": [100.0, 100.0],
        }
    )
    write_prediction_threshold_diagnostics(df, symbol="ES", split=1, config=RootConfig())
    import json

    rows = json.loads((tmp_path / "reports/validation/threshold_candidate_economics.json").read_text())
    assert rows
    assert {r["run_id"] for r in rows} == {"run_abc12345"}
    assert all(isinstance(r["run_id"], str) for r in rows)
    assert all(isinstance(r["split"], str) for r in rows)
