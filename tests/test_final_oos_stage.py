import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from pipeline.common.config import DataSectionConfig, RootConfig
from pipeline.features.frozen import create_frozen_feature_set
from pipeline.stage_status import build_pipeline_status
from pipeline.validation.final_oos import materialize_final_oos_predictions, validate_final_oos_predictions
from pipeline.validation.final_lineage import file_sha256


def _cfg(root: Path) -> RootConfig:
    return RootConfig(symbols=["ES"], start_year=2025, end_year=2025, data=DataSectionConfig(root=str(root)))


def _write_prereqs(tmp_path: Path) -> RootConfig:
    root = tmp_path / "data/feature_matrices/expanded"
    start = datetime(2025, 1, 1)
    df = pl.DataFrame(
        {
            "ts_event": [start + timedelta(days=i) for i in range(80)],
            "open": [100.0 + i for i in range(80)],
            "high": [101.0 + i for i in range(80)],
            "low": [99.0 + i for i in range(80)],
            "close": [100.5 + i for i in range(80)],
            "volume": [1000 + i for i in range(80)],
            "target_15m_ret": [float((i % 3) - 1) for i in range(80)],
            "ret_lag_1": [float((i % 3) - 1) for i in range(80)],
        }
    )
    for out_root in [root, tmp_path / "data/feature_matrices/baseline"]:
        p = out_root / "ES" / "2025.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)
    (tmp_path / "data/feature_matrices/baseline/column_registry.json").write_text(
        json.dumps({"feature_columns": ["ret_lag_1"], "target_columns": ["target_15m_ret"]}),
        encoding="utf-8",
    )
    reports = tmp_path / "reports/validation"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage_21_feature_discovery_audit_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (reports / "stage_22_train_only_selection_audit_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    cfg = _cfg(root)
    create_frozen_feature_set(config=cfg, run_id="run_test", profile="tier_1_bare_minimum_alpha", source_feature_matrix_root=root)
    stage24 = df.with_columns(
        pl.lit("ES").alias("symbol"),
        pl.lit("1").alias("split"),
        pl.lit(0.1).alias("prediction"),
        pl.lit(0.55).alias("prediction_prob"),
        pl.col("ts_event").alias("prediction_time"),
        (pl.col("ts_event") + pl.duration(minutes=1)).alias("execution_time"),
        pl.lit(0.0).alias("pnl"),
    )
    stage24.write_parquet(reports / "stage_24_final_wfa_backtest_results.parquet")
    return cfg


def _stage(rows, idx):
    return next(r for r in rows if str(r["stage_index"]) == str(idx))


def _write_wfa_splits(count: int) -> None:
    path = Path("reports/validation/wfa_contract_debug.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["symbol,split,status"]
    rows.extend(f"ES,{i},PASS" for i in range(1, count + 1))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_stage26_lineage() -> None:
    path = Path("reports/validation/stage_26_final_metrics_diagnostics_audit_report.json")
    source = Path("reports/validation/stage_25_final_oos_predictions.parquet")
    path.write_text(
        json.dumps(
            {
                "stage": 26,
                "status": "PASS",
                "run_id": "run_test",
                "profile": "tier_1_bare_minimum_alpha",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_artifact_path": str(source),
                "source_artifact_checksum": file_sha256(source),
            }
        ),
        encoding="utf-8",
    )


def _write_stage27_lineage() -> None:
    path = Path("reports/validation/stage_27_strategy_acceptance_audit_report.json")
    source = Path("reports/validation/stage_26_final_metrics_diagnostics_audit_report.json")
    path.write_text(
        json.dumps(
            {
                "stage": 27,
                "status": "PASS",
                "run_id": "run_test",
                "profile": "tier_1_bare_minimum_alpha",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_artifact_path": str(source),
                "source_artifact_checksum": file_sha256(source),
            }
        ),
        encoding="utf-8",
    )


def test_stage25_missing_prediction_reports_clear_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_prereqs(tmp_path)
    bad = pl.DataFrame({"timestamp": [1], "run_id": ["run_test"], "profile": ["p"], "symbol": ["ES"], "split": ["1"], "target_15m_ret": [0.0]})
    bad.write_parquet("reports/validation/stage_25_final_oos_predictions.parquet")

    result = validate_final_oos_predictions()

    assert result["status"] == "FAIL"
    assert "prediction" in result["required_columns"]
    assert "available_columns" in result
    assert "Stage 25" in result["producing_stage"]


def test_stage25_passes_when_prediction_contract_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_prereqs(tmp_path)

    result = materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")

    assert result["status"] == "PASS"


def test_stage25_stale_wrong_schema_fails_and_26_27_skip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    pl.DataFrame({"prediction": [0.1]}).write_parquet("reports/validation/stage_25_final_oos_predictions.parquet")

    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert _stage(rows, 25)["status"] == "FAIL"
    assert "artifact_path=" in _stage(rows, 25)["reason"]
    assert _stage(rows, 26)["status"] == "SKIPPED"
    assert _stage(rows, 27)["status"] == "SKIPPED"


def test_stage25_pass_makes_26_27_eligible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")

    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert _stage(rows, 25)["status"] == "PASS"
    assert _stage(rows, 26)["status"] == "MISSING"
    assert _stage(rows, 26)["upstream_stage_status"] == "PASS"


def test_stage25_schema_pass_but_coverage_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")
    _write_wfa_splits(2)

    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert _stage(rows, 25)["status"] == "FAIL"
    assert "(symbol,split) coverage incomplete" in _stage(rows, 25)["reason"]


def test_stage25_row_count_too_small_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")
    _write_wfa_splits(100)

    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert _stage(rows, 25)["status"] == "FAIL"
    assert "row_count too small" in _stage(rows, 25)["reason"]


def test_stage26_missing_lineage_is_stale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")
    Path("reports/validation/stage_26_final_metrics_diagnostics_audit_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert _stage(rows, 26)["status"] == "STALE"
    assert "missing lineage" in _stage(rows, 26)["reason"]


def test_stage27_missing_lineage_is_stale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")
    _write_stage26_lineage()
    Path("reports/validation/stage_27_strategy_acceptance_audit_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert _stage(rows, 26)["status"] == "PASS"
    assert _stage(rows, 27)["status"] == "STALE"
    assert "missing lineage" in _stage(rows, 27)["reason"]


def test_matching_lineage_passes_final_stages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")
    _write_stage26_lineage()
    _write_stage27_lineage()

    rows = build_pipeline_status(cfg, data_root="data/feature_matrices/baseline")

    assert _stage(rows, 25)["status"] == "PASS"
    assert _stage(rows, 26)["status"] == "PASS"
    assert _stage(rows, 27)["status"] == "PASS"


def test_status_not_final_ready_when_final_stale(tmp_path, monkeypatch, capsys):
    from pipeline.stage_status import print_pipeline_status

    monkeypatch.chdir(tmp_path)
    cfg = _write_prereqs(tmp_path)
    materialize_final_oos_predictions(run_id="run_test", profile="tier_1_bare_minimum_alpha")
    Path("reports/validation/stage_26_final_metrics_diagnostics_audit_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    print_pipeline_status(build_pipeline_status(cfg, data_root="data/feature_matrices/baseline"))
    out = capsys.readouterr().out

    assert "Pipeline is NOT final-strategy-ready: final metrics/gate stale relative to Stage 25" in out
