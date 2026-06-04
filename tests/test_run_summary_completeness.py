from datetime import datetime, timezone
from pathlib import Path

from pipeline.common.config import RootConfig
import run


def test_final_summary_rows_are_unique_and_missing_paths_are_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run._VERIFICATION_TABLE.clear()
    run._RUN_SPLIT_ARTIFACTS.clear()

    cfg = RootConfig(symbols=["ES", "CL"])
    split = (
        [2025],
        [2025],
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 31, tzinfo=timezone.utc),
    )
    files = []
    for symbol in cfg.symbols:
        p = tmp_path / "data" / "feature_matrices" / "baseline" / symbol / "2025.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("placeholder", encoding="utf-8")
        files.append(p)

    run._record_split_result(
        {
            "symbol": "ES",
            "split": 1,
            "path": "old",
            "pred_cs": "missing",
            "pnl_cs": "missing",
            "status": "FAILED",
            "error": "original failure",
            "rc": 1,
            "command": "python -m pipeline.cli run",
            "stderr_tail": "RuntimeError: original failure",
            "exception_type": "RuntimeError",
            "exception_message": "original failure",
        }
    )
    run._record_split_result(
        {
            "symbol": "ES",
            "split": 1,
            "path": "new",
            "pred_cs": "missing",
            "pnl_cs": "missing",
            "status": "FAILED",
            "error": "original failure",
        }
    )

    result = run._prepare_final_artifact_completeness(cfg, [split], files)

    assert result["status"] == "FAIL"
    assert len(run._VERIFICATION_TABLE) == 2
    assert len({(r["symbol"], r["split"]) for r in run._VERIFICATION_TABLE}) == 2
    csv_text = Path(result["failure_reasons_csv"]).read_text(encoding="utf-8")
    assert "original failure" in csv_text
    assert "stderr_tail" in csv_text
    assert "RuntimeError" in csv_text
    assert "backtest_results missing at" in csv_text
    assert "output" in csv_text


def test_duplicate_split_plan_entries_fail_before_execution():
    rows = [{"symbol": "ES", "split": 1}, {"symbol": "ES", "split": 1}]
    try:
        run._assert_execution_plan_unique(rows, ["ES"], [object(), object()])
    except AssertionError as exc:
        assert "DUPLICATE" in str(exc) or "ROW COUNT" in str(exc)
    else:
        raise AssertionError("expected duplicate split-plan assertion")


def test_year_boundary_splits_do_not_duplicate_symbol_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = RootConfig(symbols=["CL", "ES", "ZN"])
    files = []
    for symbol in cfg.symbols:
        for year in [2023, 2024, 2025]:
            p = tmp_path / "data" / "feature_matrices" / "baseline" / symbol / f"{year}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("placeholder", encoding="utf-8")
            files.append(p)
    splits = []
    for idx in range(1, 21):
        test_years = [2023, 2024] if idx == 7 else ([2024, 2025] if idx == 19 else [2024])
        splits.append(([2023], test_years, None, None, None, None))

    rows = run._expected_runtime_rows(cfg, splits, files)
    run._assert_execution_plan_unique(rows, cfg.symbols, splits)

    assert len(rows) == len(cfg.symbols) * len(splits)
    for split_id in [7, 19]:
        assert len([r for r in rows if r["split"] == split_id]) == len(cfg.symbols)
