import polars as pl

from pipeline.common.config import ExecutionConfig, RootConfig, TargetConfig, WalkforwardConfig, DataSectionConfig
from pipeline.walkforward.contract_debug import build_wfa_contract_debug_row


def _df(n=10, *, target=None, target_valid=None):
    vals = list(range(n))
    data = {
        "ts_event": vals,
        "open": [100.0 + i for i in vals],
        "high": [101.0 + i for i in vals],
        "low": [99.0 + i for i in vals],
        "close": [100.0 + i for i in vals],
        "volume": [100 + i for i in vals],
        "x": [float(i) for i in vals],
        "target_15m_ret": target if target is not None else [0.01] * n,
    }
    if target_valid is not None:
        data["target_valid"] = target_valid
    return pl.DataFrame(data)


def _ctx(cfg=None):
    return {"config": cfg or RootConfig(), "symbol": "ES", "split_id": 3}


def test_split_outside_data_coverage_reason():
    row = build_wfa_contract_debug_row(
        _df(),
        feature_cols=["x"],
        target_col="target_15m_ret",
        train_start=0,
        train_end=5,
        test_start=30,
        test_end=40,
        context=_ctx(),
    )
    assert row["reason"] == "test window outside feature matrix coverage"


def test_target_valid_all_false_reason():
    row = build_wfa_contract_debug_row(
        _df(target_valid=[False] * 10),
        feature_cols=["x"],
        target_col="target_15m_ret",
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=10,
        context=_ctx(),
    )
    assert row["reason"] == "target_valid removed all rows"


def test_target_all_null_reason():
    row = build_wfa_contract_debug_row(
        _df(target=[None] * 10),
        feature_cols=["x"],
        target_col="target_15m_ret",
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=10,
        context=_ctx(),
    )
    assert row["reason"] == "target column all null"


def test_purge_removes_train_rows_reason():
    cfg = RootConfig(
        walkforward=WalkforwardConfig(embargo_bars=10, purge_target_overlap=False),
        target=TargetConfig(target_15m_horizon=0),
        execution=ExecutionConfig(entry_lag_bars=0),
    )
    row = build_wfa_contract_debug_row(
        _df(),
        feature_cols=["x"],
        target_col="target_15m_ret",
        train_start=0,
        train_end=5,
        test_start=5,
        test_end=10,
        context=_ctx(cfg),
    )
    assert row["reason"] == "purge removed all train rows"


def test_feasibility_check_fails_before_subprocess_for_impossible_split(tmp_path, monkeypatch):
    import run

    monkeypatch.chdir(tmp_path)
    run._VERIFICATION_TABLE.clear()
    run._RUN_SPLIT_ARTIFACTS.clear()
    root = tmp_path / "data" / "feature_matrices" / "baseline"
    p = root / "ES" / "2023.parquet"
    p.parent.mkdir(parents=True)
    _df(10).write_parquet(p)
    cfg = RootConfig(
        symbols=["ES"],
        data=DataSectionConfig(root=str(root)),
        target=TargetConfig(target_15m_horizon=0),
        execution=ExecutionConfig(entry_lag_bars=0),
    )
    result = run._check_wfa_feasibility(
        cfg,
        [([2023], [2023], 0, 5, 30, 40)],
        [p],
    )
    assert result["status"] == "FAIL"
    assert result["failures"][0]["reason"] == "test window outside feature matrix coverage"
    assert (tmp_path / "reports" / "validation" / "failure_reasons.csv").exists()
