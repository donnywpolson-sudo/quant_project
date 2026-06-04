import numpy as np
import polars as pl
import pytest

import pipeline.common.config as config_module
from pipeline.common.config import ExecutionConfig, RootConfig
from pipeline.validation.threshold_used import resolve_threshold_from_train, write_threshold_used


def _reset_config_loader():
    config_module._LOADED = False


def test_production_profile_remains_fixed_threshold():
    _reset_config_loader()
    cfg = config_module.load_config("tier_1_bare_minimum_alpha")
    assert cfg.execution.threshold_mode == "fixed"
    assert cfg.execution.prediction_entry_threshold == 0.25
    assert cfg.execution.threshold_quantile is None


def test_experimental_profile_uses_p995_quantile_mode():
    _reset_config_loader()
    cfg = config_module.load_config("tier_1_threshold_p995_experiment")
    assert cfg.execution.prediction_entry_threshold == 0.25
    assert cfg.execution.threshold_mode == "prediction_abs_quantile"
    assert cfg.execution.threshold_quantile == 0.995


def test_experimental_profile_uses_p999_quantile_mode():
    _reset_config_loader()
    cfg = config_module.load_config("tier_1_threshold_p999_experiment")
    assert cfg.execution.prediction_entry_threshold == 0.25
    assert cfg.execution.threshold_mode == "prediction_abs_quantile"
    assert cfg.execution.threshold_quantile == 0.999


def test_cl_es_p999_experimental_profile_config_and_expected_rows():
    _reset_config_loader()
    cfg = config_module.load_config("tier_1_threshold_p999_CL_ES_experiment")
    assert list(cfg.symbols) == ["CL", "ES"]
    assert cfg.execution.prediction_entry_threshold == 0.25
    assert cfg.execution.threshold_mode == "prediction_abs_quantile"
    assert cfg.execution.threshold_quantile == 0.999
    assert len(cfg.symbols) * 30 == 60


def test_cl_es_p997_experimental_profile_config_and_expected_rows():
    _reset_config_loader()
    cfg = config_module.load_config("tier_1_threshold_p997_CL_ES_experiment")
    assert list(cfg.symbols) == ["CL", "ES"]
    assert cfg.execution.prediction_entry_threshold == 0.25
    assert cfg.execution.threshold_mode == "prediction_abs_quantile"
    assert cfg.execution.threshold_quantile == 0.997
    assert len(cfg.symbols) * 30 == 60


def test_cl_es_p998_experimental_profile_config_and_expected_rows():
    _reset_config_loader()
    cfg = config_module.load_config("tier_1_threshold_p998_CL_ES_experiment")
    assert list(cfg.symbols) == ["CL", "ES"]
    assert cfg.execution.prediction_entry_threshold == 0.25
    assert cfg.execution.threshold_mode == "prediction_abs_quantile"
    assert cfg.execution.threshold_quantile == 0.998
    assert len(cfg.symbols) * 30 == 60


def test_experimental_threshold_computed_from_train_predictions_only():
    cfg = RootConfig(
        execution=ExecutionConfig(
            prediction_entry_threshold=0.25,
            threshold_mode="prediction_abs_quantile",
            threshold_quantile=0.995,
        )
    )
    train = np.array([0.0, 0.001, -0.002, 0.003])
    threshold, mode, q, n, train_q = resolve_threshold_from_train(train, cfg, calibration_source="train")
    assert mode == "prediction_abs_quantile"
    assert q == 0.995
    assert n == 4
    assert threshold == pytest.approx(np.quantile(np.abs(train), 0.995))
    assert train_q == threshold


def test_test_predictions_are_not_allowed_for_calibration():
    cfg = RootConfig(execution=ExecutionConfig(threshold_mode="prediction_abs_quantile", threshold_quantile=0.995))
    with pytest.raises(RuntimeError, match="THRESHOLD LEAKAGE FAIL"):
        resolve_threshold_from_train([0.1, 0.2], cfg, calibration_source="test")


def test_threshold_used_artifacts_are_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = RootConfig(execution=ExecutionConfig(threshold_mode="prediction_abs_quantile", threshold_quantile=0.995))
    train = np.array([0.0, 0.001, 0.002, 0.003])
    threshold, _, _, _, train_q = resolve_threshold_from_train(train, cfg)
    test = pl.DataFrame(
        {
            "prediction": [-threshold * 2, 0.0, threshold * 2],
            "position_after": [-1.0, 0.0, 1.0],
            "position_delta": [1.0, 1.0, 1.0],
        }
    )
    row = write_threshold_used(
        symbol="CL",
        split=1,
        config=cfg,
        train_predictions=train,
        test_result=test,
        threshold=threshold,
        train_abs_prediction_quantile=train_q,
    )
    assert row["test_active_bar_pct"] == pytest.approx(2 / 3)
    assert (tmp_path / "reports" / "validation" / "threshold_used.csv").exists()
    assert (tmp_path / "reports" / "validation" / "threshold_used.json").exists()


def test_p995_experiment_activates_expected_synthetic_bars():
    cfg = RootConfig(execution=ExecutionConfig(threshold_mode="prediction_abs_quantile", threshold_quantile=0.995))
    train = np.linspace(-0.001, 0.001, 1000)
    threshold, _, _, _, _ = resolve_threshold_from_train(train, cfg)
    test = pl.DataFrame(
        {
            "prediction": [-threshold * 1.1, -threshold * 0.5, 0.0, threshold * 0.5, threshold * 1.1],
            "position_after": [-1.0, 0.0, 0.0, 0.0, 1.0],
            "position_delta": [1.0, 1.0, 0.0, 0.0, 1.0],
        }
    )
    row = write_threshold_used(
        symbol="ES",
        split=2,
        config=cfg,
        train_predictions=train,
        test_result=test,
        threshold=threshold,
        train_abs_prediction_quantile=threshold,
    )
    assert row["test_long_bars"] == 1
    assert row["test_short_bars"] == 1
    assert row["test_active_bar_pct"] == pytest.approx(0.4)
