from pipeline.common.config import config
from pipeline.common.market import load_market_config


def test_load_market_config_applies_nested_risk_overrides(monkeypatch):
    monkeypatch.setattr(config, "MARKET_CONFIGS", {"ES": "configs/markets/ES.yaml"}, raising=False)
    monkeypatch.setattr(config, "MAX_LEVERAGE", 1.0, raising=False)
    monkeypatch.setattr(config, "SLIPPAGE_K", 0.0, raising=False)
    monkeypatch.setattr(config, "VOL_PENALTY", 0.0, raising=False)
    monkeypatch.setattr(config, "MAX_POSITION_SIZE", float("inf"), raising=False)

    load_market_config("ES")

    assert config.MAX_LEVERAGE == 3.0
    assert config.MAX_POSITION_SIZE == 50
    assert config.SLIPPAGE_K == 0.0
    assert config.VOL_PENALTY == 0.0
