import yaml
import logging
import math
from pathlib import Path
from pipeline.common.config import config
logger = logging.getLogger(__name__)

def detect_symbol_from_path(data_path: str) -> str:
    path = Path(data_path)
    for part in path.parent.parts:
        if part in config.MARKET_CONFIGS:
            return part
    import glob as _glob
    for f in _glob.glob(data_path):
        fp = Path(f)
        for part in fp.parent.parts:
            if part in config.MARKET_CONFIGS:
                return part
        for known in config.MARKET_CONFIGS:
            if fp.stem == known or fp.stem.startswith(known + '_') or fp.stem.startswith(known + '.'):
                return known
    raise RuntimeError(
        f'SYMBOL FAIL: cannot detect symbol from path {data_path}. '
        f'No known market ({sorted(config.MARKET_CONFIGS.keys())}) '
        f'found in path parts {list(path.parent.parts)} or any matched file. '
        f'Ensure data directory structure includes the symbol name '
        f'(e.g. data/ES/2024.parquet).'
    )

def load_market_config(symbol: str):
    yaml_path = config.MARKET_CONFIGS.get(symbol)
    if not yaml_path or not Path(yaml_path).exists():
        logger.warning(f'Market config for {symbol} not found at {yaml_path}, using global defaults.')
        return
    with open(yaml_path, 'r') as f:
        market_cfg = yaml.safe_load(f) or {}
    risk_cfg = market_cfg.get('risk') or {}

    def _cfg_value(key: str):
        return market_cfg.get(key, risk_cfg.get(key))

    # Market YAML risk fields define instrument limits.  Execution-cost
    # assumptions stay profile-controlled so alpha_0 remains a zero-cost
    # signal-quality baseline; use an explicit cost profile for net testing.
    overrides = {'ROLL_WINDOWS': market_cfg.get('roll_windows'), 'ROLL_WINDOWS_1H': market_cfg.get('roll_windows_1h'), 'ROLL_WINDOWS_DAILY': market_cfg.get('roll_windows_daily'), 'REGIME_HIGH_THRESH': market_cfg.get('regime_high_thresh'), 'REGIME_LOW_THRESH': market_cfg.get('regime_low_thresh'), 'HTF_TREND_WINDOWS': market_cfg.get('htf_trend_windows'), 'HTF_VOLATILITY_WINDOWS': market_cfg.get('htf_volatility_windows'), 'MAX_LEVERAGE': _cfg_value('max_leverage'), 'MAX_POSITION_SIZE': _cfg_value('max_position_size')}
    for attr, value in overrides.items():
        if value is not None:
            setattr(config, attr, value)
            logger.info(f'Overrode {attr} = {value} for {symbol}')


def get_contract_multiplier(symbol: str) -> float:
    if not symbol:
        raise RuntimeError(
            'CONTRACT FAIL: symbol is required. Cannot resolve contract multiplier.'
        )
    yaml_path = config.MARKET_CONFIGS.get(symbol)
    if not yaml_path or not Path(yaml_path).exists():
        raise RuntimeError(
            f'CONTRACT FAIL: no market config found for symbol={symbol}. '
            'Cannot resolve contract multiplier.'
        )
    with open(yaml_path, 'r') as f:
        market_cfg = yaml.safe_load(f) or {}
    metadata = market_cfg.get('metadata') or {}
    if 'contract_multiplier' not in metadata:
        raise RuntimeError(
            f'CONTRACT FAIL: contract_multiplier missing for symbol={symbol}.'
        )
    try:
        multiplier = float(metadata['contract_multiplier'])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'CONTRACT FAIL: invalid contract_multiplier for symbol={symbol}: '
            f'{metadata["contract_multiplier"]!r}.'
        ) from exc
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise RuntimeError(
            f'CONTRACT FAIL: invalid contract_multiplier for symbol={symbol}: '
            f'{multiplier}.'
        )
    return multiplier
