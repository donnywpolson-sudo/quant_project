import yaml
import logging
import math
from pathlib import Path
from pipeline.common.config import config
logger = logging.getLogger(__name__)

def _read_market_config(symbol: str):
    yaml_path = getattr(config, "MARKET_CONFIGS", {}).get(symbol) or 'configs/market_specs.yaml'
    if not yaml_path or not Path(yaml_path).exists():
        return None, yaml_path

    with open(yaml_path, 'r') as f:
        raw_cfg = yaml.safe_load(f) or {}

    if 'markets' in raw_cfg:
        return (raw_cfg.get('markets') or {}).get(symbol), yaml_path

    return raw_cfg, yaml_path


def _known_symbols() -> set[str]:
    symbols = set(getattr(config, "MARKET_CONFIGS", {}).keys())
    specs_path = Path('configs/market_specs.yaml')
    if specs_path.exists():
        with open(specs_path, 'r') as f:
            specs = yaml.safe_load(f) or {}
        symbols.update((specs.get('markets') or {}).keys())
    return symbols


def detect_symbol_from_path(data_path: str) -> str:
    path = Path(data_path)
    known_symbols = _known_symbols()
    for part in path.parent.parts:
        if part in known_symbols:
            return part
    import glob as _glob
    for f in _glob.glob(data_path):
        fp = Path(f)
        for part in fp.parent.parts:
            if part in known_symbols:
                return part
        for known in known_symbols:
            if fp.stem == known or fp.stem.startswith(known + '_') or fp.stem.startswith(known + '.'):
                return known
    raise RuntimeError(
        f'SYMBOL FAIL: cannot detect symbol from path {data_path}. '
        f'No known market ({sorted(known_symbols)}) '
        f'found in path parts {list(path.parent.parts)} or any matched file. '
        f'Ensure data directory structure includes the symbol name '
        f'(e.g. data/ES/2024.parquet).'
    )

def load_market_config(symbol: str):
    market_cfg, yaml_path = _read_market_config(symbol)
    if not market_cfg:
        logger.warning(f'Market config for {symbol} not found at {yaml_path}, using global defaults.')
        return
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
    market_cfg, yaml_path = _read_market_config(symbol)
    if not market_cfg:
        raise RuntimeError(
            f'CONTRACT FAIL: no market config found for symbol={symbol}. '
            'Cannot resolve contract multiplier.'
        )
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


def get_tick_value(symbol: str) -> float:
    market_cfg, yaml_path = _read_market_config(symbol)
    if not market_cfg:
        raise RuntimeError(f'CONTRACT FAIL: no market config found for symbol={symbol}. Cannot resolve tick_value.')
    specs = market_cfg.get('contract_specs') or {}
    if 'tick_value' not in specs:
        raise RuntimeError(f'CONTRACT FAIL: tick_value missing for symbol={symbol} in {yaml_path}.')
    value = float(specs['tick_value'])
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f'CONTRACT FAIL: invalid tick_value for symbol={symbol}: {value}.')
    return value


def get_tick_size(symbol: str) -> float:
    market_cfg, yaml_path = _read_market_config(symbol)
    if not market_cfg:
        raise RuntimeError(f'CONTRACT FAIL: no market config found for symbol={symbol}. Cannot resolve tick_size.')
    specs = market_cfg.get('contract_specs') or {}
    if 'tick_size' not in specs:
        raise RuntimeError(f'CONTRACT FAIL: tick_size missing for symbol={symbol} in {yaml_path}.')
    value = float(specs['tick_size'])
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f'CONTRACT FAIL: invalid tick_size for symbol={symbol}: {value}.')
    return value
