import polars as pl
import yaml
from pipeline.common.config import config


BASELINE_FEATURES = [
    'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'ret_15', 'ret_30', 'ret_60',
    'ret_mean_5', 'ret_mean_15', 'ret_mean_30',
    'ret_std_5', 'ret_std_15', 'ret_std_30', 'ret_std_60',
    'ret_zscore_15', 'ret_zscore_30',
    'mom_3', 'mom_5', 'mom_10', 'mom_15', 'mom_30', 'mom_60',
    'close_vs_sma_5', 'close_vs_sma_15', 'close_vs_sma_30', 'close_vs_sma_60',
    'close_vs_ema_5', 'close_vs_ema_15', 'close_vs_ema_30',
    'volatility_5', 'volatility_15', 'volatility_30', 'volatility_60',
    'realized_vol_15', 'realized_vol_30', 'vol_ratio_15_60',
    'bar_range', 'body_size', 'upper_wick', 'lower_wick', 'body_to_range',
    'close_position', 'wick_imbalance', 'range_mean_15', 'range_std_15', 'body_mean_15',
    'dist_to_high_15', 'dist_to_high_30', 'dist_to_high_60',
    'dist_to_low_15', 'dist_to_low_30', 'dist_to_low_60',
    'price_position_15', 'price_position_30', 'price_position_60',
    'breakout_high_15', 'breakout_low_15', 'breakout_high_30', 'breakout_low_30',
    'volume_log', 'volume_change_1', 'volume_ratio_5', 'volume_ratio_15',
    'volume_ratio_30', 'volume_ratio_60',
    'volume_zscore_15', 'volume_zscore_30', 'volume_zscore_60',
    'signed_volume_ratio_5', 'signed_volume_ratio_15', 'signed_volume_ratio_30',
    'volume_weighted_ret', 'volume_imbalance_15', 'volume_imbalance_30',
    'price_volume_corr_30',
    'trend_efficiency_15', 'trend_efficiency_30', 'trend_efficiency_60',
    'positive_ret_ratio_15', 'positive_ret_ratio_30',
    'negative_ret_ratio_15', 'negative_ret_ratio_30',
    'rsi_7', 'rsi_14', 'rsi_30',
    'macd_norm', 'macd_hist_norm', 'bb_position_20', 'bb_width_20', 'atr_14_norm',
]


def load_baseline_feature_names() -> list:
    path = getattr(config, 'BASELINE_FEATURES_FILE', 'configs/baseline_features.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data.get('baseline_features', BASELINE_FEATURES)


def _rolling_z(x: pl.Expr, window: int, min_periods: int | None = None) -> pl.Expr:
    min_periods = min_periods or max(3, window // 3)
    mean = x.shift(1).rolling_mean(window_size=window, min_periods=min_periods)
    std = x.shift(1).rolling_std(window_size=window, min_periods=min_periods)
    return (x - mean) / std.clip(_eps(), None)


def _eps() -> float:
    return float(getattr(config, 'EPS', 1e-9))


def _clip_bounds() -> tuple[float, float]:
    return (
        float(getattr(config, 'CLIP_MIN', -10.0)),
        float(getattr(config, 'CLIP_MAX', 10.0)),
    )


def compute_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    eps = _eps()
    clip_min, clip_max = _clip_bounds()

    close = pl.col('close').cast(pl.Float64)
    high = pl.col('high').cast(pl.Float64)
    low = pl.col('low').cast(pl.Float64)
    open_ = pl.col('open').cast(pl.Float64)
    volume = pl.col('volume').cast(pl.Float64)

    ret_1 = (close / close.shift(1).clip(eps, None)).log()
    bar_range_raw = (high - low).clip(eps, None)
    bar_range_pct = bar_range_raw / close.clip(eps, None)
    body_raw = (close - open_).abs()
    upper_wick_raw = high - pl.max_horizontal(open_, close)
    lower_wick_raw = pl.min_horizontal(open_, close) - low
    signed_volume = pl.when(ret_1 > 0).then(volume).when(ret_1 < 0).then(-volume).otherwise(0.0)

    exprs: list[pl.Expr] = []

    # Returns and momentum. `ret_*` are log returns; `mom_*` are simple returns.
    for lag in (1, 2, 3, 5, 10, 15, 30, 60):
        exprs.append(((close / close.shift(lag).clip(eps, None)).log()).alias(f'ret_{lag}'))
    for w in (5, 15, 30):
        exprs.append(ret_1.rolling_mean(window_size=w, min_periods=max(2, w // 3)).alias(f'ret_mean_{w}'))
    for w in (5, 15, 30, 60):
        exprs.append(ret_1.rolling_std(window_size=w, min_periods=max(2, w // 3)).alias(f'ret_std_{w}'))
    for w in (15, 30):
        mean = ret_1.shift(1).rolling_mean(window_size=w, min_periods=max(3, w // 3))
        std = ret_1.shift(1).rolling_std(window_size=w, min_periods=max(3, w // 3))
        exprs.append(((ret_1 - mean) / std.clip(eps, None)).alias(f'ret_zscore_{w}'))
    for lag in (3, 5, 10, 15, 30, 60):
        exprs.append((close / close.shift(lag).clip(eps, None) - 1.0).alias(f'mom_{lag}'))

    # Moving-average distance.
    for w in (5, 15, 30, 60):
        sma = close.shift(1).rolling_mean(window_size=w, min_periods=max(2, w // 3))
        exprs.append(((close - sma) / close.clip(eps, None)).alias(f'close_vs_sma_{w}'))
    for span in (5, 15, 30):
        ema = close.ewm_mean(span=span, adjust=False)
        exprs.append(((close - ema) / close.clip(eps, None)).alias(f'close_vs_ema_{span}'))

    # Volatility.
    vol_15 = ret_1.rolling_std(window_size=15, min_periods=5)
    vol_60 = ret_1.rolling_std(window_size=60, min_periods=20)
    for w in (5, 15, 30, 60):
        exprs.append(ret_1.rolling_std(window_size=w, min_periods=max(2, w // 3)).alias(f'volatility_{w}'))
    for w in (15, 30):
        exprs.append((ret_1.pow(2).rolling_sum(window_size=w, min_periods=max(5, w // 3)).sqrt()).alias(f'realized_vol_{w}'))
    exprs.append((vol_15 / vol_60.clip(eps, None)).alias('vol_ratio_15_60'))

    # Bar shape.
    exprs.extend([
        bar_range_pct.alias('bar_range'),
        (body_raw / close.clip(eps, None)).alias('body_size'),
        (upper_wick_raw / close.clip(eps, None)).alias('upper_wick'),
        (lower_wick_raw / close.clip(eps, None)).alias('lower_wick'),
        (body_raw / bar_range_raw).alias('body_to_range'),
        ((close - low) / bar_range_raw).alias('close_position'),
        ((upper_wick_raw - lower_wick_raw) / bar_range_raw).alias('wick_imbalance'),
        bar_range_pct.rolling_mean(window_size=15, min_periods=5).alias('range_mean_15'),
        bar_range_pct.rolling_std(window_size=15, min_periods=5).alias('range_std_15'),
        (body_raw / close.clip(eps, None)).rolling_mean(window_size=15, min_periods=5).alias('body_mean_15'),
    ])

    # Rolling range / breakout.
    for w in (15, 30, 60):
        roll_high = high.shift(1).rolling_max(window_size=w, min_periods=max(5, w // 3))
        roll_low = low.shift(1).rolling_min(window_size=w, min_periods=max(5, w // 3))
        roll_range = (roll_high - roll_low).clip(eps, None)
        exprs.extend([
            ((close - roll_high) / close.clip(eps, None)).alias(f'dist_to_high_{w}'),
            ((close - roll_low) / close.clip(eps, None)).alias(f'dist_to_low_{w}'),
            ((close - roll_low) / roll_range).alias(f'price_position_{w}'),
        ])
        if w in (15, 30):
            exprs.extend([
                (close > roll_high).cast(pl.Float64).alias(f'breakout_high_{w}'),
                (close < roll_low).cast(pl.Float64).alias(f'breakout_low_{w}'),
            ])

    # Volume.
    exprs.extend([
        volume.log1p().alias('volume_log'),
        (volume / volume.shift(1).clip(eps, None) - 1.0).alias('volume_change_1'),
    ])
    for w in (5, 15, 30, 60):
        vol_mean = volume.shift(1).rolling_mean(window_size=w, min_periods=max(2, w // 3))
        exprs.append((volume / vol_mean.clip(eps, None)).alias(f'volume_ratio_{w}'))
    for w in (15, 30, 60):
        vol_mean = volume.shift(1).rolling_mean(window_size=w, min_periods=max(5, w // 3))
        vol_std = volume.shift(1).rolling_std(window_size=w, min_periods=max(5, w // 3))
        exprs.append(((volume - vol_mean) / vol_std.clip(eps, None)).alias(f'volume_zscore_{w}'))
    for w in (5, 15, 30):
        vol_sum = volume.rolling_sum(window_size=w, min_periods=max(2, w // 3)).clip(eps, None)
        exprs.append((signed_volume.rolling_sum(window_size=w, min_periods=max(2, w // 3)) / vol_sum).alias(f'signed_volume_ratio_{w}'))
    vol_ratio_15 = volume / volume.shift(1).rolling_mean(window_size=15, min_periods=5).clip(eps, None)
    exprs.append((ret_1 * vol_ratio_15).alias('volume_weighted_ret'))
    for w in (15, 30):
        vol_sum = volume.rolling_sum(window_size=w, min_periods=max(5, w // 3)).clip(eps, None)
        exprs.append((signed_volume.rolling_sum(window_size=w, min_periods=max(5, w // 3)) / vol_sum).alias(f'volume_imbalance_{w}'))

    # Rolling price-volume correlation, computed as covariance / std products.
    x = ret_1
    y = volume.log1p().diff()
    mx = x.rolling_mean(window_size=30, min_periods=10)
    my = y.rolling_mean(window_size=30, min_periods=10)
    mxy = (x * y).rolling_mean(window_size=30, min_periods=10)
    sx = x.rolling_std(window_size=30, min_periods=10)
    sy = y.rolling_std(window_size=30, min_periods=10)
    exprs.append(((mxy - mx * my) / (sx * sy).clip(eps, None)).alias('price_volume_corr_30'))

    # Regime / efficiency.
    abs_ret = ret_1.abs()
    for w in (15, 30, 60):
        directional = (close - close.shift(w)).abs() / close.shift(w).clip(eps, None)
        path = abs_ret.rolling_sum(window_size=w, min_periods=max(5, w // 3)).clip(eps, None)
        exprs.append((directional / path).alias(f'trend_efficiency_{w}'))
    pos = pl.when(ret_1 > 0).then(1.0).otherwise(0.0)
    neg = pl.when(ret_1 < 0).then(1.0).otherwise(0.0)
    for w in (15, 30):
        exprs.extend([
            pos.rolling_mean(window_size=w, min_periods=max(5, w // 3)).alias(f'positive_ret_ratio_{w}'),
            neg.rolling_mean(window_size=w, min_periods=max(5, w // 3)).alias(f'negative_ret_ratio_{w}'),
        ])

    # RSI / MACD / Bollinger / ATR.
    delta = close.diff()
    gain_base = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss_base = pl.when(delta < 0).then(-delta).otherwise(0.0)
    for w in (7, 14, 30):
        gain = gain_base.rolling_mean(window_size=w, min_periods=max(3, w // 3))
        loss = loss_base.rolling_mean(window_size=w, min_periods=max(3, w // 3))
        rs = gain / loss.clip(eps, None)
        exprs.append(((100.0 - (100.0 / (1.0 + rs))) / 100.0).alias(f'rsi_{w}'))

    ema12 = close.ewm_mean(span=12, adjust=False)
    ema26 = close.ewm_mean(span=26, adjust=False)
    macd = ema12 - ema26
    macd_signal = macd.ewm_mean(span=9, adjust=False)
    exprs.extend([
        (macd / close.clip(eps, None)).alias('macd_norm'),
        ((macd - macd_signal) / close.clip(eps, None)).alias('macd_hist_norm'),
    ])

    sma20 = close.rolling_mean(window_size=20, min_periods=10)
    std20 = close.rolling_std(window_size=20, min_periods=10)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    exprs.extend([
        ((close - lower) / (upper - lower).clip(eps, None)).alias('bb_position_20'),
        ((upper - lower) / close.clip(eps, None)).alias('bb_width_20'),
    ])

    prev_close = close.shift(1)
    true_range = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())
    exprs.append((true_range.rolling_mean(window_size=14, min_periods=5) / close.clip(eps, None)).alias('atr_14_norm'))

    df = df.with_columns(exprs)
    cols = [c for c in BASELINE_FEATURES if c in df.columns]
    df = df.with_columns([
        pl.col(c).fill_nan(0.0).fill_null(0.0).clip(clip_min, clip_max).cast(pl.Float32)
        for c in cols
    ])
    return df
