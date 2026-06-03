"""
volume_profile.py
Deterministic, vectorized Volume Profile and VPA (Volume Price Analysis) features.

Volume Profile (Horizontal):
  - Rolling Point of Control (POC): price level with the highest volume in a lookback window.
  - Value Area High (VAH) / Value Area Low (VAL): 70% volume distribution boundaries.
  - Distance from current price to rolling POC.
  - Inside/Outside Value Area binary flag.

VPA (Vertical):
  - Volume-to-Spread efficiency ratio: |close - open| / (high - low) * volume_ratio
    Measures effort (volume) vs. result (spread) — anomalies signal absorption/exhaustion.
  - Volume spike detection: current volume relative to rolling median.
  - Spread compression/expansion flags.

ALL features use strictly lagged (t-1) data for rolling windows.
No future leakage. Fully vectorized with polars. No explicit Python loops over bars.
"""

import polars as pl
import numpy as np
from pipeline.common.config import config

# ---------------------------------------------------------------------------
# Volume Profile: horizontal volume binning across a lookback window
# ---------------------------------------------------------------------------

def _build_volume_profile(
    df: pl.DataFrame,
    window: int = 260,       # ~1 day of 5m bars (22h * 12 bars/h = 264)
    n_bins: int = 50,
    poc_roll_window: int = 5,
) -> pl.DataFrame:
    """
    Compute a rolling Volume Profile using price bins over a lookback window.

    Strategy (strictly causal, no future leakage):
      1. For each bar at index t, the lookback window is [t-window, t-1].
      2. Price bins are defined as uniformly spaced between the window's low and high.
      3. Volume is distributed proportionally across bins that each bar's [low, high]
         range spans, weighted by how much of the bar's range falls in each bin.
      4. POC = bin with the maximum allocated volume.
      5. VAH/VAL = boundaries containing 70% of total volume, centered on POC.

    To keep this vectorized and memory-safe, we do NOT create a full (N, n_bins)
    matrix. Instead, we approximate using a rolling quantile approach on close
    prices, weighted by volume.

    The approximation:
      - Compute volume-weighted rolling statistics of the close price.
      - Estimate POC as rolling median of close weighted by local volume.
      - Estimate VAH/VAL using volume-weighted quantiles.
      - This is a lossy approximation of true Volume Profile but is fully
        vectorized and O(n * log(window)) via polars rolling operations.

    Returns columns added in-place (side effect via df.with_columns).
    """
    eps = config.EPS

    # --- Strictly lag volume and close by 1 bar ---
    close_lag = pl.col('close').shift(1)
    vol_lag = pl.col('volume').shift(1)

    # --- Rolling total volume in lookback window (for normalization) ---
    total_vol = vol_lag.rolling_sum(window_size=window, min_periods=min(10, window // 2))

    # --- Volume-weighted rolling mean of close (proxy for center of mass) ---
    close_x_vol = (close_lag * vol_lag).rolling_sum(window_size=window, min_periods=min(10, window // 2))
    vwap_close = close_x_vol / total_vol.clip(eps, None)

    # --- Rolling volume-weighted standard deviation of close ---
    # Var = E[X^2] - E[X]^2  for volume-weighted
    close_sq_x_vol = (close_lag * close_lag * vol_lag).rolling_sum(window_size=window, min_periods=min(10, window // 2))
    vol_weighted_var = (close_sq_x_vol / total_vol.clip(eps, None)) - (vwap_close * vwap_close)
    vol_weighted_std = vol_weighted_var.sqrt().clip(eps, None)

    # --- Approximate POC: rolling median of close in the lookback ---
    # Polars doesn't have rolling_median for weighted data directly,
    # but rolling_median on close is a reasonable proxy for POC when
    # volume distribution is roughly symmetric around the median.
    # Better: use the volume-weighted mean (VWAP) as a proxy for POC
    # since it's the center of volume mass.
    rolling_poc = vwap_close

    # --- Value Area boundaries (68-70% of volume) ---
    # Approximate with rolling quantiles of close, volume-aware.
    # We compute rolling 15th and 85th percentiles of close as VAH/VAL proxy.
    # A true Volume Profile uses volume-weighted distributions but this
    # approximation captures the same intuition (price extremes that
    # contain the bulk of activity).
    rolling_low_15 = close_lag.rolling_quantile(quantile=0.15, window_size=window, min_periods=min(10, window // 2))
    rolling_high_85 = close_lag.rolling_quantile(quantile=0.85, window_size=window, min_periods=min(10, window // 2))

    # --- Distance from current close to rolling POC (as % of price) ---
    poc_distance = (pl.col('close') - rolling_poc) / rolling_poc.clip(eps, None)

    # --- Inside/Outside Value Area binary flag ---
    inside_value_area = (
        (pl.col('close') >= rolling_low_15) &
        (pl.col('close') <= rolling_high_85)
    ).cast(pl.Float32)

    # --- POC stability: rolling std of POC over a shorter window ---
    poc_stability = rolling_poc.rolling_std(window_size=poc_roll_window, min_periods=2)

    # --- Volume concentration: fraction of total volume in the value area ---
    # Approximation: sum of volume for bars where close was in [VAL, VAH] in the window
    # We compute this by checking each bar's close against the rolling boundaries
    close_in_va_window = (
        (close_lag >= rolling_low_15) &
        (close_lag <= rolling_high_85)
    ).cast(pl.Float32)
    vol_in_va = (close_in_va_window * vol_lag).rolling_sum(window_size=window, min_periods=min(10, window // 2))
    volume_concentration = vol_in_va / total_vol.clip(eps, None)

    exprs = [
        rolling_poc.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_volume_poc'),
        rolling_low_15.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_volume_val'),
        rolling_high_85.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_volume_vah'),
        poc_distance.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_distance_to_poc'),
        inside_value_area.alias('feature_inside_value_area'),
        poc_stability.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_poc_stability'),
        volume_concentration.clip(0.0, 1.0).cast(pl.Float32).alias('feature_volume_conc_in_va'),
    ]

    df = df.with_columns(exprs)
    return df


# ---------------------------------------------------------------------------
# VPA: Vertical Volume Analysis (Effort vs. Result)
# ---------------------------------------------------------------------------

def _add_volume_to_spread_efficiency(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """
    Volume-to-Spread Efficiency Ratio (VPA metric).

    Measures: how much spread (high-low range) is produced per unit of volume.
    Anomalies occur when:
      - High volume + small spread = absorption (smart money accumulating)
      - Low volume + large spread = weak move, likely to reverse

    Formula per bar:
      spread = (high - low) / close
      vol_ratio = volume / rolling_median_volume
      efficiency = spread / vol_ratio   (higher = more efficient price movement)

    All rolling stats strictly lagged (shift(1)).

    Also adds:
      - Volume spike flag: volume > 2x rolling median
      - Spread compression flag: spread < 0.5x rolling median spread
    """
    eps = config.EPS

    close = pl.col('close')
    high = pl.col('high')
    low = pl.col('low')
    open_ = pl.col('open')
    volume = pl.col('volume')

    # --- Per-bar spread (normalized by close) ---
    spread = (high - low) / close.clip(eps, None)

    # --- Lagged volume and spread for rolling median ---
    vol_lag = volume.shift(1)
    spread_lag = spread.shift(1)

    # --- Rolling median volume (strictly t-1) ---
    med_vol = vol_lag.rolling_median(window_size=window, min_periods=5)

    # --- Rolling median spread (strictly t-1) ---
    med_spread = spread_lag.rolling_median(window_size=window, min_periods=5)

    # --- Current volume ratio vs. rolling median ---
    vol_ratio = volume / med_vol.clip(eps, None)

    # --- Current spread ratio vs. rolling median ---
    spread_ratio = spread / med_spread.clip(eps, None)

    # --- Volume-to-Spread Efficiency: spread per unit of relative volume ---
    # High values = price moves efficiently on normal/low volume
    # Low values = high volume but narrow range (absorption/compression)
    efficiency = spread_ratio / vol_ratio.clip(eps, None)

    # --- Volume spike: current volume > 2x rolling median ---
    vol_spike = (volume > 2.0 * med_vol).cast(pl.Float32)

    # --- Volume drought: current volume < 0.5x rolling median ---
    vol_drought = (volume < 0.5 * med_vol).cast(pl.Float32)

    # --- Spread compression: current spread < 0.5x rolling median ---
    spread_compression = (spread < 0.5 * med_spread).cast(pl.Float32)

    # --- Spread expansion: current spread > 2x rolling median ---
    spread_expansion = (spread > 2.0 * med_spread).cast(pl.Float32)

    # --- Volume Climax: volume spike + spread expansion (exhaustion signal) ---
    vol_climax = (vol_spike * spread_expansion).cast(pl.Float32)

    # --- Absorption: volume spike + spread compression (accumulation signal) ---
    absorption = (vol_spike * spread_compression).cast(pl.Float32)

    # --- Body-to-range ratio (candlestick shape) ---
    body_ratio = (close - open_).abs() / (high - low + eps).clip(eps, None)

    # --- Effort-vs-Result composite: body_ratio * volume_ratio ---
    # Large body + large volume = conviction
    # Small body + large volume = indecision/absorption
    effort_result = body_ratio * vol_ratio

    exprs = [
        efficiency.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_vol_to_spread_eff'),
        vol_ratio.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_vol_ratio_vs_med'),
        spread_ratio.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_spread_ratio_vs_med'),
        vol_spike.alias('feature_vol_spike'),
        vol_drought.alias('feature_vol_drought'),
        spread_compression.alias('feature_spread_compression'),
        spread_expansion.alias('feature_spread_expansion'),
        vol_climax.alias('feature_vol_climax'),
        absorption.alias('feature_absorption'),
        body_ratio.clip(0.0, 1.0).cast(pl.Float32).alias('feature_body_ratio'),
        effort_result.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_effort_vs_result'),
    ]

    df = df.with_columns(exprs)
    return df


# ---------------------------------------------------------------------------
# Multi-timeframe Volume Profile (1h and daily windows)
# ---------------------------------------------------------------------------

def _add_short_term_volume_profile(df: pl.DataFrame, window: int = 48) -> pl.DataFrame:
    """
    Short-term volume profile (~4 hours of 5m bars = 48 bars).
    Captures intra-session volume structures.
    """
    eps = config.EPS
    close_lag = pl.col('close').shift(1)
    vol_lag = pl.col('volume').shift(1)

    total_vol = vol_lag.rolling_sum(window_size=window, min_periods=5)
    close_x_vol = (close_lag * vol_lag).rolling_sum(window_size=window, min_periods=5)
    st_poc = close_x_vol / total_vol.clip(eps, None)

    st_val = close_lag.rolling_quantile(quantile=0.15, window_size=window, min_periods=5)
    st_vah = close_lag.rolling_quantile(quantile=0.85, window_size=window, min_periods=5)

    st_poc_dist = (pl.col('close') - st_poc) / st_poc.clip(eps, None)
    st_inside_va = (
        (pl.col('close') >= st_val) & (pl.col('close') <= st_vah)
    ).cast(pl.Float32)

    exprs = [
        st_poc.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_volume_poc_4h'),
        st_val.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_volume_val_4h'),
        st_vah.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_volume_vah_4h'),
        st_poc_dist.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_distance_to_poc_4h'),
        st_inside_va.alias('feature_inside_va_4h'),
    ]
    df = df.with_columns(exprs)
    return df


# ---------------------------------------------------------------------------
# Composite: add all volume profile and VPA features
# ---------------------------------------------------------------------------

def add_volume_profile_features(
    df: pl.DataFrame,
    daily_window: int = 260,
    short_window: int = 48,
    vpa_window: int = 20,
    n_bins: int = 50,
    poc_roll: int = 5,
) -> pl.DataFrame:
    """
    Add all Volume Profile and VPA features to the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: 'close', 'high', 'low', 'open', 'volume'.
    daily_window : int
        Lookback bars for daily volume profile (~260 = 1 session day).
    short_window : int
        Lookback bars for short-term volume profile (~48 = 4 hours).
    vpa_window : int
        Lookback for rolling medians in VPA calculations.
    n_bins : int
        Number of price bins for volume profile (used conceptually, not directly).
    poc_roll : int
        Window for POC stability calculation.

    Returns
    -------
    pl.DataFrame with added feature columns.
    """
    if df is None or df.height == 0:
        return df

    # Daily volume profile
    df = _build_volume_profile(
        df, window=daily_window, n_bins=n_bins, poc_roll_window=poc_roll
    )

    # Short-term (4h) volume profile
    df = _add_short_term_volume_profile(df, window=short_window)

    # VPA: vertical volume analysis
    df = _add_volume_to_spread_efficiency(df, window=vpa_window)

    # Clean all new feature columns
    new_feature_cols = [
        c for c in df.columns
        if c.startswith('feature_volume_') or
           c.startswith('feature_distance_to_poc') or
           c.startswith('feature_inside_') or
           c.startswith('feature_poc_') or
           c.startswith('feature_vol_') or
           c.startswith('feature_spread_') or
           c.startswith('feature_body_') or
           c.startswith('feature_effort_') or
           c.startswith('feature_absorption')
    ]

    for col in new_feature_cols:
        df = df.with_columns(
            pl.col(col)
            .fill_nan(0.0)
            .fill_null(0.0)
            .clip(config.CLIP_MIN, config.CLIP_MAX)
            .cast(pl.Float32)
        )

    return df


# ---------------------------------------------------------------------------
# Feature name registry (for manifest / YAML)
# ---------------------------------------------------------------------------

VOLUME_PROFILE_FEATURE_NAMES = [
    # Daily volume profile
    'feature_volume_poc',
    'feature_volume_val',
    'feature_volume_vah',
    'feature_distance_to_poc',
    'feature_inside_value_area',
    'feature_poc_stability',
    'feature_volume_conc_in_va',
    # 4h volume profile
    'feature_volume_poc_4h',
    'feature_volume_val_4h',
    'feature_volume_vah_4h',
    'feature_distance_to_poc_4h',
    'feature_inside_va_4h',
    # VPA: volume-to-spread efficiency
    'feature_vol_to_spread_eff',
    'feature_vol_ratio_vs_med',
    'feature_spread_ratio_vs_med',
    'feature_vol_spike',
    'feature_vol_drought',
    'feature_spread_compression',
    'feature_spread_expansion',
    'feature_vol_climax',
    'feature_absorption',
    'feature_body_ratio',
    'feature_effort_vs_result',
]
