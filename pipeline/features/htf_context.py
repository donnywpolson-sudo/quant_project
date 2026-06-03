"""
htf_context.py
Compute higher-timeframe context features using strictly causal (t-1) logic.
All expanding intraday boundaries use shift(1).cum_max/min over date groups.
No look-ahead into current incomplete daily bars.
"""
import polars as pl
from pipeline.common.config import config


def _session_prior_close_table(df: pl.DataFrame) -> pl.DataFrame:
    """One row per session with prior completed-session closes.

    This must be computed at session granularity. Row-level shifts inside a
    session leak the current session's final close into earlier rows.
    """
    return (
        df.select(['session_id', 'ts_event', 'close'])
        .sort('ts_event')
        .group_by('session_id', maintain_order=True)
        .agg([
            pl.col('ts_event').first().alias('_session_start'),
            pl.col('close').last().alias('_session_close'),
        ])
        .sort('_session_start')
        .with_columns([
            pl.col('_session_close').shift(1).alias('_prev_day_close'),
            pl.col('_session_close').shift(2).alias('_two_days_ago_close'),
        ])
        .select(['session_id', '_prev_day_close', '_two_days_ago_close'])
    )


def add_htf_context_features(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.height == 0:
        return df

    eps = config.EPS
    session_col = pl.col('session_id')

    # --- Expanding intraday high/low (strictly causal: shift(1) excludes current bar) ---
    df = df.with_columns([
        pl.col('high').shift(1).cum_max().over(session_col).alias('_daily_high_expanding'),
        pl.col('low').shift(1).cum_min().over(session_col).alias('_daily_low_expanding'),
    ])

    # Distance to expanding daily high/low (causal: uses only bars before current)
    df = df.with_columns([
        ((pl.col('_daily_high_expanding') - pl.col('close'))
         / pl.col('_daily_high_expanding').clip(eps, None)).alias('htf_distance_to_daily_high'),
        ((pl.col('close') - pl.col('_daily_low_expanding'))
         / pl.col('_daily_low_expanding').clip(eps, None)).alias('htf_distance_to_daily_low'),
    ])

    # Previous two completed sessions' closes.
    # Compute at session granularity, then join back. Do not use row-level
    # shift inside a session; that leaks the current session's final close.
    df = df.join(_session_prior_close_table(df), on='session_id', how='left')

    # Daily return: previous FULL day's realized return (constant per day)
    # log(yesterday's close / day-before-yesterday's close)
    # This is genuinely past-only — it cannot be influenced by any bar in the current day
    df = df.with_columns(
        ((pl.col('_prev_day_close') / pl.col('_two_days_ago_close').clip(eps, None)).log()
         ).alias('htf_daily_return_1')
    )

    # Daily trend slope: prior bar vs ~10 days ago (260 bars/day * 10), lagged
    bars_per_day_approx = 260
    df = df.with_columns(
        ((pl.col('close').shift(1) - pl.col('close').shift(1 + bars_per_day_approx * 10))
         / (bars_per_day_approx * 10 * pl.col('close').shift(1 + bars_per_day_approx * 10).clip(eps, None))
         ).alias('htf_daily_trend_slope_10')
    )

    # Daily volatility: rolling std of 1-bar returns, lagged by 1 bar
    ret_1 = (pl.col('close') / pl.col('close').shift(1)).log()
    df = df.with_columns(
        ret_1.shift(1).rolling_std(window_size=260).clip(eps, None).alias('htf_daily_vol_5')
    )

    # Hourly features from 1h aligned columns (via join_asof backward, timestamp at bar start)
    if '1h_close' in df.columns:
        df = df.with_columns(
            (pl.col('1h_close') / pl.col('1h_close').shift(1)).log().alias('_1h_return')
        )
        df = df.with_columns(
            pl.col('_1h_return').shift(1).rolling_std(window_size=4).alias('_1h_vol_4')
        )

    # Trend alignment: 1h return sign x daily trend sign (both lagged)
    if '_1h_return' in df.columns:
        df = df.with_columns(
            (pl.col('_1h_return').shift(1) * pl.col('htf_daily_trend_slope_10').sign()
             ).alias('htf_hourly_trend_alignment')
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias('htf_hourly_trend_alignment'))

    # Volatility ratio: 1h vol / daily vol (both lagged)
    if '_1h_vol_4' in df.columns:
        df = df.with_columns(
            (pl.col('_1h_vol_4') / pl.col('htf_daily_vol_5').clip(eps, None)
             ).alias('htf_volatility_ratio')
        )
    else:
        df = df.with_columns(pl.lit(1.0).alias('htf_volatility_ratio'))

    # Clip and cast all HTF columns
    htf_cols = [c for c in df.columns if c.startswith('htf_')]
    for col in htf_cols:
        df = df.with_columns(
            pl.col(col).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32)
        )

    # Drop temporary columns
    temp_cols = ['_daily_high_expanding', '_daily_low_expanding', '_prev_day_close',
                 '_two_days_ago_close', '_1h_return', '_1h_vol_4']
    df = df.drop([c for c in temp_cols if c in df.columns])
    return df
