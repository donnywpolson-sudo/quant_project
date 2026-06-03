"""
align.py
Align higher-timeframe (1h, daily) OHLCV streams to 5-min bars.
Uses join_asof(strategy='backward') which matches each 5-min row to the
most recent HTF bar whose timestamp is <= the 5-min bar's timestamp.

CRITICAL: Daily bar timestamps are shifted forward by 1 day so that
each 5-min bar on day D matches the PREVIOUS completed day's daily bar
(at midnight D-1 shifted to midnight D), NEVER the current incomplete day.
"""

import polars as pl

_OHLCV_COLS = {'open', 'high', 'low', 'close', 'volume', 'session_id'}


def align_htf_streams(
    df_5min: pl.DataFrame,
    df_1h: pl.DataFrame,
    df_daily: pl.DataFrame
) -> pl.DataFrame:
    df_5min = df_5min.sort('ts_event')

    # 1h bars: SHIFT TIMESTAMP FORWARD by 1 hour so backward join yields
    # the PREVIOUS completed hourly bar, not the current incomplete one.
    # Without this shift, a 5-min bar at 10:05 matches the 1h bar at 10:00
    # which contains the full hour's close (at 10:55) — 50 min of future data.
    if df_1h is not None and (not df_1h.is_empty()):
        df_1h = df_1h.sort('ts_event')
        ts_dtype = df_5min['ts_event'].dtype
        df_1h = df_1h.with_columns(
            (pl.col('ts_event') + pl.duration(hours=1)).cast(ts_dtype).alias('ts_event')
        )
        renames_1h = {c: '1h_' + c for c in df_1h.columns if c in _OHLCV_COLS}
        if renames_1h:
            df_1h = df_1h.rename(renames_1h)
        df_5min = df_5min.join_asof(df_1h, on='ts_event', strategy='backward')

    # Daily bars: SHIFT TIMESTAMP FORWARD by 1 day so that a 5-min bar
    # at e.g. 10:30 AM on day D matches the daily bar at midnight (D-1)+1d = midnight D
    # which represents the COMPLETED prior day's data.
    # Forward-fill after the join handles the boundary: the first few 5-min bars
    # of a session may fall before the shifted daily timestamp; they legitimately
    # carry forward the most recent completed daily bar (from the prior session).
    if df_daily is not None and (not df_daily.is_empty()):
        df_daily = df_daily.sort('ts_event')
        # Shift daily timestamps forward by 1 day so backward join yields prior day
        # Cast to match the 5-min timestamp dtype (preserves time unit)
        ts_dtype = df_5min['ts_event'].dtype
        df_daily = df_daily.with_columns(
            (pl.col('ts_event') + pl.duration(days=1)).cast(ts_dtype).alias('ts_event')
        )
        renames_daily = {c: 'daily_' + c for c in df_daily.columns if c in _OHLCV_COLS}
        if renames_daily:
            df_daily = df_daily.rename(renames_daily)
        df_5min = df_5min.join_asof(df_daily, on='ts_event', strategy='backward')
        # Forward-fill only. Backward-fill within a session would leak a later
        # daily value into earlier rows before that value is causally available.
        daily_cols = [c for c in df_5min.columns if c.startswith('daily_')]
        if daily_cols:
            df_5min = df_5min.with_columns(
                [pl.col(c).forward_fill().over('session_id') for c in daily_cols]
            )

    return df_5min
