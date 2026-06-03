"""
pipeline/02_normalization/gap_filter.py — Corrected gap detection (Patch 1).

CRITICAL BUG FIX: The original quant/gap_filter.py used divisor 60_000_000_000.0
which converts nanoseconds to minutes (60e9 ns = 1 min).  The ts_event column
in this project uses datetime[us] (microseconds).  Diff → cast(Int64) yields
microseconds, so the correct divisor is 60_000_000.0 (60e6 µs = 1 min).

The old divisor under-reported every gap by a factor of 1000, effectively
disabling intraday gap filtering (threshold of 30 minutes became 30,000 minutes
≈ 20.8 days).
"""

import polars as pl


def filter_gaps(
    df: pl.DataFrame, max_gap_minutes: float = 30
) -> pl.DataFrame:
    """
    Remove bars where the time gap between consecutive ts_event values
    exceeds *max_gap_minutes*.

    Detects the time unit of the ts_event column and uses the correct
    divisor (60e9 for ns, 60e6 for us) so the gap filter behaves
    correctly regardless of the upstream Datetime resolution.
    """
    df = df.sort("ts_event")
    time_unit = df["ts_event"].dtype.time_unit
    us_per_minute = 60_000_000
    ns_per_minute = 60_000_000_000
    divisor = float(ns_per_minute) if time_unit == 'ns' else float(us_per_minute)
    gap = df["ts_event"].diff().cast(pl.Int64) / divisor
    df = df.with_columns(gap.alias("_gap_minutes"))
    df = df.filter(
        pl.col("_gap_minutes").is_null()
        | (pl.col("_gap_minutes") <= max_gap_minutes)
    )
    return df.drop("_gap_minutes")
