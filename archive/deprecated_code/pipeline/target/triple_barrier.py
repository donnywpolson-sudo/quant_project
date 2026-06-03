import polars as pl
import numpy as np


def add_triple_barrier_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Volatility-adjusted triple-barrier label.

    For each bar t, constructs a 64-bar (~5.3h) forward window and tests
    which of three barriers is touched first:

      Upper:  entry * exp(+MULT_UPPER * vol_4h)   (MULT_UPPER=1.0, ~1σ)
      Lower:  entry * exp(-MULT_LOWER * vol_4h)   (MULT_LOWER=1.0, ~1σ)
      Time:   64 bars (expiry)

    vol_4h = daily_vol * sqrt(64 / 276)  (daily vol scaled to window)

    Labels:
      +1   — upper barrier touched first (bullish)
      -1   — lower barrier touched first (bearish)
       0   — time barrier expired (neutral / sideways)
      NaN  — no forward window available (last 64 bars of dataset)

    The daily volatility is read from the ``htf_daily_vol_5`` column which
    is computed from lagged 1-bar returns (rolling std, window=260).
    Falls back to a 20-bar rolling std when the column is unavailable.
    """
    H_BARS = 64
    BARS_PER_DAY = 276

    close = df['close'].to_numpy().astype(np.float64)
    high = df['high'].to_numpy().astype(np.float64)
    low = df['low'].to_numpy().astype(np.float64)
    n = len(close)

    # Daily volatility — use aligned column when available.
    # htf_daily_vol_5 is the 1-bar (5-min) log return std, computed from
    # lagged returns with a 260-bar rolling window (decimal form, ~0.0005).
    # Scale to daily vol: daily_vol = bar_vol * sqrt(BARS_PER_DAY).
    # Then scale to window:  vol_4h = daily_vol * sqrt(H_BARS / BARS_PER_DAY).
    # Equivalent to vol_4h = bar_vol * sqrt(H_BARS).
    if 'htf_daily_vol_5' in df.columns:
        bar_vol = df['htf_daily_vol_5'].to_numpy().astype(np.float64)
        bar_vol = np.nan_to_num(bar_vol, nan=0.0005)
    else:
        # Vectorized fallback: use Polars rolling_std on log returns.
        # Avoids O(n²) Python for-loop over np.diff slices.
        log_ret = (
            pl.col('close')
            .log()
            .diff()
            .cast(pl.Float64)
        )
        bar_vol = (
            log_ret
            .rolling_std(window_size=260, min_samples=20)
            .fill_null(0.0005)
        )
        bar_vol = df.select(bar_vol).to_series().to_numpy(allow_copy=False).astype(np.float64)

    bar_vol = np.maximum(bar_vol, 0.0001)
    vol_4h = bar_vol * np.sqrt(H_BARS)

    # Volatility-adjusted barriers per market.
    # Barrier width = multiplier * vol_4h where vol_4h = bar_vol * sqrt(64).
    # Upper:  exp(+MULT_UPPER * vol_4h)   ~ 1.0σ per 64-bar window
    # Lower:  exp(-MULT_LOWER * vol_4h)   ~ 1.0σ per 64-bar window
    VOL_MULT_UPPER = 1.0
    VOL_MULT_LOWER = 1.0
    upper_mult = np.exp(VOL_MULT_UPPER * vol_4h)
    lower_mult = np.exp(-VOL_MULT_LOWER * vol_4h)

    labels = np.full(n, np.nan, dtype=np.float64)

    # Use raw close for market segmentation — continuous_price has zero
    # variance in the merged multi-market feature matrix.
    cp = close
    market_id = (np.log10(np.maximum(cp, 1e-9)) // 0.5).astype(int)

    # Process each contiguous market segment separately
    seg_start = 0
    mid_prev = market_id[0]
    for i in range(1, n + 1):
        if i == n or market_id[i] != mid_prev:
            seg_end = i
            _compute_barriers_segment(
                seg_start, seg_end, H_BARS, close, high, low,
                upper_mult, lower_mult, labels
            )
            if i < n:
                seg_start = i
                mid_prev = market_id[i]

    values = [float(v) if np.isfinite(v) else None for v in labels]
    df = df.with_columns(pl.Series('target_tb', values))
    return df


def _compute_barriers_segment(start, end, H_BARS, close, high, low,
                              upper_mult, lower_mult, labels):
    """Compute triple-barrier labels for a contiguous single-market segment."""
    for t in range(start, end - H_BARS):
        entry = close[t]
        if entry <= 0:
            continue
        window_end = min(t + 1 + H_BARS, end)

        upper_barrier = entry * upper_mult[t]
        lower_barrier = entry * lower_mult[t]

        high_window = high[t + 1:window_end]
        low_window = low[t + 1:window_end]

        upper_hit = np.argmax(high_window >= upper_barrier)
        lower_hit = np.argmax(low_window <= lower_barrier)

        wl = window_end - t - 1
        upper_idx = int(upper_hit) if high_window[upper_hit] >= upper_barrier else wl
        lower_idx = int(lower_hit) if low_window[lower_hit] <= lower_barrier else wl

        if upper_idx < wl and upper_idx <= lower_idx:
            labels[t] = 1.0
        elif lower_idx < wl and lower_idx < upper_idx:
            labels[t] = -1.0
        else:
            labels[t] = 0.0
