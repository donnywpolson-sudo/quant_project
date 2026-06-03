"""
continuous_contract.py
Continuous contract construction via ratio adjustment.
Computes roll dates per symbol and builds price-adjusted continuous series.

Strategy:
  - For equity index futures (ES, NQ, YM, RTY): roll on the Thursday before
    the 3rd Friday of the contract month (HMUZ quarterly).
  - For energy futures (CL, NG): monthly rolls, ~7 days before expiry.
  - For bond futures (ZB, ZN): quarterly rolls on last business day
    before the first delivery notice day.
  - At each roll date, the ratio (front_close / back_close) is computed
    and applied as a cumulative multiplier to pre-roll prices, producing
    a continuous price series free of roll-date jumps.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Standard quarterly roll schedule for equity index futures (HMUZ)
EQUITY_ROLL_MONTHS = {3, 6, 9, 12}

# Energy futures: monthly rolls, roll ~1 week before expiry
ENERGY_ROLL_DAYS_BEFORE = 7

# Contract-month letter codes
_MONTH_CODE = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
}


def _third_friday(year: int, month: int) -> datetime:
    """Return the 3rd Friday of a given month and year (UTC)."""
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    # weekday(): Monday=0, Sunday=6; Friday=4
    days_to_fri = (4 - first.weekday()) % 7
    first_fri = first + timedelta(days=days_to_fri)
    return first_fri + timedelta(days=14)


def _contract_code(month: int) -> str:
    """Futures month code: F(1) G(2) H(3) J(4) K(5) M(6)
       N(7) Q(8) U(9) V(10) X(11) Z(12)."""
    return _MONTH_CODE.get(month, '?')


def compute_roll_dates(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    rule: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute roll dates for a given futures symbol.

    For equity index futures (ES, NQ, YM, RTY), rolls occur on the
    Thursday before the 3rd Friday of March/June/September/December.
    For energy futures (CL, NG) and bond futures (ZB, ZN), a simplified
    monthly/quarterly schedule is used.

    Args:
        symbol: Ticker (ES, CL, ZB, etc.).
        start_date: Start of the date range.
        end_date: End of the date range.
        rule: Optional override for the roll rule (reserved for future use).

    Returns:
        DataFrame with columns:
          [roll_date, front_contract, back_contract, front_month, back_month]
    """
    rolls = []

    # Normalize dates to UTC for comparison with timezone-aware roll dates
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    current = start_date

    equity_symbols = {'ES', 'MES', 'NQ', 'YM', 'RTY'}
    energy_symbols = {'CL', 'NG'}
    bond_symbols = {'ZB', 'ZN', 'ZT', 'ZF'}

    if symbol in equity_symbols:
        # Quarterly HMUZ rolls: 3rd Friday minus 1 day (Thursday)
        year = current.year
        while year <= end_date.year:
            for month in sorted(EQUITY_ROLL_MONTHS):
                third_fri = _third_friday(year, month)
                roll_date = third_fri - timedelta(days=1)  # Thursday
                if start_date <= roll_date <= end_date:
                    front_month = month
                    back_month = month + 3 if month <= 9 else month - 9
                    back_year = year if month <= 9 else year + 1
                    rolls.append({
                        'roll_date': roll_date,
                        'front_contract': f'{symbol}{_contract_code(front_month)}{str(year)[-2:]}',
                        'back_contract': f'{symbol}{_contract_code(back_month)}{str(back_year)[-2:]}',
                        'front_month': front_month,
                        'back_month': back_month,
                    })
            year += 1

    elif symbol in energy_symbols:
        # Monthly rolls: ~7 days before the 3rd Friday of each month
        # (simplified — actual CL expiry is the 3rd business day before
        #  the 25th calendar day of the month preceding delivery)
        year = current.year
        while year <= end_date.year:
            for month in range(1, 13):
                roll_date = _third_friday(year, month) - timedelta(days=ENERGY_ROLL_DAYS_BEFORE)
                if start_date <= roll_date <= end_date:
                    front_month = month
                    back_month = month + 1 if month < 12 else 1
                    back_year = year if month < 12 else year + 1
                    rolls.append({
                        'roll_date': roll_date,
                        'front_contract': f'{symbol}{_contract_code(front_month)}{str(year)[-2:]}',
                        'back_contract': f'{symbol}{_contract_code(back_month)}{str(back_year)[-2:]}',
                        'front_month': front_month,
                        'back_month': back_month,
                    })
            year += 1

    elif symbol in bond_symbols:
        # Quarterly HMUZ rolls (same schedule as equity futures)
        year = current.year
        while year <= end_date.year:
            for month in sorted(EQUITY_ROLL_MONTHS):
                # Bond roll date: last 5 business days of contract month
                # Simplified: use 3rd Friday minus 7 days
                third_fri = _third_friday(year, month)
                roll_date = third_fri - timedelta(days=7)
                if start_date <= roll_date <= end_date:
                    front_month = month
                    back_month = month + 3 if month <= 9 else month - 9
                    back_year = year if month <= 9 else year + 1
                    rolls.append({
                        'roll_date': roll_date,
                        'front_contract': f'{symbol}{_contract_code(front_month)}{str(year)[-2:]}',
                        'back_contract': f'{symbol}{_contract_code(back_month)}{str(back_year)[-2:]}',
                        'front_month': front_month,
                        'back_month': back_month,
                    })
            year += 1

    else:
        logger.warning(
            f'No roll schedule defined for {symbol}; returning empty roll dates.'
        )

    if not rolls:
        return pl.DataFrame(schema={
            'roll_date': pl.Datetime('us'),
            'front_contract': pl.Utf8,
            'back_contract': pl.Utf8,
            'front_month': pl.Int32,
            'back_month': pl.Int32,
        })

    return pl.DataFrame(rolls)


def build_ratio_adjusted_series(
    df_front: pl.DataFrame,
    df_back: pl.DataFrame,
    roll_date: datetime,
) -> pl.DataFrame:
    """
    Build a ratio-adjusted continuous price series for one roll point.

    At the roll date, compute the adjustment factor as:
        factor = front_close / back_close  (at the roll bar)

    All pre-roll prices are multiplied by this factor so that the
    resulting series has no price discontinuity at the roll point.

    NOTE: This function is the test-helper / single-roll path.
    Production uses ``build_continuous_series`` which handles
    multiple rolls via cumulative splice. Both paths set
    ``adjustment_factor`` to the per-roll ratio (the multiplier
    applied at that specific roll), so downstream consumers see
    consistent column semantics.

    Args:
        df_front: Front-month (expiring) contract data.
                  Must contain [ts_event, close, open, high, low, volume].
        df_back:  Back-month (next) contract data with the same schema.
        roll_date: Timestamp at which the switch from front to back occurs.

    Returns:
        DataFrame with columns:
          [ts_event, open, high, low, close, volume, adjustment_factor,
           continuous_price]
    """
    eps = 1e-12

    df_front = df_front.sort('ts_event')
    df_back = df_back.sort('ts_event')

    # Split at roll_date: pre-roll uses front, post-roll uses back
    pre_roll = df_front.filter(pl.col('ts_event') <= roll_date)
    post_roll = df_back.filter(pl.col('ts_event') > roll_date)

    # Find the closing prices at the roll point
    front_at_roll = df_front.filter(pl.col('ts_event') == roll_date)
    back_at_roll = df_back.filter(pl.col('ts_event') == roll_date)

    if front_at_roll.is_empty() or back_at_roll.is_empty():
        logger.warning(
            f'No exact match at roll date {roll_date}; using last bar before roll.'
        )
        front_at_roll = df_front.filter(
            pl.col('ts_event') <= roll_date
        ).tail(1)
        back_at_roll = df_back.filter(
            pl.col('ts_event') <= roll_date
        ).tail(1)

    if not front_at_roll.is_empty() and not back_at_roll.is_empty():
        front_close = front_at_roll['close'][0]
        back_close = back_at_roll['close'][0]
        ratio = front_close / max(back_close, eps)
    else:
        ratio = 1.0
        logger.warning(
            f'Could not compute ratio at roll {roll_date}; using 1.0 (no adjustment).'
        )

    # Apply adjustment to pre-roll data
    ohlc_cols = ['open', 'high', 'low', 'close']
    pre_roll = pre_roll.with_columns([
        pl.lit(ratio, dtype=pl.Float32).alias('adjustment_factor'),
    ])
    pre_roll = pre_roll.with_columns([
        (pl.col(c) * pl.col('adjustment_factor')).alias('continuous_price')
        if c == 'close'
        else (pl.col(c) * pl.lit(ratio, dtype=pl.Float32)).alias(f'continuous_{c}')
        for c in ohlc_cols
    ])

    post_roll = post_roll.with_columns([
        pl.lit(1.0, dtype=pl.Float32).alias('adjustment_factor'),
    ])
    post_roll = post_roll.with_columns([
        pl.col(c).alias('continuous_price')
        if c == 'close'
        else pl.col(c).alias(f'continuous_{c}')
        for c in ohlc_cols
    ])

    combined = pl.concat([pre_roll, post_roll], how='diagonal_relaxed').sort('ts_event')
    return combined


def apply_splice(
    df: pl.DataFrame,
    adjustments: pl.DataFrame,
) -> pl.DataFrame:
    """
    Apply cumulative adjustment factors to splice multiple contracts.

    Multiplies the close price (and optionally open/high/low) by the
    cumulative adjustment factor at each bar so that the resulting
    'continuous_price' column reflects a price series free of roll jumps.

    Args:
        df: Price data with [ts_event, open, high, low, close, volume].
        adjustments: DataFrame with [ts_event, adjustment_factor,
                      cumulative_factor]. The cumulative_factor is the
                      product of all adjustment_factors up to that roll.

    Returns:
        DataFrame with 'continuous_price' column added (and
        'adjustment_factor' / 'contract_month' preserved if present).
    """
    df = df.sort('ts_event')
    adjustments = adjustments.sort('ts_event')

    # Join adjustments onto the price data
    df = df.join(
        adjustments.select(['ts_event', 'cumulative_factor']),
        on='ts_event',
        how='left',
    )

    # Forward-fill the cumulative factor so every bar has one
    df = df.with_columns(
        pl.col('cumulative_factor')
        .fill_null(strategy='forward')
        .fill_null(1.0)
    )

    # Compute continuous_price = close * cumulative_factor
    df = df.with_columns(
        (pl.col('close') * pl.col('cumulative_factor'))
        .alias('continuous_price')
    )

    return df


def build_continuous_series(
    df: pl.DataFrame,
    symbol: str,
    contract_multiplier: float = 1.0,
) -> pl.DataFrame:
    """
    Full pipeline: compute roll dates, build ratio-adjusted series,
    and apply cumulative splice factors to produce a continuous price
    series with adjustment metadata.

    This is the high-level entry point called from ingest.py.

    Args:
        df: Aligned 5-min OHLCV data with [ts_event, open, high, low,
            close, volume, session_id].
        symbol: Ticker string (e.g., 'ES', 'CL', 'ZB').
        contract_multiplier: Contract multiplier for the symbol
            (ES=50, CL=1000, ZB=1000).

    Returns:
        DataFrame with added columns:
          [continuous_price, adjustment_factor, contract_month,
           contract_multiplier, continuous_open, continuous_high,
           continuous_low]
    """
    if df.is_empty():
        raise ValueError(
            f'build_continuous_series: received empty DataFrame for {symbol} '
            f'— upstream filtering may have removed all rows.'
        )

    df = df.sort('ts_event')
    start_ts = df['ts_event'].min()
    end_ts = df['ts_event'].max()

    logger.info(
        f'Building continuous contract series for {symbol} '
        f'from {start_ts} to {end_ts}'
    )

    roll_dates_df = compute_roll_dates(symbol, start_ts, end_ts)

    if roll_dates_df.is_empty():
        logger.info(
            f'No roll dates found for {symbol} in range; '
            f'using unchanged price as continuous_price.'
        )
        ohlc_cols = ['open', 'high', 'low', 'close']
        df = df.with_columns([
            pl.lit(1.0, dtype=pl.Float32).alias('adjustment_factor'),
            pl.lit(symbol, dtype=pl.Utf8).alias('contract_month'),
            pl.lit(contract_multiplier, dtype=pl.Float32).alias(
                'contract_multiplier'
            ),
        ])
        df = df.with_columns([
            pl.col(c).alias('continuous_price')
            if c == 'close'
            else pl.col(c).alias(f'continuous_{c}')
            for c in ohlc_cols
        ])
        return df

    logger.info(
        f'Found {roll_dates_df.height} roll dates for {symbol}'
    )

    # Build cumulative adjustment factors across all roll dates
    roll_dates = roll_dates_df.sort('roll_date')['roll_date'].to_list()
    cumulative_factor = 1.0
    adjustment_rows = []
    ohlc_cols = ['open', 'high', 'low', 'close']

    # Start with the initial contract
    current_contract = roll_dates_df['front_contract'][0]

    for i, roll_date in enumerate(roll_dates):
        # At each roll, compute ratio using adjacent bars
        roll_bar_front = df.filter(
            pl.col('ts_event') <= roll_date
        ).tail(1)
        roll_bar_back = df.filter(
            pl.col('ts_event') > roll_date
        ).head(1)

        if not roll_bar_front.is_empty() and not roll_bar_back.is_empty():
            front_close = roll_bar_front['close'][0]
            back_close = roll_bar_back['close'][0]
            ratio = front_close / max(back_close, 1e-12)
        else:
            ratio = 1.0

        cumulative_factor *= ratio

        if not np.isfinite(cumulative_factor) or cumulative_factor <= 0.0:
            raise RuntimeError(
                f'CONTRACT FAIL: cumulative_factor became {cumulative_factor:.6f} '
                f'at roll {roll_date} (ratio={ratio:.6f}). '
                f'Cannot build continuous series — adjustment factor is invalid.'
            )

        front_contract = roll_dates_df['front_contract'][i]
        back_contract = roll_dates_df['back_contract'][i]

        adjustment_rows.append({
            'ts_event': roll_date,
            'adjustment_factor': ratio,
            'cumulative_factor': cumulative_factor,
            'front_contract': front_contract,
            'back_contract': back_contract,
            'contract_month': back_contract,
        })

    # Sanity guard: cumulative_factor should stay within reasonable bounds.
    # Over very long histories with persistent contango/backwardation the
    # factor may drift, but extreme values indicate data or roll logic issues.
    if not (0.01 <= cumulative_factor <= 50.0):
        logger.warning(
            'cumulative_factor=%s for %s over %d rolls is outside safe range [0.01, 50.0]. '
            'Check roll date computation and price ratio integrity.',
            round(cumulative_factor, 6), symbol, len(roll_dates)
        )

    adjustments_df = pl.DataFrame(adjustment_rows)

    # Align time unit with the main DataFrame to avoid join schema errors
    data_dtype = df['ts_event'].dtype
    adjustments_df = adjustments_df.with_columns(
        pl.col('ts_event').cast(data_dtype)
    )

    # Apply splice: join cumulative factors and compute continuous_price
    df = df.join(
        adjustments_df.select(['ts_event', 'cumulative_factor', 'contract_month']),
        on='ts_event',
        how='left',
    )

    # Forward-fill cumulative_factor and contract_month
    df = df.with_columns([
        pl.col('cumulative_factor')
        .fill_null(strategy='forward')
        .fill_null(1.0),
        pl.col('contract_month')
        .fill_null(strategy='forward')
        .fill_null(current_contract),
    ])

    # Add adjustment_factor: the per-roll ratio, derived from cumulative
    # factors. Equivalent to the ratio set by build_ratio_adjusted_series
    # at each roll point — both paths produce the same per-roll multiplier
    # for downstream consumers (1.0 between rolls, ratio at roll bars).
    df = df.with_columns(
        pl.when(pl.col('cumulative_factor').is_not_null())
        .then(
            pl.col('cumulative_factor')
            / pl.col('cumulative_factor').shift(1).fill_null(1.0)
        )
        .otherwise(pl.lit(1.0, dtype=pl.Float32))
        .fill_null(1.0)
        .alias('adjustment_factor')
    )

    # Add contract_multiplier
    df = df.with_columns(
        pl.lit(contract_multiplier, dtype=pl.Float32).alias(
            'contract_multiplier'
        )
    )

    # Compute continuous_price = close * cumulative_factor
    df = df.with_columns(
        (pl.col('close') * pl.col('cumulative_factor'))
        .alias('continuous_price')
    )

    # Also produce continuous OHLC columns for reference
    df = df.with_columns([
        (pl.col(c) * pl.col('cumulative_factor')).alias(f'continuous_{c}')
        for c in ohlc_cols
    ])

    logger.info(
        f'Continuous contract series built for {symbol}: '
        f'{roll_dates_df.height} rolls applied, '
        f'final cumulative_factor={cumulative_factor:.6f}'
    )

    return df
