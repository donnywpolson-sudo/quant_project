"""
simulator.py
Realistic execution simulator with economic friction.
- 1.5 bps fixed cost per round-turn (applied to position changes)
- Signal generated at bar t, executed at bar t+1 open
- Volatility computed from lagged returns [t-N, t-1]
- Position changes tracked for turnover calculation
- HTF context-aware trade gating (directional bias, vol scaling, trend alignment)
- Session close flatting (zero position within 5 min of 16:00 ET)
- Intrabar stop/take-profit with linear path logic and gap simulation
"""
import numpy as np
import numba
import polars as pl
from pipeline.common.config import config

# Fixed contract size multiplier for position sizing
FIXED_CONTRACT_SIZE = 1.0


def _cfg_float(name: str, default: float = 0.0) -> float:
    value = getattr(config, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _transaction_cost_bps() -> float:
    if hasattr(config, 'TRANSACTION_COST_BPS'):
        return _cfg_float('TRANSACTION_COST_BPS')
    if hasattr(config, 'TX_COST_BPS'):
        return _cfg_float('TX_COST_BPS')
    # Backward compatibility: existing TX_COST_PER_ROUNDTURN is decimal
    # round-turn cost. A position change is one side, so divide by 2.
    return _cfg_float('TX_COST_PER_ROUNDTURN') * 10000.0 / 2.0


def _slippage_bps() -> float:
    if hasattr(config, 'SLIPPAGE_BPS'):
        return _cfg_float('SLIPPAGE_BPS')
    # Backward compatibility: existing SLIPPAGE_K is treated as decimal bps rate.
    return _cfg_float('SLIPPAGE_K') * 10000.0


@numba.njit(cache=True)
def simulate_intrabar_stops(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    position_arr: np.ndarray,
    stop_pct: float,
    target_pct: float,
    gap_slippage_pct: float,
    eps: float = 1e-09,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate intrabar stop-loss and take-profit fills using linear-price-path
    logic.  For every row with a non-zero position, the bar's high/low are
    checked against stop/target levels placed relative to the entry price
    (the bar's open).  Gap openings — where the open is already beyond the
    stop level — are filled at the open plus an adverse slippage allowance.

    Args:
        open_arr:    bar open prices (aligned: position_i entered at open_i).
        high_arr:    bar high prices.
        low_arr:     bar low prices.
        position_arr: signed position size at the start of each bar.
        stop_pct:    stop-loss distance, as a decimal (e.g. 0.005 = 0.5 %).
        target_pct:  take-profit distance, as a decimal (e.g. 0.01 = 1.0 %).
        gap_slippage_pct: additional adverse slippage for gap fills (decimal).
        eps:         small epsilon to avoid division-by-zero.

    Returns:
        adj_position:  position array after intrabar fills (0.0 where filled).
        intrabar_pnl:  per-row PnL contribution from intrabar fills (price units).
    """
    n = len(open_arr)
    adj_position = position_arr.copy()
    intrabar_pnl = np.zeros(n, dtype=np.float64)

    if stop_pct <= 0.0 and target_pct <= 0.0:
        return adj_position, intrabar_pnl

    for i in range(n):
        pos = position_arr[i]
        if abs(pos) < eps:
            continue

        entry = open_arr[i]
        if entry < eps:
            continue

        is_long = pos > 0.0

        # Stop and target absolute price levels
        if is_long:
            stop_level = entry * (1.0 - stop_pct)
            target_level = entry * (1.0 + target_pct)
        else:
            stop_level = entry * (1.0 + stop_pct)
            target_level = entry * (1.0 - target_pct)

        bar_high = high_arr[i]
        bar_low = low_arr[i]

        # ----------------------------------------------------------------
        # 1. Gap opening check: if the bar opens already beyond the stop
        #    level, the stop is triggered immediately at the open, plus
        #    gap_slippage_pct of adverse movement.
        # ----------------------------------------------------------------
        gap_filled = False
        fill_price = 0.0

        if is_long and entry <= stop_level:
            # Gapped down through stop — fill at open minus slippage
            gap_filled = True
            fill_price = entry * (1.0 - gap_slippage_pct)
        elif (not is_long) and entry >= stop_level:
            # Gapped up through stop — fill at open plus slippage
            gap_filled = True
            fill_price = entry * (1.0 + gap_slippage_pct)

        if gap_filled:
            pnl_impact = (fill_price - entry) * float(np.sign(pos)) * abs(pos)
            intrabar_pnl[i] = pnl_impact
            adj_position[i] = 0.0
            continue

        # ----------------------------------------------------------------
        # 2. Linear-path intrabar check: determine which level
        #    (stop or target) is touched first within the bar's range.
        # ----------------------------------------------------------------
        stop_hit = False
        target_hit = False

        if is_long:
            if bar_low <= stop_level:
                stop_hit = True
            if bar_high >= target_level:
                target_hit = True

            if stop_hit and target_hit:
                # Both levels reached — use distance to determine first touch
                dist_to_stop = entry - stop_level   # positive distance
                dist_to_target = target_level - entry  # positive distance
                if dist_to_target <= dist_to_stop:
                    # Target is closer → hit first
                    fill_price = target_level
                    stop_hit = False  # target overrides
                else:
                    fill_price = stop_level
                    target_hit = False
            elif stop_hit:
                fill_price = stop_level
            elif target_hit:
                fill_price = target_level
            else:
                continue  # neither level touched — hold through bar
        else:
            # Short
            if bar_high >= stop_level:
                stop_hit = True
            if bar_low <= target_level:
                target_hit = True

            if stop_hit and target_hit:
                dist_to_stop = stop_level - entry    # positive distance
                dist_to_target = entry - target_level  # positive distance
                if dist_to_target <= dist_to_stop:
                    fill_price = target_level
                    stop_hit = False
                else:
                    fill_price = stop_level
                    target_hit = False
            elif stop_hit:
                fill_price = stop_level
            elif target_hit:
                fill_price = target_level
            else:
                continue

        # PnL = (fill - entry) * sign(pos) * |pos|
        pnl_impact = (fill_price - entry) * float(np.sign(pos)) * abs(pos)
        intrabar_pnl[i] = pnl_impact
        adj_position[i] = 0.0

    return adj_position, intrabar_pnl


def simulate_execution_classification(df: pl.DataFrame) -> pl.DataFrame:
    eps = config.EPS

    # ------------------------------------------------------------------------
    # 1. Generate trading signal from prediction probability.
    #    Use absolute probability thresholds: prob > 0.55 → long,
    #    prob < 0.45 → short.  This preserves the model's directional
    #    prediction directly, unlike z-score which measures prediction
    #    SURPRISE (deviation from recent average) — a metric that goes
    #    to zero when the model is consistently correct, suppressing all
    #    signals.
    # ------------------------------------------------------------------------
    prob = pl.col('prediction_prob').fill_null(0.5)
    signal_expr = (
        pl.when(prob > 0.55)
        .then(pl.lit(1.0, dtype=pl.Float32))
        .when(prob < 0.45)
        .then(pl.lit(-1.0, dtype=pl.Float32))
        .otherwise(pl.lit(0.0, dtype=pl.Float32))
    )
    df = df.with_columns(signal_expr.alias('raw_signal'))

    # Initialize target_exec from raw_signal as Float32
    df = df.with_columns(
        pl.col('raw_signal').cast(pl.Float32).alias('target_exec')
    )

    # ------------------------------------------------------------------------
    # 2. HTF Directional Bias: gate trades against HTF hourly trend alignment.
    #    Long only if htf_hourly_trend_alignment > 0 (suppress longs otherwise)
    #    Short only if htf_hourly_trend_alignment < 0 (suppress shorts otherwise)
    #    Strict zero-boundary gating — no neutral zone threshold.
    # ------------------------------------------------------------------------
    col_names = df.columns
    if 'htf_hourly_trend_alignment' in col_names:
        htf_align = (
            pl.col('htf_hourly_trend_alignment')
            .fill_null(0.0)
            .cast(pl.Float32)
        )
        target = pl.col('target_exec')

        target_gated = (
            pl.when((target > 0.0) & (htf_align <= 0.0))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .when((target < 0.0) & (htf_align >= 0.0))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(target)
        )
        df = df.with_columns(target_gated.alias('target_exec'))

    # ------------------------------------------------------------------------
    # 3. Session break filter (flatten position during exchange break)
    # ------------------------------------------------------------------------
    df = df.with_columns(
        pl.col('ts_event')
        .dt.convert_time_zone(config.TIMEZONE)
        .dt.time()
        .alias('t_local')
    )
    df = df.with_columns(
        pl.when(
            (pl.col('t_local') >= pl.lit(config.SESSION_BREAK_START_LOCAL))
            & (pl.col('t_local') < pl.lit(config.SESSION_BREAK_END_LOCAL))
        )
        .then(pl.lit(0.0, dtype=pl.Float32))
        .otherwise(pl.col('target_exec'))
        .alias('target_exec')
    )

    # ------------------------------------------------------------------------
    # 4. Session Close Flatting: zero out position within
    #    FLAT_BEFORE_CLOSE_MINUTES of SESSION_END_LOCAL (16:00 ET).
    # ------------------------------------------------------------------------
    close_time = config.SESSION_END_LOCAL  # time(16, 0)
    flat_minutes = config.FLAT_BEFORE_CLOSE_MINUTES  # 5
    t_minutes = (
        pl.col('t_local').dt.hour().cast(pl.Int32) * 60
        + pl.col('t_local').dt.minute().cast(pl.Int32)
    )
    close_minutes = close_time.hour * 60 + close_time.minute  # 960
    flat_start_minutes = close_minutes - flat_minutes          # 955 (15:55)
    df = df.with_columns(
        pl.when(t_minutes >= pl.lit(flat_start_minutes, dtype=pl.Int32))
        .then(pl.lit(0.0, dtype=pl.Float32))
        .otherwise(pl.col('target_exec'))
        .alias('target_exec')
    )
    df = df.drop('t_local')

    # ------------------------------------------------------------------------
    # 5. HTF Volatility Scaling: scale position size by
    #    1 / (htf_daily_vol_5 * HTF_VOL_WINDOW), capped at MAX_LEVERAGE.
    #    Only applied when config.HTF_VOL_SCALING is enabled and the
    #    column exists in the DataFrame.
    # ------------------------------------------------------------------------
    if config.HTF_VOL_SCALING and ('htf_daily_vol_5' in col_names):
        htf_vol = (
            pl.col('htf_daily_vol_5')
            .fill_null(1e-06)
            .cast(pl.Float32)
            .clip(eps, None)
        )
        vol_scaler = (
            pl.lit(1.0, dtype=pl.Float32)
            / (htf_vol * pl.lit(config.HTF_VOL_WINDOW, dtype=pl.Float32))
        ).clip(
            pl.lit(-config.MAX_LEVERAGE, dtype=pl.Float32),
            pl.lit(config.MAX_LEVERAGE, dtype=pl.Float32),
        )
        df = df.with_columns(
            (pl.col('target_exec') * vol_scaler)
            .cast(pl.Float32)
            .alias('target_exec')
        )

    # ------------------------------------------------------------------------
    # 6. ATR-Based Volatility-Parity Position Sizing
    #    S = TARGET_RISK_PER_TRADE / ATR_pct(14)
    #    ATR computed as rolling mean of bar range as percentage of close.
    #    Position size capped at MAX_LEVERAGE. Falls back to FIXED_CONTRACT_SIZE.
    # ------------------------------------------------------------------------
    bar_range_pct = ((pl.col('high') - pl.col('low')) / pl.col('close').clip(eps, None)).clip(eps, None)
    atr14_pct = bar_range_pct.shift(1).rolling_mean(window_size=14, min_samples=5).clip(eps, None)

    volatility_size = (
        pl.lit(config.TARGET_RISK_PER_TRADE, dtype=pl.Float32) / atr14_pct
    ).clip(
        pl.lit(0.1, dtype=pl.Float32),
        pl.lit(config.MAX_LEVERAGE, dtype=pl.Float32),
    ).fill_null(pl.lit(FIXED_CONTRACT_SIZE, dtype=pl.Float32))

    df = df.with_columns(
        (pl.col('target_exec') * volatility_size)
        .cast(pl.Float32)
        .alias('target_exec')
    )

    # ------------------------------------------------------------------------
    # 6b. Position clipping: max_position_size and notional cap
    #     using per-market contract_multiplier from market config.
    #     Symbol is resolved from config.CURRENT_SYMBOL, set by cli.py
    #     when the command is invoked. Fails fast if unset — no silent
    #     fallback to ES because wrong multiplier silently corrupts PnL.
    # ------------------------------------------------------------------------
    from pipeline.common.config import config as _cfg
    from pipeline.common.market import get_contract_multiplier

    symbol = getattr(_cfg, 'CURRENT_SYMBOL', None)
    if not symbol:
        raise RuntimeError(
            'CONTRACT FAIL: CURRENT_SYMBOL is unset. '
            'Cannot resolve contract multiplier for position clipping. '
            'Ensure cli.py sets config.CURRENT_SYMBOL before execution.'
        )
    contract_multiplier = get_contract_multiplier(symbol)

    import yaml
    from pathlib import Path
    market_cfg_path = _cfg.MARKET_CONFIGS.get(symbol)
    with open(market_cfg_path, 'r') as f:
        market_cfg = yaml.safe_load(f) or {}
    max_pos_size_raw = market_cfg.get('risk', {}).get('max_position_size')
    max_pos = float(max_pos_size_raw) if max_pos_size_raw else float('inf')

    # Clip to max_position_size
    df = df.with_columns(
        pl.col('target_exec')
        .clip(pl.lit(-max_pos, dtype=pl.Float32), pl.lit(max_pos, dtype=pl.Float32))
        .alias('target_exec')
    )

    # Notional clip: |target_exec| <= MAX_LEVERAGE (equity-normalized)
    open_current = pl.col('open').clip(config.EPS, None)
    equity = getattr(config, 'EQUITY', 100000.0)
    max_notional = equity * config.MAX_LEVERAGE
    df = df.with_columns(
        pl.col('target_exec')
        .clip(
            pl.lit(-max_notional, dtype=pl.Float32) / (open_current * pl.lit(contract_multiplier, dtype=pl.Float32)),
            pl.lit(max_notional, dtype=pl.Float32) / (open_current * pl.lit(contract_multiplier, dtype=pl.Float32))
        )
        .alias('target_exec')
    )

    # ------------------------------------------------------------------------
    # 7. HTF Trend Alignment: if config.HTF_TREND_ALIGNMENT is enabled,
    #    suppress signals that disagree with htf_daily_trend_slope_10.
    #    - Suppress longs when daily trend is strongly down (< -threshold)
    #    - Suppress shorts when daily trend is strongly up (> +threshold)
    # ------------------------------------------------------------------------
    if config.HTF_TREND_ALIGNMENT and ('htf_daily_trend_slope_10' in col_names):
        daily_trend = (
            pl.col('htf_daily_trend_slope_10')
            .fill_null(0.0)
            .cast(pl.Float32)
        )
        trend_threshold = pl.lit(config.HTF_TREND_THRESHOLD, dtype=pl.Float32)
        target = pl.col('target_exec')

        target_aligned = (
            pl.when((target > 0.0) & (daily_trend < -trend_threshold))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .when((target < 0.0) & (daily_trend > trend_threshold))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(target)
        )
        df = df.with_columns(target_aligned.alias('target_exec'))

    # ------------------------------------------------------------------------
    # 8. Volatility: strictly lagged, computed from returns [t-N, t-1]
    # ------------------------------------------------------------------------
    ret = (pl.col('close') / pl.col('close').shift(1)).log()
    ret_lagged = ret.shift(1)
    vol = ret_lagged.rolling_std(window_size=20).clip(eps, None)
    df = df.with_columns(vol.fill_null(1e-06).alias('vol'))

    # ------------------------------------------------------------------------
    # 9. Spread proxy (bid-ask approximation)
    # ------------------------------------------------------------------------
    spread = (pl.col('high') - pl.col('low')) / pl.col('close').clip(eps, None)
    spread = spread.clip(0.0, 0.05)
    df = df.with_columns(spread.alias('spread'))

    # ------------------------------------------------------------------------
    # 10. Per-bar bps cost rate charged on position CHANGE.
    #     Dollar conversion happens in _compute_pnl_from_target_exec via:
    #       notional_traded * bps / 10000
    # ------------------------------------------------------------------------
    unit_cost = pl.lit(
        max((_transaction_cost_bps() + _slippage_bps()) / 10000.0, 0.0),
        dtype=pl.Float32,
    )
    df = df.with_columns(unit_cost.alias('unit_cost'))

    df = _compute_pnl_from_target_exec(df, contract_multiplier)
    return validate_execution_simulation_result(df)


def _compute_pnl_from_target_exec(df: pl.DataFrame, contract_multiplier: float = 1.0) -> pl.DataFrame:
    """
    Compute position, intrabar stops, and PnL from target_exec.

    This is the full PnL calculation pipeline used by both the main
    execution path and the HMM recompute path.  It assumes that
    ``target_exec`` (the signed, sized, gated position signal),
    ``vol`` and ``spread`` may exist on the DataFrame. Costs are computed
    in dollars from absolute position change, open price, contract multiplier,
    commission_per_contract, transaction bps, and slippage bps.

    Includes:
      - Execution return: buy at t+1 open, sell at t+1 close
      - Position: signal from t-1 executed at t
      - Position change (turnover cost basis)
      - Intrabar stop-loss / take-profit with gap-slippage logic
      - PnL = pos * ret_exec * entry_price * multiplier
        + intrabar_pnl * multiplier
        - contracts_changed * commission_per_contract
        - notional_traded * transaction_cost_bps / 10000
        - notional_traded * slippage_bps / 10000
        - round-turn settlement on flatting
      - Proportional PnL clip (5 % of notional)
    """
    eps = config.EPS

    # Futures contracts are discrete. Fractional positions create impossible
    # fills and understate execution constraints for low-risk accounts.
    target_exec_sized = pl.col('target_exec').cast(pl.Float32)
    df = df.with_columns(target_exec_sized.alias('target_exec_sized'))
    df = df.with_columns(target_exec_sized.round(0).cast(pl.Float32).alias('target_exec'))

    # ------------------------------------------------------------------------
    # Execution return: signal[t-1] executed at open[t] earns return[t].
    # return[t] = (close[t] - open[t]) / open[t]
    # ------------------------------------------------------------------------
    ret_exec = ((pl.col('close') - pl.col('open')) / pl.col('open').clip(eps, None)).fill_null(0.0)
    ret_exec = ret_exec.clip(-0.02, 0.02)
    df = df.with_columns(ret_exec.alias('ret_exec'))

    # ------------------------------------------------------------------------
    # Compute position: signal is generated at t, executed at t+1.
    # So the position for PnL at bar t uses the signal from bar t-1.
    # ------------------------------------------------------------------------
    position = pl.col('target_exec').shift(1).fill_null(0.0)
    df = df.with_columns(position.alias('position'))

    # ------------------------------------------------------------------------
    # Position change (for transaction cost): |pos_t - pos_t-1|
    # ------------------------------------------------------------------------
    pos_change = (pl.col('position') - pl.col('position').shift(1)).abs()
    df = df.with_columns(pos_change.fill_null(0.0).alias('pos_change'))

    # ------------------------------------------------------------------------
    # Intrabar Stops / Take-Profit (preprocessing before PnL).
    # ------------------------------------------------------------------------
    open_vals = df['open'].to_numpy().astype(np.float64)
    high_vals = df['high'].to_numpy().astype(np.float64)
    low_vals = df['low'].to_numpy().astype(np.float64)
    pos_vals = df['position'].to_numpy().astype(np.float64)

    adj_position, intrabar_pnl = simulate_intrabar_stops(
        open_arr=open_vals,
        high_arr=high_vals,
        low_arr=low_vals,
        position_arr=pos_vals,
        stop_pct=config.STOP_LOSS_PCT,
        target_pct=config.TAKE_PROFIT_PCT,
        gap_slippage_pct=config.GAP_SLIPPAGE_PCT,
        eps=config.EPS,
    )

    df = df.with_columns(
        pl.Series('position', adj_position).cast(pl.Float32)
    )
    df = df.with_columns(
        pl.Series('intrabar_pnl', intrabar_pnl).cast(pl.Float32)
    )

    # Recompute pos_change after intrabar adjustments
    pos_change = (pl.col('position') - pl.col('position').shift(1)).abs()
    df = df.with_columns(pos_change.fill_null(0.0).alias('pos_change'))

    # ------------------------------------------------------------------------
    # PnL: position * forward return * contract_multiplier * entry_price
    #     - transaction costs + round-turn settlement on flatting
    #     + intrabar PnL (stop/take-profit fills within the bar).
    # ------------------------------------------------------------------------
    entry_price = pl.col('open')

    gross_pnl = pl.col('position') * pl.col('ret_exec') * entry_price * pl.lit(contract_multiplier, dtype=pl.Float32)
    gross_pnl = gross_pnl + pl.col('intrabar_pnl') * pl.lit(contract_multiplier, dtype=pl.Float32)
    gross_pnl = gross_pnl.fill_nan(0.0)
    pnl_clip = pl.lit(0.05, dtype=pl.Float32) * entry_price * pl.lit(contract_multiplier, dtype=pl.Float32)
    gross_pnl = gross_pnl.clip(-pnl_clip, pnl_clip)
    df = df.with_columns(gross_pnl.alias('gross_pnl'))

    notional_turnover = pl.col('pos_change') * entry_price * pl.lit(contract_multiplier, dtype=pl.Float32)
    commission_cost = pl.col('pos_change') * pl.lit(_cfg_float('COMMISSION_PER_CONTRACT'), dtype=pl.Float32)
    transaction_cost = notional_turnover * pl.lit(_transaction_cost_bps() / 10000.0, dtype=pl.Float32)
    slippage_cost = notional_turnover * pl.lit(_slippage_bps() / 10000.0, dtype=pl.Float32)
    total_cost = commission_cost + transaction_cost + slippage_cost
    df = df.with_columns([
        notional_turnover.alias('notional_traded'),
        commission_cost.alias('commission_cost'),
        transaction_cost.alias('transaction_cost'),
        slippage_cost.alias('slippage_cost'),
        total_cost.alias('execution_cost'),
        ((pl.lit(_transaction_cost_bps() + _slippage_bps(), dtype=pl.Float32)) / 10000.0).alias('unit_cost'),
    ])

    pnl = gross_pnl - pl.col('execution_cost')
    pnl = pnl.fill_nan(0.0)
    pnl = pnl.clip(-pnl_clip, pnl_clip)
    df = df.with_columns(pnl.alias('pnl'))

    starting_equity = float(getattr(config, 'EQUITY', 100000.0))
    if starting_equity <= 0:
        raise RuntimeError(f'ACCOUNTING FAIL: EQUITY must be positive, got {starting_equity}')
    df = df.with_columns([
        (pl.col('pnl') / pl.lit(starting_equity, dtype=pl.Float32)).alias('return_on_equity'),
        (pl.col('gross_pnl') / pl.lit(starting_equity, dtype=pl.Float32)).alias('gross_return_on_equity'),
    ])
    df = df.with_columns([
        (pl.lit(starting_equity, dtype=pl.Float32) + pl.col('pnl').cum_sum()).alias('equity_curve'),
        (pl.lit(starting_equity, dtype=pl.Float32) + pl.col('gross_pnl').cum_sum()).alias('gross_equity_curve'),
    ])
    df = df.with_columns(
        ((pl.col('equity_curve') / pl.col('equity_curve').cum_max()) - 1.0)
        .fill_nan(0.0)
        .alias('drawdown_pct')
    )

    return df


def validate_execution_simulation_result(df: pl.DataFrame) -> pl.DataFrame:
    """Step 9 hard gate: execution simulation output contract."""
    required = [
        'ts_event', 'prediction_prob', 'raw_signal',
        'target_exec', 'target_exec_sized',
        'position', 'pos_change', 'ret_exec',
        'gross_pnl', 'pnl',
        'notional_traded', 'commission_cost', 'transaction_cost',
        'slippage_cost', 'execution_cost',
        'return_on_equity', 'gross_return_on_equity',
        'equity_curve', 'gross_equity_curve', 'drawdown_pct',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'EXECUTION FAIL: missing columns {missing}')
    if df.height == 0:
        raise RuntimeError('EXECUTION FAIL: empty result')
    if df['ts_event'].null_count() > 0:
        raise RuntimeError('EXECUTION FAIL: null ts_event values')
    if not df['ts_event'].is_sorted():
        raise RuntimeError('EXECUTION FAIL: ts_event not sorted')
    if df['ts_event'].n_unique() != df.height:
        raise RuntimeError('EXECUTION FAIL: duplicate ts_event values')

    numeric_cols = [c for c in required if c != 'ts_event']
    for col in numeric_cols:
        vals = df[col].to_numpy()
        if not np.isfinite(vals.astype(np.float64)).all():
            raise RuntimeError(f'EXECUTION FAIL: non-finite values in {col}')

    probs = df['prediction_prob'].to_numpy().astype(np.float64)
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise RuntimeError('EXECUTION FAIL: prediction_prob outside [0, 1]')

    raw_signal = df['raw_signal'].to_numpy().astype(np.float64)
    if not np.isin(raw_signal, [-1.0, 0.0, 1.0]).all():
        raise RuntimeError('EXECUTION FAIL: raw_signal outside {-1,0,1}')

    target_exec = df['target_exec'].to_numpy().astype(np.float64)
    if not np.allclose(target_exec, np.round(target_exec), atol=1e-6):
        raise RuntimeError('EXECUTION FAIL: target_exec contains fractional contracts')

    non_negative_cols = [
        'pos_change', 'notional_traded', 'commission_cost',
        'transaction_cost', 'slippage_cost', 'execution_cost',
    ]
    for col in non_negative_cols:
        vals = df[col].to_numpy().astype(np.float64)
        if np.any(vals < -1e-9):
            raise RuntimeError(f'EXECUTION FAIL: negative values in {col}')

    gross = df['gross_pnl'].to_numpy().astype(np.float64)
    net = df['pnl'].to_numpy().astype(np.float64)
    costs = df['execution_cost'].to_numpy().astype(np.float64)
    if not np.allclose(net, gross - costs, atol=1e-3):
        raise RuntimeError('EXECUTION FAIL: pnl != gross_pnl - execution_cost')

    return df


def run_execution_simulation(df: pl.DataFrame) -> pl.DataFrame:
    """Step 9 boundary: signal shift, sizing, costs/slippage, and PnL."""
    if 'prediction_prob' not in df.columns:
        raise RuntimeError('EXECUTION FAIL: prediction_prob missing before execution')
    return validate_execution_simulation_result(simulate_execution_classification(df))


def simulate_execution(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError(
        'Use simulate_execution_classification for new pipeline.'
    )
