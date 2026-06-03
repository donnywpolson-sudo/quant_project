import logging
import importlib
import json
import os
import hashlib
import time
from datetime import timedelta
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.special import expit
from joblib import Parallel, delayed
from pipeline.common.config import config

from pipeline.execution.simulator import run_execution_simulation
from pipeline.features.corr_prune import correlation_prune
from pipeline.features.variance_filter import remove_constant_features
from pipeline.meta.meta_label import add_meta_label_target
from pipeline.meta.meta_gate import train_meta_model, apply_meta_gate
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pipeline.regime.hmm_filter import HMMRegimeFilter

def safe_clip(X, min_val=-10.0, max_val=10.0):
    return np.clip(X, min_val, max_val)

def safe_replace(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def robust_scale(X_train, X_test):
    med = np.median(X_train, axis=0)
    q1 = np.percentile(X_train, 25, axis=0)
    q3 = np.percentile(X_train, 75, axis=0)
    iqr = np.clip(q3 - q1, 1e-06, None)
    iqr = np.clip(iqr, 0.01, None)
    scale = np.where(iqr > 0, 1.0 / iqr, 1.0)
    X_train = (X_train - med) * scale
    X_test = (X_test - med) * scale
    X_train = np.clip(X_train, -10.0, 10.0)
    X_test = np.clip(X_test, -10.0, 10.0)
    return (X_train.astype(np.float32), X_test.astype(np.float32))

def stabilize_targets(y):
    y = safe_replace(y)
    return np.clip(y, -1.0, 1.0)

def train_and_predict(train_X: pl.DataFrame, train_y: pl.Series, test_X: pl.DataFrame, feature_cols: list) -> np.ndarray:
    # ---- Pre-ML contract validation ----
    if train_X.height == 0:
        raise RuntimeError('CONTRACT FAIL: train_X has 0 rows')
    if len(train_y) == 0:
        raise RuntimeError('CONTRACT FAIL: train_y has 0 rows')
    if train_X.height != len(train_y):
        raise RuntimeError('CONTRACT FAIL: X/y mismatch (X=%d, y=%d)' % (train_X.height, len(train_y)))
    if test_X.height == 0:
        raise RuntimeError('CONTRACT FAIL: test_X has 0 rows')
    if 'ts_event' not in train_X.columns or 'ts_event' not in test_X.columns:
        raise RuntimeError('CONTRACT FAIL: ts_event missing from X frame')

    feature_cols = remove_constant_features(train_X.select(feature_cols), feature_cols, threshold=1e-08)
    if len(feature_cols) == 0:
        return np.full(len(test_X), 0.5, dtype=np.float32)
    X_train = train_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = stabilize_targets(train_y.to_numpy().astype(np.float32).ravel())
    X_test = test_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    # Validate y after stabilization
    if np.any(np.isnan(y_train)):
        raise RuntimeError('SKLEARN FAIL: NaN in y_train after stabilize_targets')
    if np.any(np.isinf(y_train)):
        raise RuntimeError('SKLEARN FAIL: inf in y_train after stabilize_targets')
    if X_train.shape[0] != len(y_train):
        raise RuntimeError('CONTRACT FAIL: X/y shape mismatch (X=%s, y=%s)' % (X_train.shape, y_train.shape))

    X_train = safe_replace(safe_clip(X_train, -8.0, 8.0))
    X_test = safe_replace(safe_clip(X_test, -8.0, 8.0))
    X_train, X_test = robust_scale(X_train, X_test)
    X_train = safe_clip(X_train, -4.0, 4.0)
    X_test = safe_clip(X_test, -4.0, 4.0)
    if config.MODEL_TYPE == 'Ridge':
        ridge_params = config.RIDGE_PARAMS.copy()
        ridge_params['alpha'] = ridge_params.get('alpha', 1.0)
        model = Ridge(**ridge_params)
        model.fit(X_train, y_train)
        raw_pred = model.predict(X_test)
        raw_pred = raw_pred * float(getattr(config, 'TARGET_SCALE_FACTOR', 100.0))
        raw_pred = safe_clip(raw_pred, -2.0, 2.0)
        probs = expit(raw_pred).astype(np.float32)
    elif config.MODEL_TYPE == 'LogisticRegression':
        y_class = (y_train > 0).astype(np.int8)
        classes = np.unique(y_class)
        if len(classes) < 2:
            prior = float(classes[0]) if len(classes) else 0.5
            logger.warning(
                'LogisticRegression skipped: single-class train fold; '
                'returning empirical prior %.4f',
                prior,
            )
            probs = np.full(len(test_X), prior, dtype=np.float32)
        else:
            model = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                class_weight=None,
                random_state=getattr(config, 'SEED', 42),
            )
            model.fit(X_train, y_class)
            probs = model.predict_proba(X_test)[:, 1].astype(np.float32)
    elif config.MODEL_TYPE == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_split=200, min_samples_leaf=100, max_features=0.2, random_state=config.SEED, n_jobs=1, class_weight='balanced_subsample')
        model.fit(X_train, (y_train > 0).astype(np.int8))
        probs = model.predict_proba(X_test)[:, 1].astype(np.float32)
    else:
        raise ValueError(f'Unknown MODEL_TYPE: {config.MODEL_TYPE}')
    probs = safe_clip(probs, 0.05, 0.95)
    return probs.astype(np.float32)

def smooth_probabilities(probs: np.ndarray, session_ids: np.ndarray, alpha: float=0.3) -> np.ndarray:
    if alpha <= 0:
        return probs.astype(np.float32)
    alpha = min(max(alpha, 0.2), 0.5)
    smoothed = np.zeros_like(probs, dtype=np.float32)
    current = 0.5
    last_session = None
    for i in range(len(probs)):
        p = float(probs[i])
        sess = session_ids[i]
        if sess != last_session:
            current = 0.5
            last_session = sess
        p = min(max(p, 0.1), 0.9)
        current = alpha * p + (1 - alpha) * current
        smoothed[i] = current
    return smoothed.astype(np.float32)

def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    close = pl.col('close').cast(pl.Float32)
    open_ = pl.col('open').cast(pl.Float32)
    close_lagged = close.shift(1)
    sma20 = close_lagged.rolling_mean(window_size=20, min_periods=20)
    signal = pl.when(close_lagged > sma20).then(pl.lit(1.0, dtype=pl.Float32)).otherwise(pl.lit(0.0, dtype=pl.Float32))
    position = signal.shift(1).fill_null(0.0)
    ret_exec = (close - open_) / open_.clip(config.EPS, None)
    pnl = position * ret_exec.fill_nan(0.0).fill_null(0.0)
    return df.select(pnl.cast(pl.Float32).alias('benchmark_pnl')).to_series()

def exclude_warmup(df: pl.DataFrame, burn_in_bars: int) -> pl.DataFrame:
    """Drop the first *burn_in_bars* rows plus one extra bar (position carry-over
    from the warmup period). Returns the trimmed DataFrame unchanged
    if burn_in_bars <= 0 or the DataFrame is shorter than burn_in_bars + 1."""
    if burn_in_bars <= 0 or df.height <= burn_in_bars:
        return df
    trim = min(burn_in_bars + 1, df.height)
    return df.slice(trim)


def _safe_corr(a, b):
    """Safe rank correlation: handles constant arrays, NaN, and short samples."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    n = mask.sum()
    if n < 5:
        return float('nan')
    a_m = a[mask]
    b_m = b[mask]
    sd_a, sd_b = np.std(a_m), np.std(b_m)
    if sd_a < 1e-15 or sd_b < 1e-15:
        return float('nan')
    return float(np.corrcoef(a_m, b_m)[0, 1])


def _hash_col(df, col):
    if col not in df.columns:
        return 'missing'
    vals = df[col].to_numpy()
    mask = np.isfinite(vals)
    if mask.sum() == 0:
        return 'all_nan'
    return hashlib.sha256(vals[mask].tobytes()).hexdigest()[:8]


def _label_horizon_minutes(target_col: str) -> int:
    if target_col in {'target_15m_ret', 'target_15m_dir', 'target_15m_trade_class'}:
        return int(getattr(config, 'TARGET_15M_HORIZON', 15)) * 5
    return int(getattr(config, 'TARGET_15M_HORIZON', 15)) * 5


def _purge_train_tail_for_label_horizon(
    train_df: pl.DataFrame,
    boundary_ts,
    target_col: str,
) -> pl.DataFrame:
    """Drop training rows whose forward label horizon can cross into test."""
    horizon_minutes = _label_horizon_minutes(target_col)
    cutoff = boundary_ts - timedelta(minutes=horizon_minutes)
    before = train_df.height
    purged = train_df.filter(pl.col('ts_event') < cutoff)
    print(
        f'[PURGE] train rows before={before} after={purged.height} '
        f'cutoff={cutoff} horizon={horizon_minutes}m target={target_col}',
        flush=True,
    )
    if purged.height == 0:
        raise RuntimeError(
            f'PURGE FAILURE: removed all train rows for target={target_col}, '
            f'cutoff={cutoff}, horizon={horizon_minutes}m'
        )
    return purged


def _execution_state(df: pl.DataFrame) -> dict:
    active = 0
    trades = 0
    if 'target_exec' in df.columns:
        target_exec = df['target_exec'].to_numpy().astype(np.float64)
        active = int(np.sum(np.abs(target_exec) > 1e-12))
    if 'position' in df.columns:
        pos = df['position'].to_numpy().astype(np.float64)
        trades = int(np.sum(np.abs(np.diff(pos, prepend=pos[0])) > 1e-9)) if len(pos) else 0
    pnl_sum = float(df['pnl'].sum()) if 'pnl' in df.columns else float('nan')
    pnl_cs = _hash_col(df, 'pnl') if 'pnl' in df.columns else 'missing'
    return {'rows': df.height, 'active': active, 'trades': trades, 'pnl_sum': pnl_sum, 'pnl_cs': pnl_cs}


def _log_fold_diagnostics(result: pl.DataFrame, test_original: pl.DataFrame, fold_idx: int, prefix: str = ''):
    """Log prediction distribution, IC, gross/net Sharpe, turnover, trade count."""
    import numpy as np
    if result.height == 0:
        logger.warning('[DIAG] skip fold=%d%s — empty result', fold_idx, prefix)
        return
    probs = result['prediction_prob'].to_numpy() if 'prediction_prob' in result.columns else None
    if probs is None:
        logger.warning('[DIAG] skip fold=%d%s — no prediction_prob', fold_idx, prefix)
        return
    pmean = float(np.mean(probs))
    pstd = float(np.std(probs))
    gt055 = float(np.mean(probs > 0.55))
    lt045 = float(np.mean(probs < 0.45))
    pmin = float(np.min(probs))
    pmax = float(np.max(probs))
    ic = 'missing'
    for tcol in ('target_15m_ret',):
        if tcol in test_original.columns:
            ic_val = _safe_corr(probs, test_original[tcol].to_numpy())
            if not np.isnan(ic_val):
                ic = f'{ic_val:.4f}'
                break
    bar_sqrt = 252 ** 0.5
    gross_sharpe = 'missing'
    net_sharpe = 'missing'
    cost_drag = 'missing'
    if 'gross_pnl' in result.columns:
        gp = result['gross_pnl'].to_numpy().astype(np.float64)
        if gp.std() > 1e-12:
            gross_sharpe = f'{float(gp.mean() / gp.std() * bar_sqrt):.3f}'
    if 'pnl' in result.columns:
        np_ = result['pnl'].to_numpy().astype(np.float64)
        if np_.std() > 1e-12:
            net_sharpe = f'{float(np_.mean() / np_.std() * bar_sqrt):.3f}'
        if 'gross_pnl' in result.columns:
            gp_tot = float(gp.sum())
            np_tot = float(np_.sum())
            cost_drag = f'{np_tot - gp_tot:+.2f}'
    turnover = 'missing'
    if 'pos_change' in result.columns:
        pc = result['pos_change'].to_numpy().astype(np.float64)
        turnover = f'{float(pc.sum()):.1f}'
    trades = 'missing'
    if 'position' in result.columns:
        pos = result['position'].to_numpy().astype(np.float64)
        shifts = np.abs(np.diff(pos.astype(np.float64), prepend=pos[0].astype(np.float64)))
        trades = f'{int(np.sum(shifts > 1e-9))}'
    pred_cs = _hash_col(result, 'prediction_prob')
    pnl_cs = _hash_col(result, 'pnl')
    sig_cs = _hash_col(result, 'raw_signal') if 'raw_signal' in result.columns else _hash_col(result, 'target_exec')
    logger.info(
        '[DIAG] fold=%d%s prob_mean=%.4f prob_std=%.4f gt055=%.3f lt045=%.3f '
        'min=%.3f max=%.3f ic=%s gross_sharpe=%s net_sharpe=%s cost_drag=%s '
        'turnover=%s trades=%s pred_cs=%s pnl_cs=%s sig_cs=%s',
        fold_idx, prefix, pmean, pstd, gt055, lt045, pmin, pmax,
        ic, gross_sharpe, net_sharpe, cost_drag, turnover, trades,
        pred_cs, pnl_cs, sig_cs,
    )


def process_fold(train_X: pl.DataFrame, train_y: pl.Series, test_original: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    import psutil
    if train_X.height == 0 or len(train_y) == 0:
        raise RuntimeError(
            'CONTRACT FAIL: empty fold (train_X=%d, train_y=%d)' %
            (train_X.height, len(train_y))
        )
    if train_X.height != len(train_y):
        raise RuntimeError(
            'CONTRACT FAIL: fold X/y mismatch (X=%d, y=%d)' %
            (train_X.height, len(train_y))
        )
    if test_original.height == 0:
        raise RuntimeError('CONTRACT FAIL: empty test fold')

    rss_bytes = psutil.Process().memory_info().rss
    rss_stop = getattr(config, 'RSS_STOP_BYTES', int(13.5 * 1024**3))
    if rss_bytes > rss_stop:
        raise MemoryError(f'RSS {rss_bytes/(1024**3):.2f} GB exceeds RSS_STOP_BYTES ({rss_stop/(1024**3):.2f} GB) in process_fold')
    probs = train_and_predict(train_X, train_y, test_original, feature_cols)
    if config.PROBABILITY_SMOOTHING_ALPHA > 0:
        session_ids = test_original['session_id'].to_numpy()
        probs = smooth_probabilities(probs, session_ids, alpha=config.PROBABILITY_SMOOTHING_ALPHA)
    result = test_original.with_columns(pl.Series('prediction_prob', probs).cast(pl.Float32))
    result = result.with_columns(compute_benchmark(result))

    # Meta-labeling: compute primary prediction direction, train meta-model
    # on held-out portion of training data, gate target_exec by meta-prob.
    enable_meta = getattr(config, 'ENABLE_META_LABELING', False)
    meta_model = None
    if enable_meta:
        pred_dir = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)).astype(np.int8)
        unique_preds = np.unique(pred_dir).size
        if unique_preds < 2:
            # Meta-target collapses to primary target when primary has
            # zero directional variation. Meta-model adds only noise.
            logger.info('Meta skipped: primary model predicts single direction')
            enable_meta = False
    if enable_meta:
        result = result.with_columns(pl.Series('primary_prediction', pred_dir))
        result = add_meta_label_target(result, 'primary_prediction')
        meta_threshold = getattr(config, 'META_THRESHOLD', 0.5)

        if train_X.height >= 40:
            split = max(train_X.height // 3, 20)
            if split < 200:
                logger.info(
                    'Meta-model skipped: insufficient holdout (%d rows, need >= 200)',
                    split,
                )
                enable_meta = False
            else:
                train_primary_X = train_X.slice(0, split)
                train_primary_y = train_y.slice(0, split)
                train_val_X = train_X.slice(split)
                train_val_y = train_y.slice(split)
                probs_val = train_and_predict(train_primary_X, train_primary_y, train_val_X, feature_cols)
                meta_val_pred = np.where(probs_val > 0.55, 1, np.where(probs_val < 0.45, -1, 0)).astype(np.int8)
                meta_val_df = pl.DataFrame({'primary_prediction': meta_val_pred})
                meta_val_actual = np.where(
                    train_val_y.to_numpy().astype(np.float32).ravel() > 0, 1, -1
                ).astype(np.int8)
                meta_val_df = meta_val_df.with_columns(pl.Series('target_tb', meta_val_actual))
                meta_val_df = add_meta_label_target(meta_val_df, 'primary_prediction')
                meta_train_y = meta_val_df['target_meta'].to_numpy().astype(np.float32)
                meta_train_X = train_val_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
                meta_model = train_meta_model(meta_train_X, meta_val_pred.astype(np.float32), meta_train_y)

    result = run_execution_simulation(result)

    if meta_model is not None:
        test_X_np = test_original.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
        result = apply_meta_gate(result, meta_model, test_X_np, meta_threshold=meta_threshold)
        result = _recompute_pnl_after_gate(result)

    _log_fold_diagnostics(result, test_original, fold_idx=-1)
    return exclude_warmup(result, getattr(config, 'BURN_IN_BARS', 500))

# ============================================================================
# HMM Regime-Aware Walkforward
# ============================================================================

def _resample_to_1h(df_5min: pl.DataFrame) -> pl.DataFrame:
    """
    Resample 5-minute data to 1-hour frequency for HMM detection layer.
    Preserves session_id grouping and uses only OHLCV columns.
    """
    from pipeline.session.session import session_id_expr

    df = df_5min.select(['ts_event', 'open', 'high', 'low', 'close', 'volume'])
    df = df.with_columns(
        pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).alias('ts_local')
    )
    df = df.with_columns(
        pl.col('ts_local').dt.truncate('1h').alias('ts_hour')
    )
    df = df.with_columns(session_id_expr('ts_local').alias('session_id'))

    agg = df.group_by(['session_id', 'ts_hour'], maintain_order=True).agg([
        pl.col('open').first().alias('open'),
        pl.col('high').max().alias('high'),
        pl.col('low').min().alias('low'),
        pl.col('close').last().alias('close'),
        pl.col('volume').sum().alias('volume'),
        pl.len().alias('n_ticks'),
    ])
    # Filter incomplete hours (< 10 ticks)
    agg = agg.filter(pl.col('n_ticks') >= 10)
    agg = agg.rename({'ts_hour': 'ts_event'})
    agg = agg.drop('n_ticks')
    agg = agg.with_columns(
        pl.col('ts_event').dt.convert_time_zone('UTC').alias('ts_event')
    )
    agg = agg.with_columns([
        pl.col('open').cast(pl.Float32),
        pl.col('high').cast(pl.Float32),
        pl.col('low').cast(pl.Float32),
        pl.col('close').cast(pl.Float32),
    ])
    agg = agg.sort(['session_id', 'ts_event'])
    return agg


def process_fold_with_hmm(
    train_X: pl.DataFrame,
    train_y: pl.Series,
    test_original: pl.DataFrame,
    feature_cols: list,
    hmm_filter: Optional["HMMRegimeFilter"] = None,
    df_1h_test: Optional[pl.DataFrame] = None,
    fold_idx: int = 0,
    hmm_retrain_interval: int = 5,
) -> Tuple[pl.DataFrame, Optional["HMMRegimeFilter"]]:
    """
    Process a single walkforward fold with HMM regime gating.

    Steps:
      1. Train ML model and generate base predictions.
      2. Execute base strategy (computation is side-effect-free PnL).
      3. Apply HMM regime filter if active (gate trades by regime).
      4. Recompute PnL after gating.

    Args:
        train_X, train_y, test_original, feature_cols: Standard fold data.
        hmm_filter: Existing HMMRegimeFilter instance (None on first call).
        df_1h_test: 1H resampled test data for this fold.
        fold_idx: Zero-based fold index (for retrain scheduling).
        hmm_retrain_interval: Retrain HMM every N folds.

    Returns:
        (result_df, updated_hmm_filter): The executed DataFrame and filter.
    """
    _hmmf = importlib.import_module("pipeline.regime.hmm_filter")
    HMMRegimeFilter = _hmmf.HMMRegimeFilter
    apply_hmm_filter = _hmmf.apply_hmm_filter
    # --- Base execution (no HMM) ---
    probs = train_and_predict(train_X, train_y, test_original, feature_cols)
    if config.PROBABILITY_SMOOTHING_ALPHA > 0:
        session_ids = test_original['session_id'].to_numpy()
        probs = smooth_probabilities(probs, session_ids, alpha=config.PROBABILITY_SMOOTHING_ALPHA)

    result = test_original.with_columns(
        pl.Series('prediction_prob', probs).cast(pl.Float32)
    )
    result = result.with_columns(compute_benchmark(result))
    result = run_execution_simulation(result)

    # --- HMM Regime Gating ---
    if hmm_filter is None:
        hmm_filter = HMMRegimeFilter()

    if df_1h_test is not None and df_1h_test.height > 0:
        should_retrain = (fold_idx > 0 and fold_idx % hmm_retrain_interval == 0)

        # Build 1H train data from train_X if retraining
        df_1h_train = None
        if should_retrain:
            df_1h_train = _resample_to_1h(train_X)

        result, hmm_filter = apply_hmm_filter(
            df_5min_base=result,
            df_1h_train=df_1h_train,
            df_1h_test=df_1h_test,
            hmm_filter=hmm_filter,
            retrain=should_retrain,
        )

        # Recompute PnL after regime gating (target_exec may be zeroed).
        # Uses the full execution pipeline identical to simulate_execution_classification:
        # intrabar stops, contract multiplier, position clipping, round-turn settlement,
        # and proportional PnL clip — so the HMM PnL is directly comparable to the base PnL.
        result = _recompute_pnl_after_gate(result)

    _log_fold_diagnostics(result, test_original, fold_idx=fold_idx, prefix='_hmm')
    result = exclude_warmup(result, getattr(config, 'BURN_IN_BARS', 500))
    return result, hmm_filter


def _recompute_pnl_after_gate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Recompute position, intrabar stops, and PnL after target_exec has been
    gated by HMM.

    Uses the full execution pipeline (_compute_pnl_from_target_exec) so the
    recomputed PnL is identical to the main simulation path, including:
      - intrabar stop-loss / take-profit with gap-slippage logic
      - contract multiplier in PnL
      - position clipping (max_position_size + notional cap)
      - round-turn settlement on flatting
      - proportional PnL clip (5 % of notional)

    Preserves all HMM columns (hmm_regime_*, hmm_trade_gate).
    """
    from pipeline.common.market import get_contract_multiplier
    _compute_pnl_from_target_exec = importlib.import_module(
        "pipeline.execution.simulator"
    )._compute_pnl_from_target_exec

    symbol = getattr(config, 'CURRENT_SYMBOL', None)
    if not symbol:
        raise RuntimeError(
            'CONTRACT FAIL: CURRENT_SYMBOL is unset. '
            'Cannot resolve contract multiplier for HMM PnL recompute. '
            'Ensure cli.py sets config.CURRENT_SYMBOL before calling run-hmm.'
        )
    contract_multiplier = get_contract_multiplier(symbol)

    # Preserve HMM columns so they survive the recompute
    hmm_cols = [c for c in df.columns if c.startswith('hmm_')]
    hmm_data = {c: df[c].clone() for c in hmm_cols}

    # Drop columns that _compute_pnl_from_target_exec will recompute so we
    # get a clean replacement without column-name conflicts.
    recompute_cols = ['ret_exec', 'position', 'pos_change', 'intrabar_pnl', 'gross_pnl', 'pnl']
    df_clean = df.drop([c for c in recompute_cols if c in df.columns])

    # Re-run full PnL pipeline against the HMM-gated target_exec
    df_result = _compute_pnl_from_target_exec(df_clean, contract_multiplier)

    # Restore HMM columns
    for col, series in hmm_data.items():
        df_result = df_result.with_columns(series.alias(col))

    return df_result


def _build_bar_folds(df: pl.DataFrame) -> list:
    """
    Build walkforward fold index ranges from bar count (not calendar days).

    Bar-based folding avoids calendar-day assumptions that break on intraday
    futures data with session gaps, weekends, and variable trading hours.

    Fold sizing:
        train = max(MIN_TRAIN_BARS, int(bars * TRAIN_FRACTION))
        test  = max(MIN_TEST_BARS,  int(bars * TEST_FRACTION))
        step  = test

    Returns:
        list of (i0, i1, i2, i3) index tuples:
            train = df[i0:i1], test = df[i2:i3]
            train < test strictly in time (i1 <= i2).
    """
    MIN_TRAIN_BARS = 1000
    MIN_TEST_BARS = 200
    TRAIN_FRACTION = 0.6
    TEST_FRACTION = 0.2

    bars = df.height
    if bars < MIN_TRAIN_BARS + MIN_TEST_BARS:
        raise RuntimeError(
            'BACKTEST FAILURE: %d bars insufficient for walkforward '
            '(need at least %d train + %d test)' %
            (bars, MIN_TRAIN_BARS, MIN_TEST_BARS)
        )

    train_size = max(MIN_TRAIN_BARS, int(bars * TRAIN_FRACTION))
    test_size = max(MIN_TEST_BARS, int(bars * TEST_FRACTION))
    step_size = test_size
    window_size = train_size + test_size

    folds = []
    start = 0
    while start + window_size <= bars:
        folds.append((start, start + train_size, start + train_size, start + window_size))
        start += step_size

    if len(folds) < 2:
        raise RuntimeError(
            'BACKTEST FAILURE: only %d folds from %d bars '
            '(train=%d, test=%d, step=%d) -- need at least 2' %
            (len(folds), bars, train_size, test_size, step_size)
        )

    logger.info(
        'Bar-based folds: %d folds from %d bars (train=%d, test=%d, step=%d)',
        len(folds), bars, train_size, test_size, step_size,
    )
    return folds


def run_walkforward_with_hmm(
    X: pl.DataFrame,
    y: pl.DataFrame,
    feature_cols: list,
    target_col: str = 'target_15m_ret',
    hmm_retrain_interval: int = 5,
) -> Tuple[pl.DataFrame, dict]:
    """
    Walkforward with HMM regime-aware risk management.

    Same interface as run_walkforward(), but additionally:
      - Resamples 5-min data to 1H internally for the detection layer.
      - Trains HMM periodically (every N folds).
      - Bridges regime probabilities to 5-min execution bars.
      - Gates trades based on regime (allowed/prohibited).
      - Returns validation report comparing base vs HMM-filtered PnL.

    Args:
        X: Feature DataFrame (5-min).
        y: Target DataFrame (5-min).
        feature_cols: Feature column names.
        target_col: Target column name.
        hmm_retrain_interval: Retrain HMM every N folds.

    Returns:
        (df_hmm_result, validation_dict):
            df_hmm_result: Full executed DataFrame with regime columns.
            validation_dict: Dictionary with base_metrics, hmm_metrics, psr, etc.
    """
    HMMRegimeFilter = importlib.import_module(
        "pipeline.regime.hmm_filter"
    ).HMMRegimeFilter
    compare_strategies = importlib.import_module(
        "pipeline.walkforward.validation"
    ).compare_strategies

    df = X.with_columns(y)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Empty-input guard -- no data to walk forward over
    if df.height == 0:
        logger.warning('Empty input DataFrame for HMM walkforward -- returning empty result.')
        return pl.DataFrame(), {}

    df = df.sort('ts_event')
    fold_indices = _build_bar_folds(df)
    if not fold_indices:
        logger.warning('No folds produced for HMM -- returning empty result.')
        return pl.DataFrame(), {}
    test_bars_per_fold = [i3 - i2 for _, _, i2, i3 in fold_indices]
    print(f'[WALKFORWARD-HMM] {len(fold_indices)} folds, test_bars_per_fold={test_bars_per_fold}, total_input={df.height} rows', flush=True)

    # Correlation pruning on first training window
    first_train_df = df.slice(fold_indices[0][0], fold_indices[0][1] - fold_indices[0][0])
    pruned_features = (
        correlation_prune(first_train_df, feature_cols, threshold=min(config.CORR_THRESHOLD, 0.9))
        if first_train_df.height > 0
        else feature_cols
    )

    # Build folds from index ranges (zero-copy slicing)
    folds = []
    for i0, i1, i2, i3 in fold_indices:
        train_X = df.slice(i0, i1 - i0).drop([target_col])
        train_y = df.slice(i0, i1 - i0)[target_col]
        test_original = df.slice(i2, i3 - i2).drop([target_col])
        folds.append((train_X, train_y, test_original, pruned_features))

    # Initialize HMM filter
    hmm_filter = HMMRegimeFilter()

    # Try to initialize HMM on the first training window
    first_train = folds[0][0]
    df_1h_init = _resample_to_1h(first_train)
    if df_1h_init.height >= 60:  # at least 60 hours (~2.5 days)
        success = hmm_filter.initialize(df_1h_init)
        logger.info(
            f"HMM initialization: {'success' if success else 'fallback'} "
            f"({df_1h_init.height} 1H bars)"
        )

    results_base = []
    results_hmm = []

    for fold_idx, (train_X, train_y, test_original, feat_cols) in enumerate(
        tqdm(folds, desc='Walkforward + HMM', unit='fold')
    ):
        # Standard base execution
        base_result = process_fold(train_X, train_y, test_original, feat_cols)
        results_base.append(base_result)

        # HMM-aware execution
        df_1h_test = _resample_to_1h(test_original)
        hmm_result, hmm_filter = process_fold_with_hmm(
            train_X=train_X,
            train_y=train_y,
            test_original=test_original,
            feature_cols=feat_cols,
            hmm_filter=hmm_filter,
            df_1h_test=df_1h_test,
            fold_idx=fold_idx,
            hmm_retrain_interval=hmm_retrain_interval,
        )
        results_hmm.append(hmm_result)

    # Concatenate results
    df_base = pl.concat(results_base).sort(['session_id', 'ts_event'])
    df_hmm = pl.concat(results_hmm).sort(['session_id', 'ts_event'])
    print(f'[WALKFORWARD-HMM] {len(results_hmm)} folds concatenated: {df_hmm.height} rows before warmup', flush=True)

    # Validation
    report = compare_strategies(
        df_base=df_base,
        df_hmm=df_hmm,
        pnl_column='pnl',
        position_column='position',
        confidence=0.95,
        hmm_active=hmm_filter.is_active,
        fallback_reason=hmm_filter.fallback_reason,
    )

    logger.info(f"\n{report.summary()}")

    # Build validation dict for serialization
    validation_dict = report.to_dict()
    validation_dict['hmm_training_log'] = hmm_filter.training_log
    validation_dict['n_folds'] = len(folds)

    df_hmm = exclude_warmup(df_hmm, getattr(config, 'BURN_IN_BARS', 500))
    print(f'[WALKFORWARD-HMM] After exclude_warmup({getattr(config, "BURN_IN_BARS", 500)}): {df_hmm.height} rows, ts_min={df_hmm["ts_event"].min()} ts_max={df_hmm["ts_event"].max()}', flush=True)
    return df_hmm, validation_dict


def run_walkforward(X: pl.DataFrame, y: pl.DataFrame, feature_cols: list, target_col: str='target_15m_ret') -> pl.DataFrame:
    df = X.with_columns(y)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Empty-input guard -- no data to walk forward over
    if df.height == 0:
        raise RuntimeError(
            'BACKTEST FAILURE: 0 rows after feature generation (target=%s)' % target_col
        )

    df = df.sort('ts_event')
    logger.info('[ML-CONTRACT] X=%d rows, y=%d rows, features=%d', df.height, df.height, len(feature_cols))
    logger.info('[ML-CONTRACT] ts_event range: %s -> %s',
                df['ts_event'].min(), df['ts_event'].max())
    fold_indices = _build_bar_folds(df)
    if not fold_indices:
        logger.warning('No folds produced -- returning empty result.')
        return pl.DataFrame()
    test_bars_per_fold = [i3 - i2 for _, _, i2, i3 in fold_indices]
    print(f'[WALKFORWARD] {len(fold_indices)} folds, test_bars_per_fold={test_bars_per_fold}, total_input={df.height} rows, pruned_features={len(feature_cols)}', flush=True)

    # Correlation pruning on first training window
    first_train_df = df.slice(fold_indices[0][0], fold_indices[0][1] - fold_indices[0][0])
    pruned_features = (
        correlation_prune(first_train_df, feature_cols, threshold=min(config.CORR_THRESHOLD, 0.9))
        if first_train_df.height > 0
        else feature_cols
    )

    # Build folds from index ranges (zero-copy slicing)
    folds = []
    for i0, i1, i2, i3 in fold_indices:
        train_X = df.slice(i0, i1 - i0).drop([target_col])
        train_y = df.slice(i0, i1 - i0)[target_col]
        test_original = df.slice(i2, i3 - i2).drop([target_col])
        folds.append((train_X, train_y, test_original, pruned_features))
    if config.WF_PARALLEL_FOLDS == 1:
        results = []
        for train_X, train_y, test_original, feat_cols in tqdm(folds, desc='Walkforward folds', unit='fold'):
            results.append(process_fold(train_X, train_y, test_original, feat_cols))
    else:
        results = Parallel(n_jobs=config.WF_PARALLEL_FOLDS, backend='loky')((delayed(process_fold)(train_X, train_y, test_original, feat_cols) for train_X, train_y, test_original, feat_cols in folds))
    final = pl.concat(results)
    final = final.sort(['session_id', 'ts_event'])
    print(f'[WALKFORWARD] {len(results)} folds concatenated: {final.height} rows before warmup', flush=True)
    final = exclude_warmup(final, getattr(config, 'BURN_IN_BARS', 500))
    print(f'[WALKFORWARD] After exclude_warmup({getattr(config, "BURN_IN_BARS", 500)}): {final.height} rows, ts_min={final["ts_event"].min()} ts_max={final["ts_event"].max()}', flush=True)
    return final


# ============================================================================
# True Outer-Split Evaluation Mode
# ============================================================================
#
# Differs from bar-based inner-walkforward:
#   1. Receives separate train_df and test_df already sliced by run.py via
#      --train-start/--train-end and --start/--end in cli.py.
#   2. Trains a single model on train_df, predicts on all of test_df in one pass.
#   3. No inner folds, no 60/40 splitting of the test window.
#   4. Minimal warmup only for position carry-over.
#
# This ensures the 180d training window is fully used and the 30d test
# window is fully evaluated.

_OUTER_BURN_IN = 50  # minimal warmup for position carry-over only


def _print_alpha_diagnostics(result: pl.DataFrame, test_df: pl.DataFrame, train_df: pl.DataFrame,
                              target_col: str, feature_cols: list, pred_cs: str, pnl_cs: str) -> None:
    y_train_vals = train_df[target_col].to_numpy().astype(np.float64) if target_col in train_df.columns else np.array([])
    y_test_vals = test_df[target_col].to_numpy().astype(np.float64) if target_col in test_df.columns else np.array([])
    y_eval_vals = result[target_col].to_numpy().astype(np.float64) if target_col in result.columns else np.array([])
    probs = result['prediction_prob'].to_numpy().astype(np.float64) if 'prediction_prob' in result.columns else np.array([])
    pos = result['position'].to_numpy().astype(np.float64) if 'position' in result.columns else np.array([])
    pnl_arr = result['pnl'].to_numpy().astype(np.float64) if 'pnl' in result.columns else np.array([])
    gross = result['gross_pnl'].to_numpy().astype(np.float64) if 'gross_pnl' in result.columns else pnl_arr
    bar_sqrt = 252 ** 0.5

    gross_sharpe = float(gross.mean() / max(gross.std(), 1e-12) * bar_sqrt) if len(gross) > 0 else 0.0
    net_sharpe = float(pnl_arr.mean() / max(pnl_arr.std(), 1e-12) * bar_sqrt) if len(pnl_arr) > 0 else 0.0
    costs = float(gross.sum() - pnl_arr.sum()) if len(gross) > 0 else 0.0
    ic = (
        float(np.corrcoef(probs, y_eval_vals)[0, 1])
        if len(probs) > 5 and len(y_eval_vals) == len(probs)
        and np.std(probs) > 1e-12 and np.std(y_eval_vals) > 1e-12
        else 0.0
    )
    turnover = (
        float(np.abs(np.diff(pos.astype(np.float64), prepend=pos[0])).sum() / len(pos))
        if len(pos) > 0 else 0.0
    )

    y_train_dist = dict(zip(*np.unique(y_train_vals, return_counts=True))) if len(y_train_vals) > 0 else {}
    y_test_dist = dict(zip(*np.unique(y_test_vals, return_counts=True))) if len(y_test_vals) > 0 else {}
    pred_bins = {f'p{i}': int(np.sum((probs >= i/3) & (probs < (i+1)/3))) for i in range(3)} if len(probs) > 0 else {}
    pos_dist = dict(zip(*np.unique(pos, return_counts=True))) if len(pos) > 0 else {}
    raw_signal_counts = {}
    if 'raw_signal' in result.columns:
        raw_signal_vals = result['raw_signal'].to_numpy().astype(np.float64)
        raw_signal_counts = {
            '-1': int(np.sum(raw_signal_vals < 0.0)),
            '0': int(np.sum(raw_signal_vals == 0.0)),
            '+1': int(np.sum(raw_signal_vals > 0.0)),
        }
    prob_min = float(np.min(probs)) if len(probs) > 0 else float('nan')
    prob_max = float(np.max(probs)) if len(probs) > 0 else float('nan')
    prob_mean = float(np.mean(probs)) if len(probs) > 0 else float('nan')
    prob_std = float(np.std(probs)) if len(probs) > 0 else float('nan')
    prob_gt055 = float(np.mean(probs > 0.55)) if len(probs) > 0 else float('nan')
    prob_lt045 = float(np.mean(probs < 0.45)) if len(probs) > 0 else float('nan')
    auc = 'NA'
    brier = 'NA'
    if len(probs) > 0 and len(y_eval_vals) == len(probs):
        y_binary = (y_eval_vals > 0).astype(np.int8)
        brier = f'{float(brier_score_loss(y_binary, probs)):.6f}'
        if len(np.unique(y_binary)) == 2:
            auc = f'{float(roc_auc_score(y_binary, probs)):.4f}'
    y_train_mean = float(np.mean(y_train_vals)) if len(y_train_vals) > 0 else float('nan')
    y_test_mean = float(np.mean(y_test_vals)) if len(y_test_vals) > 0 else float('nan')
    print(
        f'[ALPHA-DIAG] y_train={y_train_dist} y_test={y_test_dist} pred={pred_bins} '
        f'y_train_mean={y_train_mean:.4f} y_test_mean={y_test_mean:.4f} '
        f'prob_min={prob_min:.4f} prob_max={prob_max:.4f} '
        f'prob_mean={prob_mean:.4f} prob_std={prob_std:.4f} '
        f'gt055={prob_gt055:.3f} lt045={prob_lt045:.3f} '
        f'auc={auc} brier={brier} raw_signal={raw_signal_counts} '
        f'pos={pos_dist} gross_sharpe={gross_sharpe:.3f} net_sharpe={net_sharpe:.3f} '
        f'costs={costs:.2f} ic={ic:.4f} turnover={turnover:.2f} features={len(feature_cols)}',
        flush=True,
    )

    # Inverted signal check
    prob_mean = probs.mean() if len(probs) > 0 else 0.5
    inv_probs = 1.0 - probs
    inv_pred_dir = np.where(inv_probs > 0.55, 1, np.where(inv_probs < 0.45, -1, 0)).astype(np.float64)
    inv_pos = np.where(inv_pred_dir != 0, np.sign(inv_pred_dir), 0).astype(np.float64)
    ret_exec = result['ret_exec'].to_numpy().astype(np.float64) if 'ret_exec' in result.columns else np.zeros_like(pos)
    inv_pnl_arr = inv_pos * ret_exec
    inv_sharpe = float(inv_pnl_arr.mean() / max(inv_pnl_arr.std(), 1e-12) * bar_sqrt) if len(inv_pnl_arr) > 0 else 0.0
    inv_pnl = float(inv_pnl_arr.sum())
    orig_pnl = float(pnl_arr.sum())
    print(
        f'[INVERT-CHECK] original_sharpe={net_sharpe:.3f} inverted_sharpe={inv_sharpe:.3f} '
        f'original_pnl={orig_pnl:.2f} inverted_pnl={inv_pnl:.2f} prob_mean={prob_mean:.4f}',
        flush=True,
    )

    # Position hash
    pos_cs = _hash_col(result, 'position')
    print(f'[SPLIT-VERIFY] pred_cs={pred_cs} pos_cs={pos_cs} pnl_cs={pnl_cs}', flush=True)


def validate_walkforward_train_test(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list,
    target_col: str,
) -> dict:
    """Step 7 hard gate: train/test separation and model input validity."""
    if train_df.height == 0:
        raise RuntimeError('WALKFORWARD FAIL: empty train_df')
    if test_df.height == 0:
        raise RuntimeError('WALKFORWARD FAIL: empty test_df')
    if not feature_cols:
        raise RuntimeError('WALKFORWARD FAIL: no feature columns')
    for name, df in (('train', train_df), ('test', test_df)):
        if 'ts_event' not in df.columns:
            raise RuntimeError(f'WALKFORWARD FAIL: {name} missing ts_event')
        if not df['ts_event'].is_sorted():
            raise RuntimeError(f'WALKFORWARD FAIL: {name} ts_event not sorted')
        if df['ts_event'].n_unique() != df.height:
            raise RuntimeError(f'WALKFORWARD FAIL: {name} duplicate ts_event values')
        if target_col not in df.columns:
            raise RuntimeError(f'WALKFORWARD FAIL: {name} missing target {target_col}')
        if df[target_col].null_count() > 0:
            raise RuntimeError(f'WALKFORWARD FAIL: {name} null target values in {target_col}')
    missing_train = [c for c in feature_cols if c not in train_df.columns]
    missing_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_train or missing_test:
        raise RuntimeError(
            f'WALKFORWARD FAIL: missing features train={missing_train[:10]} test={missing_test[:10]}'
        )
    numeric_types = (
        pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )
    non_numeric = [
        c for c in feature_cols
        if train_df[c].dtype not in numeric_types or test_df[c].dtype not in numeric_types
    ]
    if non_numeric:
        raise RuntimeError(f'WALKFORWARD FAIL: non-numeric features {non_numeric[:10]}')

    train_ts_min = train_df['ts_event'].min()
    train_ts_max = train_df['ts_event'].max()
    test_ts_min = test_df['ts_event'].min()
    test_ts_max = test_df['ts_event'].max()
    if train_ts_max >= test_ts_min:
        raise RuntimeError(
            f'WALKFORWARD FAIL: train/test overlap train_max={train_ts_max} test_min={test_ts_min}'
        )
    horizon_minutes = _label_horizon_minutes(target_col)
    cutoff = test_ts_min - timedelta(minutes=horizon_minutes)
    purge_survivors = train_df.filter(pl.col('ts_event') <= cutoff).height
    if purge_survivors == 0:
        raise RuntimeError(
            f'WALKFORWARD FAIL: purge would remove all train rows target={target_col} cutoff={cutoff}'
        )
    return {
        'train_rows': train_df.height,
        'test_rows': test_df.height,
        'purged_train_rows': purge_survivors,
        'feature_cols': len(feature_cols),
        'target_col': target_col,
        'train_ts_min': str(train_ts_min),
        'train_ts_max': str(train_ts_max),
        'test_ts_min': str(test_ts_min),
        'test_ts_max': str(test_ts_max),
    }


def validate_walkforward_oos_predictions(result: pl.DataFrame, test_df: pl.DataFrame) -> None:
    if result.height == 0:
        raise RuntimeError('WALKFORWARD FAIL: empty result')
    if 'prediction_prob' not in result.columns:
        raise RuntimeError('WALKFORWARD FAIL: prediction_prob missing from result')
    probs = result['prediction_prob'].to_numpy().astype(np.float64)
    if not np.isfinite(probs).all():
        raise RuntimeError('WALKFORWARD FAIL: non-finite prediction_prob')
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise RuntimeError('WALKFORWARD FAIL: prediction_prob outside [0, 1]')
    result_min = result['ts_event'].min()
    result_max = result['ts_event'].max()
    test_min = test_df['ts_event'].min()
    test_max = test_df['ts_event'].max()
    if result_min < test_min or result_max > test_max:
        raise RuntimeError(
            f'WALKFORWARD FAIL: result timestamps outside test window '
            f'result=[{result_min}, {result_max}] test=[{test_min}, {test_max}]'
        )


def build_oos_prediction_frame(
    result: pl.DataFrame,
    target_col: str | None = None,
) -> pl.DataFrame:
    """
    Step 8 artifact boundary: OOS predictions and raw model-derived signal.

    This is a projection of the walkforward result, not a recomputation.
    """
    if 'ts_event' not in result.columns:
        raise RuntimeError('OOS PREDICTION FAIL: ts_event missing')
    if 'prediction_prob' not in result.columns:
        raise RuntimeError('OOS PREDICTION FAIL: prediction_prob missing')

    keep = ['ts_event', 'prediction_prob']
    for col in ('raw_signal', target_col, 'target_exec'):
        if col and col in result.columns and col not in keep:
            keep.append(col)
    out = result.select(keep)
    validate_oos_prediction_frame(out)
    return out


def validate_oos_prediction_frame(df: pl.DataFrame) -> None:
    if df.height == 0:
        raise RuntimeError('OOS PREDICTION FAIL: empty prediction frame')
    for col in ('ts_event', 'prediction_prob'):
        if col not in df.columns:
            raise RuntimeError(f'OOS PREDICTION FAIL: missing {col}')
    if df['ts_event'].null_count() > 0:
        raise RuntimeError('OOS PREDICTION FAIL: null ts_event values')
    if not df['ts_event'].is_sorted():
        raise RuntimeError('OOS PREDICTION FAIL: ts_event not sorted')
    if df['ts_event'].n_unique() != df.height:
        raise RuntimeError('OOS PREDICTION FAIL: duplicate ts_event values')
    probs = df['prediction_prob'].to_numpy().astype(np.float64)
    if not np.isfinite(probs).all():
        raise RuntimeError('OOS PREDICTION FAIL: non-finite prediction_prob')
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise RuntimeError('OOS PREDICTION FAIL: prediction_prob outside [0, 1]')
    if 'raw_signal' in df.columns:
        sig = df['raw_signal'].to_numpy().astype(np.float64)
        if not np.isfinite(sig).all():
            raise RuntimeError('OOS PREDICTION FAIL: non-finite raw_signal')
        if not np.isin(sig, [-1.0, 0.0, 1.0]).all():
            raise RuntimeError('OOS PREDICTION FAIL: raw_signal outside {-1,0,1}')


def run_walkforward_modeling(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list,
    target_col: str = 'target_15m_ret',
) -> pl.DataFrame:
    """Step 7 boundary: train on train window and produce OOS test predictions."""
    validation = validate_walkforward_train_test(train_df, test_df, feature_cols, target_col)
    print(
        f'[WALKFORWARD] train_rows={validation["train_rows"]} '
        f'purged_train_rows={validation["purged_train_rows"]} '
        f'test_rows={validation["test_rows"]} features={validation["feature_cols"]}',
        flush=True,
    )
    result = run_outer_train_test_eval(train_df, test_df, feature_cols, target_col)
    validate_walkforward_oos_predictions(result, test_df)
    return result


def run_walkforward_modeling_with_hmm(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list,
    target_col: str = 'target_15m_ret',
) -> Tuple[pl.DataFrame, dict]:
    validation = validate_walkforward_train_test(train_df, test_df, feature_cols, target_col)
    print(
        f'[WALKFORWARD-HMM] train_rows={validation["train_rows"]} '
        f'purged_train_rows={validation["purged_train_rows"]} '
        f'test_rows={validation["test_rows"]} features={validation["feature_cols"]}',
        flush=True,
    )
    result, report = run_outer_train_test_eval_with_hmm(train_df, test_df, feature_cols, target_col)
    validate_walkforward_oos_predictions(result, test_df)
    report = dict(report)
    report['walkforward_validation'] = validation
    return result, report


def run_outer_train_test_eval(train_df: pl.DataFrame, test_df: pl.DataFrame,
                              feature_cols: list, target_col: str = 'target_15m_ret') -> pl.DataFrame:
    if train_df.height == 0:
        raise RuntimeError('OUTER-TRUE FAILURE: empty train_df')
    if test_df.height == 0:
        raise RuntimeError('OUTER-TRUE FAILURE: empty test_df')
    train_ts_min = train_df['ts_event'].min()
    train_ts_max = train_df['ts_event'].max()
    test_ts_min = test_df['ts_event'].min()
    test_ts_max = test_df['ts_event'].max()
    assert train_ts_max < test_ts_min, (
        f'TRAIN/TEST OVERLAP: train max={train_ts_max} >= test min={test_ts_min}'
    )
    train_df = _purge_train_tail_for_label_horizon(train_df, test_ts_min, target_col)
    train_ts_min = train_df['ts_event'].min()
    train_ts_max = train_df['ts_event'].max()
    print(f'[OUTER-TRUE] train_rows={train_df.height} train_ts=[{train_ts_min}, {train_ts_max})', flush=True)
    print(f'[OUTER-TRUE] test_rows={test_df.height} test_ts=[{test_ts_min}, {test_ts_max})', flush=True)
    print(f'[OUTER-TRUE] feature_cols={len(feature_cols)}', flush=True)
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' missing from train/test")
    train_y = train_df[target_col]
    print('[HEARTBEAT] model fit+predict start', flush=True)
    t_model = time.perf_counter()
    probs = train_and_predict(train_df, train_y, test_df, feature_cols)
    dt_model = time.perf_counter() - t_model
    print(f'[HEARTBEAT] model fit+predict done seconds={dt_model:.1f}', flush=True)
    if dt_model > 60:
        print(f'[SLOW] stage=model_fit_predict seconds={dt_model:.1f}', flush=True)
    pred_cs = hashlib.sha256(probs.tobytes()).hexdigest()[:8]
    result = test_df.with_columns(pl.Series('prediction_prob', probs).cast(pl.Float32))
    result = result.with_columns(compute_benchmark(result))
    print('[HEARTBEAT] execution simulation start', flush=True)
    t_exec = time.perf_counter()
    result = run_execution_simulation(result)
    dt_exec = time.perf_counter() - t_exec
    print(f'[HEARTBEAT] execution simulation done rows={result.height} seconds={dt_exec:.1f}', flush=True)
    if dt_exec > 60:
        print(f'[SLOW] stage=execution_simulation seconds={dt_exec:.1f}', flush=True)
    if _OUTER_BURN_IN > 0:
        result = exclude_warmup(result, _OUTER_BURN_IN)
    output_frac = result.height / max(test_df.height, 1)
    assert output_frac >= 0.90, (
        f'OUTER-TRUE ROW COLLAPSE: output={result.height} test={test_df.height} '
        f'fraction={output_frac:.2%} < 90%'
    )
    ts_out_min = result['ts_event'].min()
    ts_out_max = result['ts_event'].max()
    pnl_cs = _hash_col(result, 'pnl') if 'pnl' in result.columns else 'missing'
    print(f'[OUTER-TRUE] pred_cs={pred_cs} pnl_cs={pnl_cs}', flush=True)
    print(f'[OUTER-TRUE] result_rows={result.height} fraction={output_frac:.1%} '
          f'result_ts=[{ts_out_min}, {ts_out_max})', flush=True)
    _print_alpha_diagnostics(result, test_df, train_df, target_col, feature_cols, pred_cs, pnl_cs)
    return result


def run_outer_train_test_eval_with_hmm(train_df: pl.DataFrame, test_df: pl.DataFrame,
                                       feature_cols: list, target_col: str = 'target_15m_ret'
                                       ) -> Tuple[pl.DataFrame, dict]:
    if TYPE_CHECKING:
        from pipeline.regime.hmm_filter import HMMRegimeFilter, apply_hmm_filter
    else:
        from pipeline.regime.hmm_filter import HMMRegimeFilter, apply_hmm_filter
    if train_df.height == 0:
        raise RuntimeError('OUTER-TRUE HMM FAILURE: empty train_df')
    if test_df.height == 0:
        raise RuntimeError('OUTER-TRUE HMM FAILURE: empty test_df')
    train_ts_min = train_df['ts_event'].min()
    train_ts_max = train_df['ts_event'].max()
    test_ts_min = test_df['ts_event'].min()
    test_ts_max = test_df['ts_event'].max()
    assert train_ts_max < test_ts_min, (
        f'TRAIN/TEST OVERLAP: train max={train_ts_max} >= test min={test_ts_min}'
    )
    train_df = _purge_train_tail_for_label_horizon(train_df, test_ts_min, target_col)
    train_ts_min = train_df['ts_event'].min()
    train_ts_max = train_df['ts_event'].max()
    print(f'[OUTER-TRUE-HMM] train_rows={train_df.height} train_ts=[{train_ts_min}, {train_ts_max})', flush=True)
    print(f'[OUTER-TRUE-HMM] test_rows={test_df.height} test_ts=[{test_ts_min}, {test_ts_max})', flush=True)
    print(f'[OUTER-TRUE-HMM] feature_cols={len(feature_cols)}', flush=True)
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' missing")
    train_y = train_df[target_col]
    print('[HEARTBEAT] model fit+predict start', flush=True)
    t_model = time.perf_counter()
    probs = train_and_predict(train_df, train_y, test_df, feature_cols)
    dt_model = time.perf_counter() - t_model
    print(f'[HEARTBEAT] model fit+predict done seconds={dt_model:.1f}', flush=True)
    if dt_model > 60:
        print(f'[SLOW] stage=model_fit_predict seconds={dt_model:.1f}', flush=True)
    pred_cs = hashlib.sha256(probs.tobytes()).hexdigest()[:8]
    result = test_df.with_columns(pl.Series('prediction_prob', probs).cast(pl.Float32))
    result = result.with_columns(compute_benchmark(result))
    print('[HEARTBEAT] execution simulation start', flush=True)
    t_exec = time.perf_counter()
    result = run_execution_simulation(result)
    dt_exec = time.perf_counter() - t_exec
    print(f'[HEARTBEAT] execution simulation done rows={result.height} seconds={dt_exec:.1f}', flush=True)
    if dt_exec > 60:
        print(f'[SLOW] stage=execution_simulation seconds={dt_exec:.1f}', flush=True)
    print('[HEARTBEAT] HMM fit start', flush=True)
    t_hmm = time.perf_counter()
    print(f'[HEARTBEAT] HMM resampling train 5m->1h start rows={train_df.height}', flush=True)
    df_1h_train = _resample_to_1h(train_df)
    print(f'[HEARTBEAT] HMM resampled train 1h rows={df_1h_train.height}', flush=True)
    hmm_filter = HMMRegimeFilter()
    if df_1h_train.height >= 60:
        hmm_cfg = hmm_filter.config
        print(f'[HEARTBEAT] HMM initializing: {df_1h_train.height} 1H bars, {len(hmm_cfg.feature_columns)} features, '
              f'n_states={hmm_cfg.n_states} n_iter={hmm_cfg.n_iter} tol={hmm_cfg.tol}', flush=True)
        hmm_filter.initialize(df_1h_train)
        print(f'[HEARTBEAT] HMM initialized', flush=True)
    else:
        print(f'[HEARTBEAT] HMM skipped – only {df_1h_train.height} 1H bars (<60)', flush=True)
    df_1h_test = _resample_to_1h(test_df)
    pre_hmm = _execution_state(result)
    result, hmm_filter = apply_hmm_filter(
        df_5min_base=result, df_1h_train=df_1h_train,
        df_1h_test=df_1h_test, hmm_filter=hmm_filter, retrain=False,
    )
    post_gate = _execution_state(result)
    result = _recompute_pnl_after_gate(result)
    post_recompute = _execution_state(result)
    print(
        '[HMM-GATE-DIAG] '
        f'before rows={pre_hmm["rows"]} active={pre_hmm["active"]} trades={pre_hmm["trades"]} '
        f'pnl_sum={pre_hmm["pnl_sum"]:.2f} pnl_cs={pre_hmm["pnl_cs"]} | '
        f'after_gate rows={post_gate["rows"]} active={post_gate["active"]} trades={post_gate["trades"]} '
        f'pnl_sum={post_gate["pnl_sum"]:.2f} pnl_cs={post_gate["pnl_cs"]} | '
        f'after_recompute rows={post_recompute["rows"]} active={post_recompute["active"]} '
        f'trades={post_recompute["trades"]} pnl_sum={post_recompute["pnl_sum"]:.2f} '
        f'pnl_cs={post_recompute["pnl_cs"]}',
        flush=True,
    )
    dt_hmm = time.perf_counter() - t_hmm
    print(f'[HEARTBEAT] HMM fit done seconds={dt_hmm:.1f}', flush=True)
    if dt_hmm > 60:
        print(f'[SLOW] stage=hmm_fit seconds={dt_hmm:.1f}', flush=True)
    if _OUTER_BURN_IN > 0:
        result = exclude_warmup(result, _OUTER_BURN_IN)
    output_frac = result.height / max(test_df.height, 1)
    assert output_frac >= 0.90, (
        f'OUTER-TRUE HMM ROW COLLAPSE: output={result.height} test={test_df.height} '
        f'fraction={output_frac:.2%} < 90%'
    )
    ts_out_min = result['ts_event'].min()
    ts_out_max = result['ts_event'].max()
    pnl_cs = _hash_col(result, 'pnl') if 'pnl' in result.columns else 'missing'
    print(f'[OUTER-TRUE-HMM] pred_cs={pred_cs} pnl_cs={pnl_cs}', flush=True)
    print(f'[OUTER-TRUE-HMM] result_rows={result.height} fraction={output_frac:.1%} '
          f'result_ts=[{ts_out_min}, {ts_out_max})', flush=True)
    _print_alpha_diagnostics(result, test_df, train_df, target_col, feature_cols, pred_cs, pnl_cs)
    validation = {'n_folds': 1, 'outer_split': True, 'mode': 'outer_split',
                  'train_rows': train_df.height, 'test_rows': test_df.height,
                  'output_rows': result.height, 'hmm_active': hmm_filter.is_active}
    return result, validation
