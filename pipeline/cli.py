import os
import sys

# ── UTF-8 everywhere (Windows hardening) ──
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print('[HEARTBEAT] pipeline.cli imported', flush=True)

import argparse
import logging
import random
import numpy as np
import psutil
from pathlib import Path
import polars as pl
import json
import hashlib
import time
from pipeline.common.io.atomic import atomic_write_parquet, atomic_write_json

_MODEL_METADATA_EXCLUDE_EXACT = {
    'continuous_price',
    'adjustment_factor',
    'cumulative_factor',
    'contract_multiplier',
}
_MODEL_METADATA_EXCLUDE_PREFIXES = (
    'continuous_',
    'roll_',
    'front_contract',
    'back_contract',
)


def _is_model_metadata_column(col: str) -> bool:
    return col in _MODEL_METADATA_EXCLUDE_EXACT or col.startswith(_MODEL_METADATA_EXCLUDE_PREFIXES)


def _model_feature_columns(df: pl.DataFrame) -> list[str]:
    _exclude = {'ts_event', 'open', 'high', 'low', 'close', 'volume', 'session_id', 'regime', 'date', 'benchmark_pnl'}
    _exclude |= {c for c in df.columns if c.startswith('target_') or _is_model_metadata_column(c)}
    _numeric_types = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    return [c for c in df.columns if c not in _exclude and df[c].dtype in _numeric_types]


def _check_target_contract(y, X, target_col):
    """Hard target validation — must pass before walkforward."""
    if len(y) == 0:
        raise RuntimeError('TARGET FAIL: y is empty after pruning')
    y_nan = y.null_count()
    if y_nan > 0:
        raise RuntimeError(
            'TARGET FAIL: %d NaN values in y (%s) after drop_incomplete_target' %
            (y_nan, target_col)
        )
    if X.height != len(y):
        raise RuntimeError(
            'TARGET FAIL: X/y misalignment (X=%d, y=%d)' % (X.height, len(y))
        )
    print('[TARGET] %s: rows=%d NaN=0 aligned=X(%d)' %
          (target_col, len(y), X.height), flush=True)


def _diag(df, stage):
    """Pipeline diagnostic — row count and ts_event span at each stage."""
    print(f'[DIAG] stage={stage} rows={df.height} cols={len(df.columns)}', flush=True)
    if 'ts_event' in df.columns and df.height > 0:
        t = df.select([pl.col('ts_event').min().alias('lo'), pl.col('ts_event').max().alias('hi')])
        print(f'[DIAG]   ts_event min={t["lo"][0]} max={t["hi"][0]}', flush=True)
from pipeline.common.config import config, load_config
from pipeline.ingest.ingest import load_and_clean_data
from pipeline.features.engine import load_or_build_feature_target_matrix
from pipeline.features.discovery import apply_frozen_feature_manifest, run_train_only_feature_discovery
from pipeline.walkforward.walkforward import build_oos_prediction_frame, run_walkforward, run_walkforward_with_hmm, run_walkforward_modeling, run_walkforward_modeling_with_hmm
from pipeline.common.io.canonical import write_canonical_parquet
from pipeline.analytics.aggregate import build_metrics_report, calculate_metrics, run_aggregation
from pipeline.risk.risk import RiskGateError, run_risk_gates
logger = logging.getLogger(__name__)

def check_memory_safety():
    try:
        mem = psutil.Process().memory_info().rss
        if mem > config.RAM_CAP_BYTES:
            raise MemoryError(f'RSS {mem / 1024 ** 3:.2f}GB > cap {config.RAM_CAP_BYTES / 1024 ** 3:.2f}GB')
    except ImportError:
        pass

def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cache_config_fingerprint(manifest_path: str | None = None) -> str:
    names = [
        'ACTIVE_PROFILE', 'CONFIG_SOURCE', 'CURRENT_SYMBOL',
        'TARGET_15M_HORIZON', 'TARGET_SCALE_FACTOR',
        'ENABLE_EXPANSION', 'ENABLE_DISCOVERY',
        'MODEL_TYPE', 'RIDGE_PARAMS', 'PROBABILITY_SMOOTHING_ALPHA',
        'CORR_THRESHOLD', 'WF_MODE', 'DISCOVERY_TARGET', 'WALKFORWARD_TARGET',
        'EXECUTE_AT', 'SLIPPAGE_K', 'TX_COST_PER_ROUNDTURN',
        'COMMISSION_PER_CONTRACT', 'MAX_LEVERAGE',
        'HTF_TREND_ALIGNMENT', 'HTF_VOL_SCALING',
        'ROLL_WINDOWS', 'ROLL_WINDOWS_1H', 'ROLL_WINDOWS_DAILY',
    ]
    payload = {name: _jsonable(getattr(config, name, None)) for name in names}
    payload['alpha_tiered_yaml_sha256'] = _file_sha256('configs/alpha_tiered.yaml')
    payload['baseline_features_sha256'] = _file_sha256(getattr(config, 'BASELINE_FEATURES_FILE', 'configs/baseline_features.yaml'))
    payload['market_config_sha256'] = _file_sha256((getattr(config, 'MARKET_CONFIGS', {}) or {}).get(getattr(config, 'CURRENT_SYMBOL', None)))
    payload['manifest_sha256'] = _file_sha256(manifest_path)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def _stable_data_tag(data_arg: str, start: str = None, end: str = None, manifest_path: str = None) -> str:
    key = f"{config.ACTIVE_PROFILE}|{data_arg}|{_cache_config_fingerprint(manifest_path)}"
    if start:
        key += '|' + start
    if end:
        key += '|' + end
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return h


def _slice_optional_window(df: pl.DataFrame, start_str: str | None, end_str: str | None, label: str) -> pl.DataFrame:
    if not start_str and not end_str:
        return df
    if not start_str or not end_str:
        raise RuntimeError(f'MISSING BOUNDARY: {label} requires both --start and --end')
    return _slice_window(df, start_str, end_str, label)


def _target_horizon_minutes(target_col: str | None) -> int:
    if target_col in {'target_15m_ret', 'target_15m_dir', 'target_15m_trade_class'}:
        return int(getattr(config, 'TARGET_15M_HORIZON', 15)) * 5
    return 0


def _slice_window(df: pl.DataFrame, start_str: str, end_str: str, label: str, target_col: str | None = None) -> pl.DataFrame:
    from datetime import datetime as _dt, timezone, timedelta
    if not start_str or not end_str:
        raise RuntimeError(f'MISSING BOUNDARY: {label} start/end required for outer_split mode')
    start_dt = _dt.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end_dt = _dt.fromisoformat(end_str).replace(tzinfo=timezone.utc)
    before = df.height
    effective_end = end_dt
    horizon_minutes = _target_horizon_minutes(target_col) if label == 'test' else 0
    if horizon_minutes > 0:
        effective_end = end_dt - timedelta(minutes=horizon_minutes)
    result = df.filter((pl.col('ts_event') >= start_dt) & (pl.col('ts_event') < effective_end))
    ts_min = result['ts_event'].min() if result.height > 0 else None
    ts_max = result['ts_event'].max() if result.height > 0 else None
    print(f'[OUTER-TRUE] {label} filter ({start_str} -> {end_str}, effective_end={effective_end}): {before} -> {result.height} rows, ts=[{ts_min}, {ts_max})', flush=True)
    if result.height == 0:
        raise RuntimeError(f'EMPTY WINDOW: {label} [{start_str}, {end_str}) returned 0 rows')
    return result

def main():
    print('[HEARTBEAT] pipeline.cli main entered', flush=True)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    discover_parser = subparsers.add_parser('discover')
    discover_parser.add_argument('--data', required=True)
    discover_parser.add_argument('--out', default='output/manifests/manifest.json')
    discover_parser.add_argument('--start', default=None, help='Discovery train-window start date (ISO)')
    discover_parser.add_argument('--end', default=None, help='Discovery train-window end date (ISO)')
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--data', required=True)
    run_parser.add_argument('--manifest', default='output/manifests/manifest.json')
    run_parser.add_argument('--out', required=True)
    run_parser.add_argument('--start', default=None, help='Start date (ISO format)')
    run_parser.add_argument('--end', default=None, help='End date (ISO format)')
    run_parser.add_argument('--train-start', default=None, help='Train window start date (ISO)')
    run_parser.add_argument('--train-end', default=None, help='Train window end date (ISO)')
    run_hmm_parser = subparsers.add_parser('run-hmm')
    run_hmm_parser.add_argument('--data', required=True)
    run_hmm_parser.add_argument('--manifest', default='output/manifests/manifest.json')
    run_hmm_parser.add_argument('--out', required=True)
    run_hmm_parser.add_argument('--start', default=None, help='Start date (ISO format)')
    run_hmm_parser.add_argument('--end', default=None, help='End date (ISO format)')
    run_hmm_parser.add_argument('--train-start', default=None, help='Train window start date (ISO)')
    run_hmm_parser.add_argument('--train-end', default=None, help='Train window end date (ISO)')
    run_hmm_parser.add_argument('--retrain-interval', type=int, default=5,
                                help='Retrain HMM every N folds (default: 5).')
    aggregate_parser = subparsers.add_parser('aggregate')
    aggregate_parser.add_argument('--artifacts', default='output')
    args = parser.parse_args()
    load_config()  # populate config namespace from configs/alpha_tiered.yaml
    wf_mode = getattr(config, 'WF_MODE', '')
    env_name = os.environ.get('CONFIG_ENV') or os.environ.get('QUANT_ENV', 'default')
    print(f'[CONFIG] env={env_name} WF_MODE={wf_mode!r} train_start={getattr(args, "train_start", None)} train_end={getattr(args, "train_end", None)} start={getattr(args, "start", None)} end={getattr(args, "end", None)}', flush=True)
    print(f'[BRANCH] command={args.command} using_outer_split={wf_mode == "outer_split"}', flush=True)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    check_memory_safety()
    if args.command in ('discover', 'run', 'run-hmm'):
        from pipeline.common.market import detect_symbol_from_path, load_market_config
        symbol = detect_symbol_from_path(args.data)
        load_market_config(symbol)
        config.CURRENT_SYMBOL = symbol
    if args.command == 'discover':
        print('\n[CLI] === PHASE 1: FEATURE DISCOVERY ===', flush=True)
        cache_dir = Path('output/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_tag = _stable_data_tag(args.data, getattr(args, 'start', None), getattr(args, 'end', None))
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        canonical_cache = cache_dir / f'canonical_data_{data_tag}.parquet'
        print(f'[DISCOVERY-WINDOW] start={getattr(args, "start", None)} end={getattr(args, "end", None)} cache_key={data_tag}', flush=True)
        print('[CLI] Loading and cleaning data...', flush=True)
        df_aligned = load_and_clean_data(
            args.data,
            cache_path=str(aligned_cache),
            canonical_cache_path=str(canonical_cache),
        )
        print(f'[CLI] Data loaded. Rows: {df_aligned.height}', flush=True)
        df_aligned = _slice_optional_window(
            df_aligned,
            getattr(args, 'start', None),
            getattr(args, 'end', None),
            'discovery',
        )
        if getattr(args, 'start', None) and getattr(args, 'end', None):
            print(f'[DISCOVERY-WINDOW] bounded rows={df_aligned.height} [{args.start}, {args.end})', flush=True)
        _diag(df_aligned, 'post-ingest')
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        df_features = load_or_build_feature_target_matrix(
            df_aligned,
            cache_path=feature_cache,
            target_col=getattr(config, 'DISCOVERY_TARGET', 'target_15m_ret'),
        )
        _diag(df_features, 'post-generate-features')
        print(f'[CLI] Feature matrix saved to {feature_cache}', flush=True)
        print('[CLI] Running feature discovery...', flush=True)
        run_train_only_feature_discovery(
            str(feature_cache),
            args.out,
            train_start=getattr(args, 'start', None),
            train_end=getattr(args, 'end', None),
        )
    elif args.command == 'run':
        print('\n[CLI] === PHASE 2: WALKFORWARD & EXECUTION ===', flush=True)
        target_col = getattr(config, 'WALKFORWARD_TARGET', 'target_15m_ret')
        cache_dir = Path('output/cache')
        data_tag = _stable_data_tag(args.data, getattr(args, 'start', None), getattr(args, 'end', None), getattr(args, 'manifest', None))
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        canonical_cache = cache_dir / f'canonical_data_{data_tag}.parquet'
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        print(f'[CLI] Cache key={data_tag} start={getattr(args, "start", None)} end={getattr(args, "end", None)}', flush=True)
        print('[CLI] Loading aligned data...', flush=True)
        df_aligned = load_and_clean_data(
            args.data,
            cache_path=str(aligned_cache) if aligned_cache.exists() else None,
            canonical_cache_path=str(canonical_cache),
        )
        print(f'[CLI] Aligned data: {df_aligned.height} rows (from {"cache" if aligned_cache.exists() else "fresh ingest"})', flush=True)
        df_features = load_or_build_feature_target_matrix(
            df_aligned,
            cache_path=feature_cache,
            target_col=target_col,
        )
        print(f'[CLI] Feature matrix: {df_features.height} rows, {len(df_features.columns)} cols', flush=True)
        print('[CLI] Applying frozen feature manifest...', flush=True)
        if config.ENABLE_DISCOVERY:
            df_pruned = apply_frozen_feature_manifest(df_features, args.manifest, target_col)
        else:
            print('[CLI] Discovery disabled - skipping manifest pruning.', flush=True)
            df_pruned = df_features
        print(f'[CLI] After manifest: {df_pruned.height} rows, {len(df_pruned.columns)} cols (target={"included" if target_col in df_pruned.columns else "MISSING"})', flush=True)
        # Per-split date window filtering (skipped in outer_split — _slice_window handles dual slicing)
        if getattr(config, 'WF_MODE', '') != 'outer_split':
            if getattr(args, 'start', None) and getattr(args, 'end', None):
                from datetime import datetime as _dt, timezone
                start_dt = _dt.fromisoformat(args.start).replace(tzinfo=timezone.utc)
                end_dt = _dt.fromisoformat(args.end).replace(tzinfo=timezone.utc)
                before = df_pruned.height
                df_pruned = df_pruned.filter(
                    (pl.col('ts_event') >= start_dt) & (pl.col('ts_event') < end_dt)
                )
                print(f'[CLI] Date filter ({args.start} -> {args.end}): {before} -> {df_pruned.height} rows', flush=True)
                if df_pruned.height == 0:
                    print('[CLI] Empty date window -- writing placeholder output and exiting', flush=True)
                    os.makedirs(args.out, exist_ok=True)
                    placeholder = pl.DataFrame(schema={'pnl': pl.Float32})
                    placeholder.write_parquet(os.path.join(args.out, 'backtest_results.parquet'))
                    print(f'[CLI] Empty backtest written to {args.out}', flush=True)
                    return
            # Assertion: all rows must be within the requested date window
            t_min = df_pruned['ts_event'].min()
            t_max = df_pruned['ts_event'].max()
            assert t_min >= start_dt, (
                f'SPLIT BOUNDARY VIOLATION: ts_event min={t_min} < start={start_dt}. '
                f'Upstream data or cache may have stale full-year data.'
            )
            assert t_max < end_dt, (
                f'SPLIT BOUNDARY VIOLATION: ts_event max={t_max} >= end={end_dt}. '
                f'Upstream data or cache may have stale full-year data.'
            )
            logger.info('[SPLIT] run: %d rows ts_event range [%s, %s) within [%s, %s)',
                df_pruned.height, t_min, t_max, start_dt, end_dt)
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target column '{target_col}' missing!")
        y = df_pruned[target_col]
        X = df_pruned.drop(target_col)

        # ---- Target contract enforcement (run path) ----
        _check_target_contract(y, X, target_col)

        feature_cols = _model_feature_columns(X)
        print(f'[CLI] Running walkforward with {len(feature_cols)} features...', flush=True)
        if getattr(config, 'WF_MODE', '') == 'outer_split':
            assert getattr(args, 'train_start', None), 'OUTER_SPLIT active but --train-start missing from CLI args'
            assert getattr(args, 'train_end', None), 'OUTER_SPLIT active but --train-end missing from CLI args'
            train_df = _slice_window(df_pruned, getattr(args, 'train_start', None), getattr(args, 'train_end', None), 'train')
            test_df = _slice_window(df_pruned, getattr(args, 'start', None), getattr(args, 'end', None), 'test', target_col)
            result_df = run_walkforward_modeling(train_df, test_df, feature_cols, target_col)
        else:
            result_df = run_walkforward(X, y, feature_cols, target_col)
        print(f'[CLI] Walkforward result: {result_df.height} rows (input was {X.height} rows, features={len(feature_cols)})', flush=True)
        os.makedirs(args.out, exist_ok=True)
        pred_path = os.path.join(args.out, 'oos_predictions.parquet')
        pred_df = build_oos_prediction_frame(result_df, target_col=target_col)
        atomic_write_parquet(pred_df, pred_path)
        print(f'[CLI] OOS predictions saved to {pred_path}', flush=True)
        risk_path = os.path.join(args.out, 'risk_report.json')
        risk_error = None
        try:
            risk_report = run_risk_gates(result_df, context={'command': args.command, 'out': args.out})
        except RiskGateError as exc:
            risk_error = exc
            risk_report = getattr(exc, 'report', None) or {
                'status': 'FAIL',
                'context': {'command': args.command, 'out': args.out},
                'error': str(exc),
            }
        atomic_write_json(risk_report, risk_path)
        print(f'[CLI] Risk report saved to {risk_path}', flush=True)
        metrics_report = build_metrics_report(
            result_df,
            context={'command': args.command, 'out': args.out, 'risk_report': risk_path},
        )
        metrics_path = os.path.join(args.out, 'metrics_report.json')
        atomic_write_json(metrics_report, metrics_path)
        print(f'[CLI] Metrics report saved to {metrics_path}', flush=True)
        out_path = os.path.join(args.out, 'backtest_results.parquet')
        atomic_write_parquet(result_df, out_path)
        print(f'[CLI] Results saved to {out_path}', flush=True)
        if risk_error is not None:
            raise risk_error
        print('\n================ METRICS ================')
        calculate_metrics(out_path)
        print('========================================\n')
        try:
            print('[CLI] Running aggregation...', flush=True)
            run_aggregation('output')
        except Exception as e:
            print(f'[CLI] Aggregation skipped: {e}', flush=True)
    elif args.command == 'run-hmm':
        t0 = time.perf_counter()
        print('\n[CLI] === PHASE 2H: WALKFORWARD + HMM REGIME FILTER ===', flush=True)
        print('[HEARTBEAT] cli entered run-hmm', flush=True)
        print(f'[HEARTBEAT] config loaded WF_MODE={getattr(config, "WF_MODE", "")!r}', flush=True)
        target_col = getattr(config, 'WALKFORWARD_TARGET', 'target_15m_ret')
        cache_dir = Path('output/cache')
        data_tag = _stable_data_tag(args.data, getattr(args, 'start', None), getattr(args, 'end', None), getattr(args, 'manifest', None))
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        canonical_cache = cache_dir / f'canonical_data_{data_tag}.parquet'
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        print(f'[CLI] Cache key={data_tag} start={getattr(args, "start", None)} end={getattr(args, "end", None)}', flush=True)
        print('[HEARTBEAT] loading aligned data start', flush=True)
        t_load = time.perf_counter()
        df_aligned = load_and_clean_data(
            args.data,
            cache_path=str(aligned_cache) if aligned_cache.exists() else None,
            canonical_cache_path=str(canonical_cache),
        )
        dt_load = time.perf_counter() - t_load
        print(f'[HEARTBEAT] loading aligned data done rows={df_aligned.height} seconds={dt_load:.1f}', flush=True)
        if dt_load > 60:
            print(f'[SLOW] stage=load_aligned_data seconds={dt_load:.1f}', flush=True)
        print('[HEARTBEAT] generating/loading feature matrix start', flush=True)
        t_feat = time.perf_counter()
        df_features = load_or_build_feature_target_matrix(
            df_aligned,
            cache_path=feature_cache,
            target_col=target_col,
        )
        dt_feat = time.perf_counter() - t_feat
        print(f'[HEARTBEAT] feature matrix done rows={df_features.height} cols={len(df_features.columns)} seconds={dt_feat:.1f}', flush=True)
        if dt_feat > 60:
            print(f'[SLOW] stage=feature_matrix seconds={dt_feat:.1f}', flush=True)
        print(f'[CLI] Feature matrix: {df_features.height} rows, {len(df_features.columns)} cols', flush=True)
        print('[HEARTBEAT] applying frozen manifest start', flush=True)
        t_man = time.perf_counter()
        print('[CLI] Applying frozen feature manifest...', flush=True)
        if config.ENABLE_DISCOVERY:
            df_pruned = apply_frozen_feature_manifest(df_features, args.manifest, target_col)
        else:
            print('[CLI] Discovery disabled - skipping manifest pruning.', flush=True)
            df_pruned = df_features
        print(f'[CLI] After manifest: {df_pruned.height} rows, {len(df_pruned.columns)} cols', flush=True)
        dt_man = time.perf_counter() - t_man
        print(f'[HEARTBEAT] manifest done rows={df_pruned.height} cols={len(df_pruned.columns)} seconds={dt_man:.1f}', flush=True)
        if dt_man > 60:
            print(f'[SLOW] stage=manifest seconds={dt_man:.1f}', flush=True)
        # Per-split date window filtering (skipped in outer_split — _slice_window handles dual slicing)
        if getattr(config, 'WF_MODE', '') != 'outer_split':
            if getattr(args, 'start', None) and getattr(args, 'end', None):
                from datetime import datetime as _dt, timezone
                start_dt = _dt.fromisoformat(args.start).replace(tzinfo=timezone.utc)
                end_dt = _dt.fromisoformat(args.end).replace(tzinfo=timezone.utc)
                before = df_pruned.height
                df_pruned = df_pruned.filter(
                    (pl.col('ts_event') >= start_dt) & (pl.col('ts_event') < end_dt)
                )
                print(f'[CLI] Date filter ({args.start} -> {args.end}): {before} -> {df_pruned.height} rows', flush=True)
                if df_pruned.height == 0:
                    print('[CLI] Empty date window — writing placeholder output and exiting', flush=True)
                    os.makedirs(args.out, exist_ok=True)
                    placeholder = pl.DataFrame(schema={'pnl': pl.Float32})
                    placeholder.write_parquet(os.path.join(args.out, 'backtest_results_hmm.parquet'))
                    print(f'[CLI] Empty backtest written to {args.out}', flush=True)
                    return
            t_min = df_pruned['ts_event'].min()
            t_max = df_pruned['ts_event'].max()
            assert t_min >= start_dt, f'SPLIT BOUNDARY VIOLATION: min={t_min} < start={start_dt}'
            assert t_max < end_dt, f'SPLIT BOUNDARY VIOLATION: max={t_max} >= end={end_dt}'
            logger.info('[SPLIT] run-hmm: %d rows ts_event [%s, %s) within [%s, %s)',
                df_pruned.height, t_min, t_max, start_dt, end_dt)
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target column '{target_col}' missing!")
        y = df_pruned[target_col]
        X = df_pruned.drop(target_col)

        # ---- Target contract enforcement (hmm path) ----
        _check_target_contract(y, X, target_col)

        feature_cols = _model_feature_columns(X)
        print(
            f'[CLI] Running HMM-aware walkforward with '
            f'{len(feature_cols)} features '
            f'(HMM retrain every {args.retrain_interval} folds)...',
            flush=True,
        )
        if getattr(config, 'WF_MODE', '') == 'outer_split':
            assert getattr(args, 'train_start', None), 'OUTER_SPLIT HMM active but --train-start missing from CLI args'
            assert getattr(args, 'train_end', None), 'OUTER_SPLIT HMM active but --train-end missing from CLI args'
            print('[HEARTBEAT] slicing train/test start', flush=True)
            t_slice = time.perf_counter()
            train_df = _slice_window(df_pruned, getattr(args, 'train_start', None), getattr(args, 'train_end', None), 'train')
            test_df = _slice_window(df_pruned, getattr(args, 'start', None), getattr(args, 'end', None), 'test', target_col)
            dt_slice = time.perf_counter() - t_slice
            print(f'[HEARTBEAT] train rows={train_df.height} test rows={test_df.height} seconds={dt_slice:.1f}', flush=True)
            if dt_slice > 60:
                print(f'[SLOW] stage=slicing seconds={dt_slice:.1f}', flush=True)
            print('[HEARTBEAT] walkforward+hmm start', flush=True)
            t_wf = time.perf_counter()
            result_df, validation = run_walkforward_modeling_with_hmm(
                train_df, test_df, feature_cols, target_col,
            )
            dt_wf = time.perf_counter() - t_wf
            print(f'[HEARTBEAT] walkforward+hmm done rows={result_df.height} seconds={dt_wf:.1f}', flush=True)
            if dt_wf > 60:
                print(f'[SLOW] stage=walkforward+hmm seconds={dt_wf:.1f}', flush=True)
        else:
            result_df, validation = run_walkforward_with_hmm(
                X, y, feature_cols, target_col,
                hmm_retrain_interval=args.retrain_interval,
            )
        print(f'[CLI] HMM walkforward result: {result_df.height} rows (input was {X.height} rows, features={len(feature_cols)})', flush=True)
        os.makedirs(args.out, exist_ok=True)
        pred_path = os.path.join(args.out, 'oos_predictions_hmm.parquet')
        pred_df = build_oos_prediction_frame(result_df, target_col=target_col)
        atomic_write_parquet(pred_df, pred_path)
        print(f'[CLI] HMM OOS predictions saved to {pred_path}', flush=True)
        risk_path = os.path.join(args.out, 'risk_report_hmm.json')
        risk_error = None
        try:
            risk_report = run_risk_gates(result_df, context={'command': args.command, 'out': args.out})
        except RiskGateError as exc:
            risk_error = exc
            risk_report = getattr(exc, 'report', None) or {
                'status': 'FAIL',
                'context': {'command': args.command, 'out': args.out},
                'error': str(exc),
            }
        atomic_write_json(risk_report, risk_path)
        print(f'[CLI] HMM risk report saved to {risk_path}', flush=True)
        metrics_report = build_metrics_report(
            result_df,
            context={'command': args.command, 'out': args.out, 'risk_report': risk_path},
        )
        metrics_path = os.path.join(args.out, 'metrics_report_hmm.json')
        atomic_write_json(metrics_report, metrics_path)
        print(f'[CLI] HMM metrics report saved to {metrics_path}', flush=True)
        out_path = os.path.join(args.out, 'backtest_results_hmm.parquet')
        print('[HEARTBEAT] writing parquet start', flush=True)
        t_write = time.perf_counter()
        atomic_write_parquet(result_df, out_path)
        dt_write = time.perf_counter() - t_write
        print(f'[HEARTBEAT] writing parquet done path={out_path} seconds={dt_write:.1f}', flush=True)
        if dt_write > 60:
            print(f'[SLOW] stage=parquet_write seconds={dt_write:.1f}', flush=True)
        dt_total = time.perf_counter() - t0
        print(f'[HEARTBEAT] total run-hmm seconds={dt_total:.1f}', flush=True)
        print(f'[CLI] HMM-filtered results saved to {out_path}', flush=True)
        # Save validation report
        val_path = os.path.join(args.out, 'hmm_validation_report.json')
        atomic_write_json(validation, val_path)
        print(f'[CLI] Validation report saved to {val_path}', flush=True)
        print('\n================ HMM METRICS ================')
        calculate_metrics(out_path)
        print('==============================================\n')
        if not validation.get('fallback_triggered', True):
            print(
                f"[CLI] PSR: {validation['psr_result']['psr']:.4f} | "
                f"Significant: {validation['psr_result']['significant']} | "
                f"ΔSR: {validation['psr_result']['sharpe_difference']:+.4f}",
                flush=True,
            )
        else:
            print(
                f"[CLI] HMM FALLBACK ACTIVE: {validation.get('fallback_reason', 'unknown')}",
                flush=True,
            )
        print(f"[CLI] Recommendation: {validation.get('recommendation', 'N/A')}", flush=True)
        try:
            print('[CLI] Running aggregation...', flush=True)
            run_aggregation('output')
        except Exception as e:
            print(f'[CLI] Aggregation skipped: {e}', flush=True)
        if risk_error is not None:
            raise risk_error
    elif args.command == 'aggregate':
        print('\n[CLI] === AGGREGATING RESULTS ===', flush=True)
        run_aggregation(args.artifacts)
if __name__ == '__main__':
    main()
