import polars as pl
import logging
from datetime import time
import pytz
from pathlib import Path
import tempfile
import glob
import shutil
from pipeline.common.config import config, load_config
from pipeline.common.io.atomic import atomic_write_parquet
from tqdm import tqdm

load_config()  # ensure config is populated (idempotent)

logger = logging.getLogger(__name__)
TZ = pytz.timezone(config.TIMEZONE)
SESSION_START = config.SESSION_START_LOCAL
SESSION_END = config.SESSION_END_LOCAL
SESSION_BREAK_START = config.SESSION_BREAK_START_LOCAL
SESSION_BREAK_END = config.SESSION_BREAK_END_LOCAL

# Column projection: only read columns needed for resampling.
# Session-normalized files may include audit columns used to drop non-tradable
# pre-observation padding before features/targets.
_OHLCV_PROJECTION = ['ts_event', 'open', 'high', 'low', 'close', 'volume']
_CAUSAL_GATE_COLUMNS = ['has_causal_price', 'session_segment_id', 'is_observed_bar']


def session_start_offset_by() -> str:
    start = config.SESSION_START_LOCAL
    start_minutes = start.hour * 60 + start.minute
    offset_minutes = (24 * 60 - start_minutes) % (24 * 60)
    hours, minutes = divmod(offset_minutes, 60)
    parts = []
    if hours:
        parts.append(f'{hours}h')
    if minutes:
        parts.append(f'{minutes}m')
    return ''.join(parts) or '0m'


def session_id_expr(local_ts_col: str = 'ts_local') -> pl.Expr:
    return (
        pl.col(local_ts_col)
        .dt.offset_by(session_start_offset_by())
        .dt.date()
        .cast(pl.String)
    )


def add_session_id(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).alias('ts_local'))
    df = df.with_columns(session_id_expr('ts_local').alias('session_id'))
    return df.drop('ts_local')


def filter_session_hours(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).alias('ts_local'))
    time_local = pl.col('ts_local').dt.time()
    in_session = (time_local >= SESSION_START) | (time_local < SESSION_END)
    gap = (time_local >= SESSION_BREAK_START) & (time_local < SESSION_BREAK_END)
    df = df.filter(in_session & ~gap)
    return df.drop('ts_local')


def resample_to_frequency(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    df = df.with_columns(pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).alias('ts_local'))
    df = df.with_columns(pl.col('ts_local').dt.truncate(every=freq).alias(f'ts_{freq}'))
    agg = df.group_by(['session_id', f'ts_{freq}'], maintain_order=True).agg([
        pl.col('open').first().alias('open'),
        pl.col('high').max().alias('high'),
        pl.col('low').min().alias('low'),
        pl.col('close').last().alias('close'),
        pl.col('volume').sum().alias('volume'),
        pl.len().alias('n_ticks'),
    ])
    if freq == '5m' and config.DROP_INCOMPLETE_ROWS:
        agg = agg.filter(pl.col('n_ticks') == 5)
    elif freq == '1h':
        agg = agg.filter(pl.col('n_ticks') >= 45)
    elif freq == '1d':
        agg = agg.filter(pl.col('n_ticks') >= 360)
    agg = agg.rename({f'ts_{freq}': 'ts_event'})
    agg = agg.drop('n_ticks')
    agg = agg.sort(['session_id', 'ts_event'])
    if freq == '1d':
        agg = agg.with_columns(pl.col('close').log().alias('log_close'))
        # One row per session. Do not shift over session_id: each group has one
        # row, so that makes every daily return null.
        agg = agg.with_columns(
            (pl.col('log_close') - pl.col('log_close').shift(1)).alias('daily_log_return')
        )
        agg = agg.with_columns(
            pl.col('daily_log_return')
            .rolling_std(window_size=5, min_periods=2)
            .alias('daily_vol_5')
        )
        agg = agg.with_columns(
            pl.col('daily_vol_5')
            .fill_null(strategy='forward')
            .fill_null(strategy='backward')
            .fill_null(0.0)
        )
        agg = agg.drop(['log_close'])
    agg = agg.with_columns(pl.col('ts_event').dt.convert_time_zone('UTC').alias('ts_event'))
    agg = agg.with_columns([
        pl.col('open').cast(pl.Float32),
        pl.col('high').cast(pl.Float32),
        pl.col('low').cast(pl.Float32),
        pl.col('close').cast(pl.Float32),
    ])
    return agg


def _read_ohlcv_with_optional_causal_gate(file_path: str) -> pl.DataFrame:
    schema_names = set(pl.scan_parquet(file_path).collect_schema().names())
    cols = _OHLCV_PROJECTION + [c for c in _CAUSAL_GATE_COLUMNS if c in schema_names]
    df = pl.read_parquet(file_path, columns=cols)

    if set(_CAUSAL_GATE_COLUMNS).issubset(df.columns):
        before = df.height
        df = df.sort(['session_segment_id', 'ts_event'])
        df = df.with_columns(
            pl.col('is_observed_bar')
            .cum_max()
            .over('session_segment_id')
            .alias('_segment_seen_observed')
        )
        viol = df.filter((~pl.col('has_causal_price')) & pl.col('_segment_seen_observed')).height
        if viol:
            raise RuntimeError(f'SESSION LOAD FAIL: non-causal rows after observed bars: {viol}')

        df = df.filter(pl.col('has_causal_price')).drop(_CAUSAL_GATE_COLUMNS + ['_segment_seen_observed'])
        dropped = before - df.height
        if dropped:
            print(f'[SESSION] Dropped non-causal padding rows: {dropped} file={file_path}', flush=True)

    return df


def process_one_file(file_path: str, out_temp_dir: str, freq: str) -> str | None:
    """Read OHLCV columns and drop session-normalized non-causal padding if present."""
    df = _read_ohlcv_with_optional_causal_gate(file_path)
    if df['ts_event'].dtype != pl.Datetime:
        df = df.with_columns(pl.col('ts_event').cast(pl.Datetime(time_unit='us', time_zone='UTC')))
    df = filter_session_hours(df)
    if df.is_empty():
        return None
    df = add_session_id(df)
    df_resampled = resample_to_frequency(df, freq)
    if df_resampled.is_empty():
        return None
    out_file = Path(out_temp_dir) / f'{Path(file_path).stem}_{freq}.parquet'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(df_resampled, out_file)
    return str(out_file)


def process_frequency(freq: str, all_files: list) -> tuple:
    print(f'\n[SESSION] Resampling {freq} (found {len(all_files)} files)', flush=True)
    temp_dir = tempfile.mkdtemp(prefix=f'resampled_{freq}_')
    temp_paths = []
    try:
        for f in tqdm(all_files, desc=f'Resampling {freq}', unit='file'):
            out = process_one_file(f, temp_dir, freq)
            if out:
                temp_paths.append(out)
        if not temp_paths:
            raise ValueError(f'No data after resampling to {freq}')
        lf = pl.scan_parquet(temp_paths)
        lf = lf.sort(['session_id', 'ts_event'])
        try:
            df = lf.collect(engine='streaming')
        except TypeError:
            df = lf.collect(streaming=True)
        print(f'[SESSION] {freq} stream has {df.height} rows.', flush=True)
        return (freq, df)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_all_streams_chunked(data_glob: str) -> dict:
    all_files = sorted(glob.glob(data_glob))
    if not all_files:
        raise FileNotFoundError(f'No parquet files found matching {data_glob}')
    print(f'[SESSION] Found {len(all_files)} files.', flush=True)
    streams = {}
    for freq in getattr(config, 'RESAMPLE_FREQUENCIES', ['5m', '1h', '1d']):
        _, df = process_frequency(freq, all_files)
        streams[freq] = df
    return streams
