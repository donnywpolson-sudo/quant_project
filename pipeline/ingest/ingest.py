import polars as pl
import logging
import psutil
from pathlib import Path
from pipeline.common.config import config
from pipeline.session.session import load_all_streams_chunked
from pipeline.align.align import align_htf_streams
from pipeline.contracts.continuous import build_continuous_series
from pipeline.common.market import detect_symbol_from_path, load_market_config, get_contract_multiplier
from pipeline.common.io.canonical import write_canonical_parquet

logger = logging.getLogger(__name__)

_CANONICAL_REQUIRED_COLS = ['ts_event', 'open', 'high', 'low', 'close', 'volume', 'session_id']
_ALIGNED_CONTINUOUS_REQUIRED_COLS = [
    *_CANONICAL_REQUIRED_COLS,
    'continuous_price',
    'adjustment_factor',
    'contract_month',
    'contract_multiplier',
    'continuous_open',
    'continuous_high',
    'continuous_low',
]


def validate_memory_and_integrity(df: pl.DataFrame) -> int:
    logger.info('Running memory and integrity validation...')
    if df.is_empty():
        raise ValueError(
            'validate_memory_and_integrity: DataFrame is empty — '
            'upstream processing (resampling, session filter, or gap filter) '
            'may have removed all rows.'
        )
    if not df['ts_event'].is_sorted():
        raise ValueError('ts_event not strictly increasing.')
    critical_cols = ['open', 'high', 'low', 'close', 'volume', 'session_id']
    for col in critical_cols:
        if df[col].null_count() > 0:
            raise ValueError(f'Nulls in column {col}.')
    if (df['high'] < df['low']).any():
        raise ValueError('High < Low detected.')
    if ((df['open'] < df['low']) | (df['open'] > df['high'])).any():
        raise ValueError('Open outside [Low, High].')
    if ((df['close'] < df['low']) | (df['close'] > df['high'])).any():
        raise ValueError('Close outside [Low, High].')
    est_bytes = df.estimated_size()
    rows = df.height
    logger.info(f'Memory usage: {est_bytes / 1024 ** 3:.2f} GB')
    if est_bytes > config.RAM_CAP_BYTES:
        raise MemoryError(f'Data size {est_bytes} exceeds RAM_CAP_BYTES.')
    avg_row_bytes = est_bytes / rows if rows > 0 else 0
    rows_per_chunk = min(
        config.ROWS_PER_CHUNK_MAX,
        int(config.RAM_CAP_BYTES * config.MEMORY_SAFETY_MARGIN / (avg_row_bytes + 1)),
    )
    logger.info(f'Safe rows_per_chunk: {rows_per_chunk}')
    return rows_per_chunk


def validate_canonical_stream(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    missing = [c for c in _CANONICAL_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'CANONICAL FAIL: {freq} missing columns {missing}')
    if df.is_empty():
        raise ValueError(f'CANONICAL FAIL: {freq} stream is empty')
    if df['ts_event'].dtype != pl.Datetime(time_unit='us', time_zone='UTC'):
        df = df.with_columns(pl.col('ts_event').cast(pl.Datetime(time_unit='us', time_zone='UTC')))
    if not df['ts_event'].is_sorted():
        raise ValueError(f'CANONICAL FAIL: {freq} ts_event not sorted')
    if df['ts_event'].n_unique() != df.height:
        raise ValueError(f'CANONICAL FAIL: {freq} duplicate ts_event values')
    null_cols = [c for c in _CANONICAL_REQUIRED_COLS if df[c].null_count() > 0]
    if null_cols:
        raise ValueError(f'CANONICAL FAIL: {freq} nulls in {null_cols}')
    if (df['high'] < df['low']).any():
        raise ValueError(f'CANONICAL FAIL: {freq} high < low')
    if ((df['open'] < df['low']) | (df['open'] > df['high'])).any():
        raise ValueError(f'CANONICAL FAIL: {freq} open outside [low, high]')
    if ((df['close'] < df['low']) | (df['close'] > df['high'])).any():
        raise ValueError(f'CANONICAL FAIL: {freq} close outside [low, high]')
    return df


def _canonical_cache_file(base_path: str | Path, freq: str) -> Path:
    p = Path(base_path)
    return p.with_name(f'{p.stem}_{freq}{p.suffix}')


def _canonical_cache_files(base_path: str | Path) -> dict[str, Path]:
    freqs = list(getattr(config, 'RESAMPLE_FREQUENCIES', ['1m']))
    return {freq: _canonical_cache_file(base_path, freq) for freq in freqs}


def _primary_bar_freq(streams: dict[str, pl.DataFrame]) -> str:
    for freq in ('1m', '5m'):
        if freq in streams:
            return freq
    raise ValueError(
        f'CANONICAL FAIL: no executable bar stream found. '
        f'Need 1m preferred or 5m fallback; available={sorted(streams)}'
    )


def load_or_build_canonical_streams(
    data_glob: str,
    cache_path: str | Path | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Step 2 artifact boundary: schema/session-normalized canonical streams.

    Produces one parquet per configured frequency:
      canonical_data_<tag>_1m.parquet
      canonical_data_<tag>_1h.parquet
      canonical_data_<tag>_1d.parquet

    Continuous-contract fields and HTF joins are intentionally not included.
    """
    cache_files = _canonical_cache_files(cache_path) if cache_path else {}
    if cache_files and all(p.exists() for p in cache_files.values()):
        print(f'[CANONICAL] Loading canonical streams from cache: {cache_path}', flush=True)
        streams = {freq: pl.read_parquet(path) for freq, path in cache_files.items()}
        for freq, df in list(streams.items()):
            streams[freq] = validate_canonical_stream(df, freq)
        return streams

    print(f'[CANONICAL] Building canonical streams from raw: {data_glob}', flush=True)
    streams = load_all_streams_chunked(data_glob)

    from pipeline.session.gap_filter import filter_gaps
    primary_freq = _primary_bar_freq(streams)
    streams[primary_freq] = filter_gaps(streams[primary_freq], max_gap_minutes=30)

    for freq, df in list(streams.items()):
        streams[freq] = validate_canonical_stream(df, freq)

    if cache_files:
        for freq, df in streams.items():
            path = cache_files.get(freq)
            if path is None:
                continue
            print(f'[CANONICAL] Caching {freq} stream to {path}', flush=True)
            write_canonical_parquet(df, path)

    return streams


def validate_aligned_continuous(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in _ALIGNED_CONTINUOUS_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'ALIGNED/CONTINUOUS FAIL: missing columns {missing}')
    validate_memory_and_integrity(df)
    if df['ts_event'].n_unique() != df.height:
        raise ValueError('ALIGNED/CONTINUOUS FAIL: duplicate ts_event values')
    null_cols = [
        c for c in _ALIGNED_CONTINUOUS_REQUIRED_COLS
        if c in df.columns and df[c].null_count() > 0
    ]
    if null_cols:
        raise ValueError(f'ALIGNED/CONTINUOUS FAIL: nulls in {null_cols}')
    if (df['contract_multiplier'] <= 0).any():
        raise ValueError('ALIGNED/CONTINUOUS FAIL: non-positive contract_multiplier')
    if (df['continuous_price'] <= 0).any():
        raise ValueError('ALIGNED/CONTINUOUS FAIL: non-positive continuous_price')
    if 'cumulative_factor' in df.columns:
        max_err = (
            df.select(
                ((pl.col('continuous_price') - (pl.col('close') * pl.col('cumulative_factor'))).abs())
                .max()
                .alias('max_err')
            )['max_err'][0]
        )
        if max_err is not None and max_err > 1e-3:
            raise ValueError(
                f'ALIGNED/CONTINUOUS FAIL: continuous_price != close*cumulative_factor '
                f'(max_err={max_err})'
            )
    for c in ('continuous_open', 'continuous_high', 'continuous_low'):
        if (df[c] <= 0).any():
            raise ValueError(f'ALIGNED/CONTINUOUS FAIL: non-positive {c}')
    if 'front_contract' in df.columns and 'back_contract' in df.columns:
        # If raw contract identifiers exist, roll metadata must not be fully null.
        if df['front_contract'].null_count() == df.height and df['back_contract'].null_count() == df.height:
            raise ValueError('ALIGNED/CONTINUOUS FAIL: raw roll contract metadata is fully null')
    return df


def load_or_build_aligned_continuous_data(
    data_glob: str,
    aligned_cache_path: str | Path | None = None,
    canonical_cache_path: str | Path | None = None,
    cross_asset_symbols: list | None = None,
) -> pl.DataFrame:
    """
    Step 3 artifact boundary: HTF-aligned, continuous-contract dataset.

    Input: Step 2 canonical streams.
    Output: one cached aligned/continuous parquet.
    """
    if aligned_cache_path and Path(aligned_cache_path).exists():
        print(f'[ALIGNED] Loading aligned/continuous data from cache: {aligned_cache_path}', flush=True)
        df_cached = pl.read_parquet(aligned_cache_path)
        return validate_aligned_continuous(df_cached)

    print('[ALIGNED] Building aligned/continuous data from canonical streams.', flush=True)
    streams = load_or_build_canonical_streams(data_glob, cache_path=canonical_cache_path)
    primary_freq = _primary_bar_freq(streams)
    df_primary = streams[primary_freq]
    df_1h = streams.get('1h')
    df_daily = streams.get('1d')

    print('[ALIGNED] Aligning HTF streams...', flush=True)
    df_aligned = align_htf_streams(df_primary, df_1h, df_daily)
    validate_memory_and_integrity(df_aligned)

    symbol = detect_symbol_from_path(data_glob)
    print(f'[ALIGNED] Building continuous contract series for {symbol}...', flush=True)
    load_market_config(symbol)
    contract_multiplier = get_contract_multiplier(symbol)
    df_aligned = build_continuous_series(
        df_aligned, symbol, contract_multiplier=contract_multiplier
    )

    if cross_asset_symbols:
        df_aligned = _join_cross_asset_features(df_aligned, cross_asset_symbols, data_glob)

    df_aligned = validate_aligned_continuous(df_aligned)

    if aligned_cache_path:
        print(f'[ALIGNED] Caching aligned/continuous data to {aligned_cache_path}', flush=True)
        logger.info(f'Caching aligned/continuous data to {aligned_cache_path}')
        write_canonical_parquet(df_aligned, aligned_cache_path)

    return df_aligned


def _load_cross_asset_feature(secondary_symbol: str, primary_path: Path) -> pl.DataFrame:
    secondary_glob = str(primary_path.parent.parent / secondary_symbol / primary_path.name)
    print(
        f'[INGEST] Loading cross-asset features for {secondary_symbol} from {secondary_glob}',
        flush=True,
    )
    try:
        streams = load_all_streams_chunked(secondary_glob)
        primary_freq = _primary_bar_freq(streams)
        df_primary = streams[primary_freq]
        df_primary = df_primary.with_columns(
            (pl.col('close') / pl.col('close').shift(1)).log().alias(
                f'{secondary_symbol}_ret_1'
            )
        )
        df_primary = df_primary.select(['ts_event', f'{secondary_symbol}_ret_1'])
        logger.info(
            f'Loaded cross-asset features for {secondary_symbol}, {df_primary.height} rows'
        )
        return df_primary
    except Exception as e:
        logger.warning(
            f'Could not load cross-asset features for {secondary_symbol}: {e}'
        )
        return pl.DataFrame()


def _join_cross_asset_features(
    df_aligned: pl.DataFrame,
    cross_asset_symbols: list,
    data_glob: str,
) -> pl.DataFrame:
    primary_path = Path(data_glob)

    cross_frames = []
    for sym in cross_asset_symbols:
        frame = _load_cross_asset_feature(sym, primary_path)
        if not frame.is_empty():
            cross_frames.append(frame)

    if not cross_frames:
        return df_aligned

    cross_combined = cross_frames[0]
    for frame in cross_frames[1:]:
        cross_combined = cross_combined.join(frame, on='ts_event', how='outer')
        right_col = 'ts_event_right'
        if right_col in cross_combined.columns:
            cross_combined = cross_combined.drop(right_col)

    df_aligned = df_aligned.join(cross_combined, on='ts_event', how='left')

    cross_cols = [c for c in cross_combined.columns if c != 'ts_event']
    if cross_cols:
        if 'session_id' in df_aligned.columns:
            df_aligned = df_aligned.with_columns(
                [pl.col(c).fill_null(strategy='forward').over('session_id')
                 for c in cross_cols]
            )
        else:
            df_aligned = df_aligned.with_columns(
                [pl.col(c).fill_null(strategy='forward') for c in cross_cols]
            )

    return df_aligned


def load_and_clean_data(
    data_glob: str,
    cache_path: str = None,
    canonical_cache_path: str = None,
    cross_asset_symbols: list = None,
) -> pl.DataFrame:
    df_aligned = load_or_build_aligned_continuous_data(
        data_glob,
        aligned_cache_path=cache_path,
        canonical_cache_path=canonical_cache_path,
        cross_asset_symbols=cross_asset_symbols,
    )

    if config.MEMORY_LOG_ENABLED:
        logger.info(
            f'RSS after load: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB'
        )
    return df_aligned
