import os
import sys

# ── UTF-8 everywhere (Windows hardening) ──
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import subprocess
import logging
import time
import json
import hashlib
import threading
import re
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import polars as pl
import numpy as np
from pipeline.common.config import load_config, RootConfig, config as _ns_cfg
from pipeline.common.market import get_contract_multiplier
from pipeline.data_gate.manifest import DatasetGateError, validate_dataset_gate
from pipeline.data_gate.preflight import DatasetPreflightError, validate_research_data_preflight
from pipeline.data_gate.checkpoint import validate_checkpoint_stage
from pipeline.data.classify_checkpoint import canonical_stage, classify_checkpoint
from pipeline.orchestration.stage_plan import START_STAGE_NUM, build_stage_plan, normalize_start_stage
from pipeline.audit.pipeline_coverage import stage_catalog
from pipeline.audit.run_manifest import write_run_manifest
from pipeline.gates.deployment import run_deployment_readiness_gate
from pipeline.walkforward.split_plan import write_wfa_split_plan

_LOG_MODE = os.environ.get('LOG_MODE', 'clean').strip().lower()
if _LOG_MODE not in {'clean', 'verbose', 'debug'}:
    _LOG_MODE = 'clean'
_DEBUG = _LOG_MODE == 'debug'
_VERBOSE = _LOG_MODE in {'verbose', 'debug'} or os.environ.get('QUANT_VERBOSE', '0') == '1'

logging.basicConfig(
    level=logging.DEBUG if _DEBUG else (logging.INFO if _VERBOSE else logging.ERROR),
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8',
    force=True,
)
logger = logging.getLogger('QuantRunner')
_MIN_TRAIN_DAYS = 90
_RUN_START = time.time()
_RUN_DT = datetime.now()
_RUN_FILE_TS = _RUN_DT.strftime('%Y-%m-%d_%H-%M-%S')
_RUN_ID = hashlib.sha256(str(_RUN_START).encode()).hexdigest()[:8]
_PER_SYMBOL_PNL_CS = {}  # {symbol: {split_id: pnl_cs}}
_VERIFICATION_TABLE = []  # rows for the verification table printed per split
_RUN_SPLIT_ARTIFACTS = []


class PipelineProgressLogger:
    # Per-split runtime diagnostics. This intentionally tracks what the runner
    # can observe from subprocess logs; it is close to, but not a strict copy of,
    # project_layout.md.
    STAGES = {
        1: 'RAW DATA',
        2: 'INGEST / CANONICAL',
        3: 'CONTINUOUS CONTRACTS',
        4: 'SESSION NORMALIZATION',
        5: 'ALIGNMENT',
        6: 'FEATURES / MANIFEST',
        7: 'TARGET CONTRACT',
        8: 'REGIME / HMM',
        9: 'META LABEL',
        10: 'EXECUTION',
        11: 'WALKFORWARD / OOS',
        12: 'METRICS',
        13: 'ARTIFACTS',
    }
    ERROR_PATTERNS = (
        'Traceback', 'RuntimeError', 'ValueError', 'Exception',
        '[TIMEOUT]', '[ERROR]', 'failed', 'FAILED', 'rc!=0',
    )

    def __init__(self, mode: str, run_id: str):
        self.mode = mode
        self.run_id = run_id
        self.context = {}
        self.stage_state = {}
        self.flushed = False
        self.stage_start = time.time()
        self.log_path = Path('output') / 'logs' / f'{_RUN_FILE_TS}_UNKNOWN_{run_id}.log'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, 'a', encoding='utf-8')

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip())
        return cleaned.strip('._-') or 'unknown'

    def rename_for_config(self, profile: str, symbols: list[str]) -> None:
        symbols_safe = '-'.join(self._safe_name(s).upper() for s in symbols) if symbols else 'unknown'
        profile_safe = self._safe_name(profile)
        dest = self.log_path.parent / f'{_RUN_FILE_TS}_{symbols_safe}_{profile_safe}_{self.run_id}.log'
        if dest == self.log_path:
            return
        if dest.exists():
            self._fh.close()
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as src, open(dest, 'a', encoding='utf-8') as out:
                out.write(src.read())
            self.log_path.unlink(missing_ok=True)
            self.log_path = dest
            self._fh = open(self.log_path, 'a', encoding='utf-8')
            return
        old_path = self.log_path
        try:
            self._fh.close()
            old_path.replace(dest)
            self.log_path = dest
            self._fh = open(self.log_path, 'a', encoding='utf-8')
        except Exception as e:
            self.log_path = old_path
            self._fh = open(self.log_path, 'a', encoding='utf-8')
            self.raw(f'[LOG] rename_failed target={dest} error={e}; keeping={self.log_path}')

    @property
    def clean(self) -> bool:
        return self.mode == 'clean'

    @property
    def verbose(self) -> bool:
        return self.mode in {'verbose', 'debug'} or _VERBOSE

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def set_context(self, **kwargs) -> None:
        self.context = {k: v for k, v in kwargs.items() if v is not None}
        self.stage_state = {}
        self.flushed = False
        self.stage_start = time.time()

    def raw(self, text: str) -> None:
        self._fh.write(text + '\n')
        self._fh.flush()
        if self.verbose:
            print(text, flush=True)

    def stage(self, idx: int, message: str, once: bool = False) -> None:
        if once and idx in self.stage_state:
            return
        elapsed = time.time() - self.stage_start
        self.stage_state[idx] = (message, elapsed)

    def flush_stages(self) -> None:
        if self.flushed:
            return
        for idx in range(1, 14):
            message, elapsed = self.stage_state.get(idx, ('not_observed', time.time() - self.stage_start))
            self._print_stage(idx, message, elapsed)
        self.flushed = True

    def _print_stage(self, idx: int, message: str, elapsed: float) -> None:
        lines = str(message).splitlines() or ['']
        print(f'[{idx:02d}/13 {self.STAGES[idx]}] {lines[0]} elapsed={elapsed:.1f}s', flush=True)
        for line in lines[1:]:
            print(line, flush=True)

    def child_line(self, prefix: str, line: str) -> None:
        raw = f'[{prefix}] {line}'
        self._fh.write(raw + '\n')
        self._fh.flush()
        if self.verbose:
            print(raw, flush=True)
            return
        if any(p in line for p in self.ERROR_PATTERNS):
            self.context.setdefault('child_errors', []).append(line)
            return
        self._summarize_child_line(line)

    def timeout(self, message: str) -> None:
        self.raw(message)
        if self.clean:
            print(message, flush=True)

    def _summarize_child_line(self, line: str) -> None:
        m = re.search(r'\[CLI\] Data loaded\. Rows: ([\d,]+)', line)
        if m:
            self.stage(2, f'rows_out={m.group(1)}')
            self._stage_contract_once()
            return
        m = re.search(r'\[CLI\] Aligned data: ([\d,]+) rows', line)
        if m:
            if 2 not in self.stage_state:
                self.stage(2, f'cache_or_ingest rows_out={m.group(1)}', once=True)
                self._stage_contract_once()
            self.stage(5, f'rows_out={m.group(1)}', once=True)
            return
        m = re.search(r'\[SESSION\] (\S+) stream has ([\d,]+) rows', line)
        if m:
            self.stage(4, f'freq={m.group(1)} rows={m.group(2)}')
            return
        if '[INGEST] No cache found' in line or '[INGEST] Loading aligned data from cache' in line:
            self.stage(2, 'loading aligned data')
            return
        if '[INGEST] Aligning HTF streams' in line:
            self._stage_contract_once()
            self.stage(5, 'aligning HTF streams', once=True)
            return
        m = re.search(r'\[CLI\] Date filter \(([^)]*)\): ([\d,]+) -> ([\d,]+) rows', line)
        if m:
            self.stage(5, f'window={m.group(1)} rows_in={m.group(2)} rows_out={m.group(3)}')
            return
        m = re.search(r'\[CLI\] Feature matrix: ([\d,]+) rows, ([\d,]+) cols', line)
        if m:
            self.context['feature_total'] = m.group(2)
            self.stage(6, f'rows={m.group(1)} total={m.group(2)}')
            return
        m = re.search(r'\[CLI\] After manifest: ([\d,]+) rows, ([\d,]+) cols', line)
        if m:
            total = self.context.get('feature_total', 'NA')
            self.stage(6, f'selected={m.group(2)} total={total} rows={m.group(1)}')
            return
        m = re.search(r'\[TARGET\] ([^:]+): rows=([\d,]+) NaN=([\d,]+)', line)
        if m:
            self.stage(7, f'target={m.group(1)} rows={m.group(2)} nan={m.group(3)}')
            return
        m = re.search(r'\[HMM-TIMING\] step=hmm_features rows=([\d,]+) cols=([\d,]+)', line)
        if m:
            self.stage(8, f'HMM rows={m.group(1)} features={m.group(2)}')
            return
        m = re.search(r'\[HMM-TIMING\] iter=([^ ]+)', line)
        if m:
            self.stage(8, f'HMM iter={m.group(1)}')
            return
        if '[CLI] HMM FALLBACK ACTIVE' in line:
            self.stage(8, 'fallback active')
            return
        if '[CLI] Running HMM-aware walkforward' in line or '[CLI] Running walkforward' in line:
            self.stage(9, 'active' if 'meta' in line.lower() else 'skipped')
            return
        m = re.search(r'\[OUTER-TRUE\] train filter .* -> ([\d,]+) rows', line)
        if m:
            self.stage(11, f'train_rows={m.group(1)}')
            return
        m = re.search(r'\[OUTER-TRUE\] test filter .* -> ([\d,]+) rows', line)
        if m:
            self.stage(11, f'test_rows={m.group(1)}')
            return
        m = re.search(r'\[HEARTBEAT\] train rows=([\d,]+) test rows=([\d,]+)', line)
        if m:
            self.stage(11, f'train_rows={m.group(1)} test_rows={m.group(2)}')
            return
        m = re.search(r'\[OUTER-TRUE(?:-HMM)?\] train_rows=([\d,]+)', line)
        if m:
            self.context['train_rows'] = m.group(1)
            self._stage_walkforward_rows()
            return
        m = re.search(r'\[OUTER-TRUE(?:-HMM)?\] test_rows=([\d,]+)', line)
        if m:
            self.context['test_rows'] = m.group(1)
            self._stage_walkforward_rows()
            return
        m = re.search(r'\[OUTER-TRUE(?:-HMM)?\] feature_cols=([\d,]+)', line)
        if m:
            self.stage(6, f'selected={m.group(1)} total={self.context.get("feature_total", "NA")}')
            return
        m = re.search(r'\[CLI\] HMM walkforward result: ([\d,]+) rows', line)
        if m:
            self.stage(11, f'output_rows={m.group(1)}')
            return
        m = re.search(r'\[CLI\] Walkforward result: ([\d,]+) rows', line)
        if m:
            self.stage(11, f'output_rows={m.group(1)}')
            return
        if '[CLI] Running aggregation' in line:
            self.stage(12, 'aggregation')
            return
        m = re.search(r'(?:HMM-filtered results|Results) saved to (.+)', line)
        if m:
            self.stage(13, f'saved={m.group(1)}')

    def _stage_contract_once(self) -> None:
        msg = self.context.get('contract_summary')
        if msg and 3 not in self.stage_state:
            self.stage(3, msg, once=True)

    def _stage_walkforward_rows(self) -> None:
        train_rows = self.context.get('train_rows')
        test_rows = self.context.get('test_rows')
        if train_rows and test_rows:
            self.stage(11, f'train_rows={train_rows} test_rows={test_rows}')


_PROGRESS = PipelineProgressLogger(_LOG_MODE, _RUN_ID)


def _stage(idx: int, message: str) -> None:
    _PROGRESS.stage(idx, message)


def _raw_log(text: str) -> None:
    _PROGRESS.raw(text)


def get_files(data_dir: str, config: RootConfig) -> list[Path]:
    files = list(Path(data_dir).rglob('*.parquet'))
    allowed_markets = set(config.symbols)
    valid = []
    for f in files:
        try:
            year = int(f.stem)
        except Exception:
            continue
        if year < config.start_year:
            continue
        if year > config.end_year:
            continue
        if allowed_markets and f.parent.name not in allowed_markets:
            continue
        valid.append(f)
    valid = sorted(valid, key=lambda x: (x.parent.name, x.stem))
    if config.io.max_files:
        valid = valid[:config.io.max_files]
    return valid

def _load_file_date_bounds(files: list[Path]) -> dict[int, tuple]:
    """
    Scan each yearly parquet file to extract its (min_date, max_date) range.
    Uses lazy scanning to avoid loading full datasets into memory.
    Returns dict mapping file_stem_year -> (date_min, date_max).
    """
    import datetime as _dt
    bounds: dict[int, tuple] = {}
    for f in files:
        try:
            year = int(f.stem)
        except ValueError:
            continue
        try:
            date_bounds = (
                pl.scan_parquet(f)
                .select([
                    pl.col('ts_event').min().alias('min_ts'),
                    pl.col('ts_event').max().alias('max_ts'),
                ])
                .collect()
            )
            min_ts = date_bounds['min_ts'][0]
            max_ts = date_bounds['max_ts'][0]
            if min_ts is not None and max_ts is not None:
                bounds[year] = (min_ts.date(), max_ts.date())
        except Exception:
            logger.debug('Could not read date bounds from %s — skipping', f)
    return bounds


def generate_walkforward_splits(files: list[Path], config: RootConfig) -> list[tuple[list[int], list[int]]]:
    """
    Generate walk-forward split index pairs.

    When the walkforward config defines positive *day-based* parameters
    (wf_train_days / wf_test_days / wf_step_days), each split is a
    rolling calendar-day window anchored to the full date range of the
    available data.  Files are assigned to each window based on whether
    their date span overlaps the train or test interval.

    Otherwise falls back to legacy year-count splitting.
    """
    import datetime as _dt

    wf_cfg = config.walkforward

    # ---- Day-based rolling-window mode ------------------------------------
    day_based = (
        wf_cfg.wf_train_days > 0
        and wf_cfg.wf_test_days > 0
        and wf_cfg.wf_step_days > 0
    )

    if day_based:
        file_bounds = _load_file_date_bounds(files)
        if not file_bounds:
            logger.warning('No file date bounds available — returning empty splits')
            return []

        all_dates = sorted({
            d for b in file_bounds.values()
            for d in (b[0], b[1])
        })
        if not all_dates:
            logger.warning('No dates extracted from files')
            return []

        data_start = all_dates[0]
        data_end = all_dates[-1]
        total_days = (data_end - data_start).days + 1
        window = wf_cfg.wf_train_days + wf_cfg.wf_test_days

        if total_days < window:
            logger.warning(
                'Insufficient date range for day-based walkforward: '
                'need at least %d days, have %d days (%s → %s)',
                window, total_days, data_start, data_end,
            )
            # Single split covering the full range — temporal separation
            # is enforced inside cli.py via --start/--end date boundaries
            # which slice by actual timestamps, not file boundaries.
            all_years = sorted(file_bounds.keys())
            return [(all_years, all_years)]

        splits: list[tuple[list[int], list[int]]] = []
        cursor = 0  # day offset from data_start

        while cursor + window <= total_days:
            train_start = data_start + _dt.timedelta(days=cursor)
            train_end   = data_start + _dt.timedelta(days=cursor + wf_cfg.wf_train_days)
            test_start  = train_end
            test_end    = data_start + _dt.timedelta(days=cursor + window)

            # File assignment by date overlap — yearly parquet files span
            # full calendar years. Overlap-based assignment is safe because
            # cli.py enforces strict temporal separation by passing
            # --start/--end date boundaries, slicing within each file at
            # the timestamp level.
            train_files = [
                yr for yr, (fmin, fmax) in file_bounds.items()
                if fmax >= train_start and fmin < train_end
            ]
            test_files = [
                yr for yr, (fmin, fmax) in file_bounds.items()
                if fmax >= test_start and fmin < test_end
            ]

            if train_files and test_files:
                splits.append((sorted(train_files), sorted(test_files),
                               train_start, train_end, test_start, test_end))
                logger.debug(
                    'Split %d: train=%s→%s (%d files) | test=%s→%s (%d files)',
                    len(splits),
                    train_start.isoformat(), train_end.isoformat(), len(train_files),
                    test_start.isoformat(), test_end.isoformat(), len(test_files),
                )

            cursor += wf_cfg.wf_step_days
            # Safety guard for very large step counts (prevent accidental infinite loop)
            if len(splits) >= 10_000:
                logger.warning('Split limit 10,000 reached — truncating')
                break

        if splits:
            logger.info(
                'Generated %d walkforward splits (day-based: train=%dd, test=%dd, step=%dd, '
                'range=%s→%s)',
                len(splits),
                wf_cfg.wf_train_days, wf_cfg.wf_test_days, wf_cfg.wf_step_days,
                data_start.isoformat(), data_end.isoformat(),
            )
            return splits

        # No valid rolling windows produced — fall back to single split.
        all_years = sorted(file_bounds.keys())
        logger.warning(
            'No day-based splits generated (step %d may be too large for '
            'the %d-day range). Falling back to single merged split.',
            wf_cfg.wf_step_days, total_days,
        )
        return [(all_years, all_years)]

    # ---- Legacy year-count mode ------------------------------------------
    train_years = config.data_years
    wf_years = config.folds
    years = sorted({int(f.stem) for f in files})

    if not years:
        logger.warning('No years available for walkforward splitting')
        return []

    min_years_needed = train_years + wf_years

    splits = []
    for i in range(len(years)):
        train_start = i
        train_end = i + train_years
        test_end = train_end + wf_years
        if test_end > len(years):
            break
        train_range = years[train_start:train_end]
        test_range = years[train_end:test_end]
        splits.append((train_range, test_range))

    if not splits:
        train_range = years[:train_years] if len(years) >= train_years else years[:-1] if len(years) > 1 else []
        test_range = years[train_years:] if len(years) > train_years else []
        if not train_range or not test_range:
            logger.error(
                'Insufficient years for walkforward: need at least %d years '
                '(train=%d + test=%d), have %d. No splits generated.',
                min_years_needed, train_years, wf_years, len(years),
            )
            return []
        logger.warning(
            'Insufficient years for walkforward: need %d (train=%d + test=%d), '
            'have %d. Falling back to reduced split: train=%s, test=%s',
            min_years_needed, train_years, wf_years, len(years),
            train_range, test_range,
        )
        splits.append((train_range, test_range))

    logger.info('Generated %d walkforward splits', len(splits))
    return splits


_MIN_ROWS = 1000
_MIN_PRICE = {'CL': 0.1, 'ES': 500, 'MES': 500, 'NQ': 1000, 'GC': 500, 'SI': 5, 'HG': 1,
              'ZB': 50, 'ZN': 50, 'ZC': 50, 'NG': 0.5}
_MAX_PRICE = {'CL': 200, 'ES': 10000, 'MES': 10000, 'NQ': 30000, 'GC': 5000, 'SI': 60, 'HG': 10,
              'ZB': 250, 'ZN': 250, 'ZC': 20, 'NG': 30}


def _validate_symbol_data(f: Path, config) -> bool:
    symbol = f.parent.name
    try:
        stats = (
            pl.scan_parquet(f)
            .select([
                pl.len().alias('rows'),
                pl.col('ts_event').null_count().alias('null_ts'),
                pl.col('close').mean().alias('close_mean'),
                pl.col('ts_event').min().alias('ts_min'),
                pl.col('ts_event').max().alias('ts_max'),
            ])
            .collect()
        )
    except Exception as e:
        try:
            schema = pl.scan_parquet(f).collect_schema()
        except Exception:
            schema = {}
        missing = [c for c in ('ts_event', 'close') if c not in schema]
        if missing:
            logger.error('[SAFETY] %s: missing columns %s -- SKIPPING', symbol, ','.join(missing))
            return False
        logger.error('[SAFETY] Cannot read %s: %s -- SKIPPING', f, e)
        return False
    rows = int(stats['rows'][0])
    if rows < _MIN_ROWS:
        logger.error('[SAFETY] %s: %d rows < %d minimum -- SKIPPING', symbol, rows, _MIN_ROWS)
        return False
    null_ts = int(stats['null_ts'][0])
    if null_ts > 0:
        logger.error('[SAFETY] %s: %d null ts_event values -- SKIPPING', symbol, null_ts)
        return False
    close_mean = stats['close_mean'][0]
    lo, hi = _MIN_PRICE.get(symbol, 0), _MAX_PRICE.get(symbol, 1e9)
    if close_mean < lo or close_mean > hi:
        logger.error('[SAFETY] %s: close mean %.2f outside [%.2f, %.2f] -- SKIPPING',
                     symbol, close_mean, lo, hi)
        return False
    ts_min = stats['ts_min'][0]
    ts_max = stats['ts_max'][0]
    if ts_min is None or ts_max is None:
        logger.error('[SAFETY] %s: null ts_event range -- SKIPPING', symbol)
        return False
    available_days = (ts_max - ts_min).days
    wf_window = getattr(config, 'WF_TRAIN_DAYS', 30) + getattr(config, 'WF_TEST_DAYS', 1)
    if available_days < wf_window * 2:
        logger.warning('[SAFETY] %s: %d days available < %d required (2x window) -- may have few folds',
                       symbol, available_days, wf_window * 2)
    return True


def _log_subprocess_failure(cmd: list, returncode: int, stderr_text: str, stdout_text: str,
                            log_dir: Path, symbol: str, split_idx: int, stage: str) -> None:
    """Persist full subprocess failure output to a deterministic log file."""
    import datetime as dt
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = log_dir / f'{timestamp}_{symbol}_split{split_idx}_{stage}_FAIL.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'command: {" ".join(cmd)}\n')
        f.write(f'returncode: {returncode}\n')
        f.write(f'symbol: {symbol}\n')
        f.write(f'split_idx: {split_idx}\n')
        f.write(f'stage: {stage}\n')
        f.write(f'timestamp: {timestamp}\n')
        f.write(f'--- STDOUT ---\n{stdout_text}\n--- STDERR ---\n{stderr_text}\n')


def _run_subprocess_streaming(cmd: list, env: dict, timeout_idle: int = 120) -> tuple:
    """Run subprocess with real-time stdout/stderr streaming and idle timeout.

    Returns (returncode, stdout_text, stderr_text).
    On idle timeout (no output for *timeout_idle* seconds): terminates child,
    prints diagnostics, and returns (-1, stdout, stderr).
    """
    env_out = env.copy()
    env_out['PYTHONUNBUFFERED'] = '1'
    full_cmd = [sys.executable, '-u'] + cmd[1:] if cmd[0] == sys.executable else cmd
    proc = subprocess.Popen(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env_out,
    )
    stdout_lines = []
    stderr_lines = []
    last_output = time.time()
    lock = threading.Lock()
    done = threading.Event()

    def _read_stream(stream, lines_list, prefix):
        nonlocal last_output
        try:
            for line in iter(stream.readline, ''):
                line = line.rstrip('\n\r')
                with lock:
                    lines_list.append(line)
                    last_output = time.time()
                _PROGRESS.child_line(prefix, line)
        except Exception:
            pass

    t_stdout = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, 'CHILD-OUT'), daemon=True)
    t_stderr = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, 'CHILD-ERR'), daemon=True)
    t_stdout.start()
    t_stderr.start()

    # Idle timeout watchdog
    def _watchdog():
        while not done.is_set():
            time.sleep(1)
            with lock:
                idle = time.time() - last_output
            if idle > timeout_idle and not done.is_set():
                _PROGRESS.timeout(f'\n[TIMEOUT] No output for {timeout_idle}s — terminating child')
                _PROGRESS.timeout(f'[TIMEOUT] cmd={" ".join(full_cmd)}')
                _PROGRESS.timeout(f'[TIMEOUT] cwd={os.getcwd()}')
                _PROGRESS.timeout(f'[TIMEOUT] env={os.environ.get("CONFIG_ENV") or os.environ.get("QUANT_ENV", "default")}')
                with lock:
                    if stdout_lines:
                        _PROGRESS.timeout(f'[TIMEOUT] last stdout lines: {stdout_lines[-5:]}')
                    if stderr_lines:
                        _PROGRESS.timeout(f'[TIMEOUT] last stderr lines: {stderr_lines[-5:]}')
                proc.kill()
                done.set()
                break

    t_watch = threading.Thread(target=_watchdog, daemon=True)
    t_watch.start()

    try:
        ret = proc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        proc.kill()
        ret = -1
    done.set()
    t_stdout.join(timeout=5)
    t_stderr.join(timeout=5)
    t_watch.join(timeout=5)
    stdout_text = '\n'.join(stdout_lines)
    stderr_text = '\n'.join(stderr_lines)
    if ret == -1:
        raise subprocess.CalledProcessError(-1, full_cmd, stdout_text, stderr_text)
    if ret != 0:
        raise subprocess.CalledProcessError(ret, full_cmd, stdout_text, stderr_text)
    return ret, stdout_text, stderr_text


_log_failures = os.environ.get('QUANT_LOG_FAILURES', '1') == '1'

def _print_verification_table():
    if not _VERIFICATION_TABLE or not _PROGRESS.verbose:
        return
    print('\n' + '=' * 140)
    print(' VERIFICATION TABLE — each row must have unique path, ts_min, pnl_cs')
    print('=' * 140)
    print(f'{"split":>5} {"sym":>4} {"path":>55} {"mtime":>12} {"run_id":>8} {"rows":>6} {"ts_min":>26} {"ts_max":>26} {"pnl_cs":>8} {"sharpe_ann":>10} {"pnl":>12} {"pos_turn":>10}')
    print('-' * 140)
    for row in _VERIFICATION_TABLE:
        path_short = row['path'][-55:] if len(row['path']) > 55 else row['path']
        print(f'{row["split"]:>5} {row["symbol"]:>4} {path_short:>55} {row["mtime"]:>12.0f} {row["run_id"]:>8} {row["rows"]:>6} {str(row["ts_min"]):>26} {str(row["ts_max"]):>26} {row["pnl_cs"]:>8} {row.get("sharpe_annualized", row.get("sharpe", 0.0)):>+10.3f} {row["pnl"]:>+12.2f} {row.get("position_turnover", row.get("turnover", 0.0)):>10.6g}')
    print('=' * 140)
    # Verify: no duplicate pnl_cs per symbol
    seen = {}
    for row in _VERIFICATION_TABLE:
        key = (row['symbol'], row['pnl_cs'])
        if row['pnl_cs'] not in ('missing', 'all_nan'):
            if key in seen:
                print(f'\n*** IDENTICAL PNL OUTPUT DETECTED: {row["symbol"]} split={row["split"]} matches split={seen[key]["split"]} pnl_cs={row["pnl_cs"]} ***\n', flush=True)
            seen[key] = row
    # Verify: no duplicate paths
    paths_seen = set()
    for row in _VERIFICATION_TABLE:
        if row['path'] in paths_seen:
            print(f'\n*** DUPLICATE RESULT PATH: {row["path"]} split={row["split"]} ***\n', flush=True)
        paths_seen.add(row['path'])
    print(flush=True)


def _print_split_dashboard(split_idx: int, total: int, per_symbol: dict):
    if not _PROGRESS.verbose:
        return
    _print_verification_table()
    if not per_symbol:
        return
    h = '\u2500'
    tl, tr, bl, br, v, c = '\u250c', '\u2510', '\u2514', '\u2518', '\u2502', '\u251c'
    r = '\u2524'
    check, cross, warn = '\u2705', '\u274c', '\u26a0\ufe0f'
    print(f'\n{tl}{h * 52}{tr}')
    print(f'{v} [SPLIT {split_idx}/{total}] Processing Assets...{" " * 3}{v}')
    print(f'{c}{h * 52}{r}')
    combined_sharpe = 0.0
    for symbol, entry in sorted(per_symbol.items()):
        sharpe = entry[0]; pnl = entry[1]; hit = entry[2]
        hmm_delta = entry[3] if len(entry) > 3 else 0.0
        icon = check if sharpe > 0 else cross
        delta_str = f' HMM_delta={hmm_delta:+.2f}' if abs(hmm_delta) > 0.001 else ''
        print(f'{v}  {icon} {symbol:<4s}  SharpeAnn={sharpe:+.2f}  PnL={pnl:+,.2f}  HitActive={hit:.1%}{delta_str} {"":>6s}{v}')
        combined_sharpe += sharpe
    status = f'{check} Success' if combined_sharpe > 0 else f'{warn} Mixed'
    print(f'{v} {"":>24s}Combined Sharpe: {combined_sharpe:+.3f}  Status: {status}{v}')
    print(f'{bl}{h * 52}{br}\n', flush=True)


def _print_final_summary_table() -> None:
    print('\nsymbol | split | sharpe_annualized | bars_per_year | pnl | ic | bar_hit_rate_active_bars | position_turnover | pred_cs | pnl_cs | status')
    print('-' * 132)
    if not _VERIFICATION_TABLE:
        print('NA | NA | NA | NA | NA | NA | NA | NA | missing | missing | NO_RESULTS', flush=True)
        return
    for row in sorted(_VERIFICATION_TABLE, key=lambda r: (str(r.get('symbol')), int(r.get('split', 0)))):
        print(
            f"{row.get('symbol', 'NA')} | {row.get('split', 'NA')} | "
            f"{row.get('sharpe_annualized', row.get('sharpe', float('nan'))):+.3f} | "
            f"{row.get('bars_per_year', 'NA')} | "
            f"{row.get('pnl', 0.0):+.2f} | "
            f"{row.get('ic', 'NA')} | "
            f"{row.get('bar_hit_rate_active_bars', row.get('hit_rate', row.get('hit', 0.0))):.2%} | "
            f"{row.get('position_turnover', row.get('turnover', 0.0)):.6g} | "
            f"{row.get('pred_cs', 'missing')} | {row.get('pnl_cs', 'missing')} | "
            f"{row.get('status', 'OK')}",
            flush=True,
        )


def _extract_error_summary(*texts) -> str:
    patterns = (
        r'^\s*([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):\s*.+)$',
        r'^\s*(RuntimeError:\s*.+)$',
        r'^\s*(ValueError:\s*.+)$',
        r'^\s*(NameError:\s*.+)$',
    )
    lines = []
    for text in texts:
        if not text:
            continue
        if isinstance(text, bytes):
            text = text.decode(errors='replace')
        lines.extend(str(text).splitlines())
    for line in reversed(lines):
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                return m.group(1).strip()
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line[:240]
    return 'unknown error'


def _tail_text(text, max_lines: int = 40, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    lines = str(text).splitlines()
    tail = "\n".join(lines[-max_lines:])
    return tail[-max_chars:]


def _exception_parts(message: str | None) -> tuple[str, str]:
    msg = str(message or "")
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(.*)$", msg)
    if m:
        return m.group(1), m.group(2)
    return "", msg


def _print_split_result(row: dict) -> None:
    print('\n[SPLIT RESULT]', flush=True)
    print(f"symbol={row.get('symbol', 'NA')}", flush=True)
    print(f"split={row.get('split', 'NA')}", flush=True)
    print(f"sharpe_annualized={row.get('sharpe_annualized', row.get('sharpe', 0.0)):.3f}", flush=True)
    print(f"sharpe_per_bar={row.get('sharpe_per_bar', 0.0):.6f}", flush=True)
    print(f"bars_per_year={row.get('bars_per_year', 'NA')}", flush=True)
    print(f"pnl={row.get('pnl', 0.0):.2f}", flush=True)
    print(f"ic={row.get('ic', 'NA')}", flush=True)
    print(f"bar_hit_rate_all_bars={row.get('bar_hit_rate_all_bars', 0.0):.2%} denominator={row.get('bar_hit_rate_all_bars_n', 0)}", flush=True)
    print(f"bar_hit_rate_active_bars={row.get('bar_hit_rate_active_bars', 0.0):.2%} denominator={row.get('bar_hit_rate_active_bars_n', 0)}", flush=True)
    print(f"trade_hit_rate={row.get('trade_hit_rate', 'NA')} denominator={row.get('trade_hit_rate_n', 0)}", flush=True)
    print(f"position_change_events={row.get('position_change_events', row.get('trades', 0))}", flush=True)
    print(f"position_turnover={row.get('position_turnover', row.get('turnover', 0.0)):.6g}", flush=True)
    print(f"status={row.get('status', 'OK')}", flush=True)


def _print_clean_split_result(row: dict) -> None:
    status = row.get('status', 'OK')
    symbol = row.get('symbol', 'NA')
    split = row.get('split', 'NA')
    if status != 'OK':
        error = row.get('error') or 'see raw log'
        print(f'[FAILED] {symbol} split={split} {error}', flush=True)
        return
    print(
        f"[RESULT] {symbol} split={split} "
        f"sharpe_ann={row.get('sharpe_annualized', row.get('sharpe', 0.0)):+.3f} "
        f"pnl={row.get('pnl', 0.0):+.2f} "
        f"ic={row.get('ic', 'NA')} "
        f"hit_active={row.get('bar_hit_rate_active_bars', row.get('hit_rate', 0.0)):.2%} "
        f"turnover={row.get('position_turnover', row.get('turnover', 0.0)):.6g} "
        f"trades={row.get('position_change_events', row.get('trades', 0))} "
        f"pred_cs={row.get('pred_cs', 'missing')} "
        f"pnl_cs={row.get('pnl_cs', 'missing')}",
        flush=True,
    )


def _record_split_result(row: dict) -> None:
    _upsert_row_by_symbol_split(_VERIFICATION_TABLE, row)


def _upsert_row_by_symbol_split(rows: list[dict], row: dict) -> None:
    key = (str(row.get("symbol")), int(row.get("split", 0)))
    for idx, existing in enumerate(rows):
        if (str(existing.get("symbol")), int(existing.get("split", 0))) == key:
            rows[idx] = {**existing, **row}
            return
    rows.append(row)


def _safe_window_name(start, end) -> str:
    raw = "_".join(str(x or "none") for x in [start, end])
    return "".join(ch if ch.isalnum() else "-" for ch in raw)[:80]


def _read_report_status(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("status", "UNKNOWN")
    except Exception:
        return "MISSING"


def _expected_artifact_paths(symbol: str, split_idx: int, command: str, out_dir: Path, test_start, test_end) -> dict:
    window = _safe_window_name(
        test_start.isoformat() if test_start else None,
        test_end.isoformat() if test_end else None,
    )
    profile = getattr(_ns_cfg, "ACTIVE_PROFILE", "profile")
    return {
        "backtest_results": str(out_dir / ("backtest_results_hmm.parquet" if command == "run-hmm" else "backtest_results.parquet")),
        "oos_predictions": str(out_dir / "oos_predictions.parquet"),
        "leakage_report": str(Path("reports") / "leakage" / f"{profile}_{symbol}_{command}_{window}.json"),
        "execution_trace_report": str(out_dir / "execution_trace_report.json"),
        "metrics_report": str(Path("reports") / "metrics" / f"{profile}_{symbol}_{command}_{window}_metrics_report.json"),
        "stress_report": str(Path("reports") / "stress" / f"{profile}_{symbol}_{command}_{window}_stress_report.json"),
        "acceptance_report": str(Path("reports") / "acceptance" / f"{profile}_{symbol}_{command}_{window}_acceptance_gate.json"),
    }


def _record_split_artifacts(
    symbol: str,
    split_idx: int,
    command: str,
    out_dir: Path,
    train_start,
    train_end,
    test_start,
    test_end,
    status: str,
    error: str | None = None,
    *,
    rc: int | None = None,
    command_line: str | None = None,
    stdout_tail: str | None = None,
    stderr_tail: str | None = None,
) -> None:
    paths = _expected_artifact_paths(symbol, split_idx, command, out_dir, test_start, test_end)
    exception_type, exception_message = _exception_parts(error)
    row = {
        "split": split_idx,
        "symbol": symbol,
        "train_start": train_start.isoformat() if train_start else None,
        "train_end": train_end.isoformat() if train_end else None,
        "test_start": test_start.isoformat() if test_start else None,
        "test_end": test_end.isoformat() if test_end else None,
        "cli_command": command,
        "status": status,
        "error": error,
        "rc": rc,
        "command": command_line or command,
        "stdout_tail": stdout_tail or "",
        "stderr_tail": stderr_tail or "",
        "exception_type": exception_type,
        "exception_message": exception_message,
        **paths,
        "leakage_status": _read_report_status(Path(paths["leakage_report"])),
        "execution_trace_status": _read_report_status(Path(paths["execution_trace_report"])),
        "metrics_status": "PASS" if Path(paths["metrics_report"]).exists() else "MISSING",
        "stress_status": _read_report_status(Path(paths["stress_report"])),
        "acceptance_status": _read_report_status(Path(paths["acceptance_report"])),
    }
    _upsert_row_by_symbol_split(_RUN_SPLIT_ARTIFACTS, row)


def _expected_runtime_rows(config: RootConfig, splits: list, files: list[Path]) -> list[dict]:
    rows = []
    command = "run-hmm" if bool(getattr(getattr(config, "hmm", None), "enabled", False)) else "run"
    profile = getattr(_ns_cfg, "ACTIVE_PROFILE", "profile")
    for split_idx, split_data in enumerate(splits, 1):
        test_years = split_data[1]
        train_start = split_data[2] if len(split_data) > 2 else None
        train_end = split_data[3] if len(split_data) > 3 else None
        test_start = split_data[4] if len(split_data) > 4 else None
        test_end = split_data[5] if len(split_data) > 5 else None
        test_files = [f for f in files if int(f.stem) in test_years]
        for symbol in config.symbols:
            symbol_files = sorted([f for f in test_files if f.parent.name == symbol])
            if symbol_files:
                test_file = symbol_files[0]
                test_year_label = "_".join(str(f.stem) for f in symbol_files)
                out_dir = Path("output") / symbol / f"{test_year_label}_split_{split_idx}_{profile}"
            else:
                test_file = None
                out_dir = Path("output") / symbol / f"missing_test_file_split_{split_idx}_{profile}"
            paths = _expected_artifact_paths(symbol, split_idx, command, out_dir, test_start, test_end)
            rows.append(
                {
                    "split": split_idx,
                    "symbol": symbol,
                    "test_file": str(test_file) if test_file else "",
                    "missing_test_file": test_file is None,
                    "train_start": train_start.isoformat() if train_start else None,
                    "train_end": train_end.isoformat() if train_end else None,
                    "test_start": test_start.isoformat() if test_start else None,
                    "test_end": test_end.isoformat() if test_end else None,
                    "cli_command": command,
                    "out_dir": str(out_dir),
                    "test_file_count": len(symbol_files),
                    "test_year_label": "_".join(str(f.stem) for f in symbol_files),
                    **paths,
                }
            )
    return rows


def _assert_execution_plan_unique(plan_rows: list[dict], symbols: list[str], splits: list) -> None:
    expected = len(symbols) * len(splits)
    assert len(plan_rows) == expected, (
        f"SPLIT PLAN ROW COUNT FAIL: rows={len(plan_rows)} expected={expected} "
        f"symbols={symbols} splits={len(splits)}"
    )
    keys = [(r.get("symbol"), int(r.get("split", 0))) for r in plan_rows]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert len(keys) == len(set(keys)), f"SPLIT PLAN DUPLICATE FAIL: duplicate (symbol, split) rows={dupes}"


def _prediction_pnl_missing_reason(result_row: dict, artifact_row: dict, col: str, checksum_col: str) -> str:
    value = result_row.get(checksum_col)
    if value not in ("missing", "all_nan", None, ""):
        return ""
    bt_path = Path(artifact_row.get("backtest_results") or result_row.get("path", ""))
    if not bt_path.exists():
        return f"{checksum_col} missing: backtest_results missing at {bt_path}"
    try:
        df = pl.read_parquet(bt_path)
    except Exception as exc:
        return f"{checksum_col} missing: cannot read backtest_results at {bt_path}: {exc}"
    if col not in df.columns:
        return f"{checksum_col} missing: column {col} absent in {bt_path}"
    return f"{checksum_col} missing: column {col} all non-finite/null in {bt_path}"


def _artifact_failure_reasons(result_row: dict, artifact_row: dict) -> list[str]:
    reasons = []
    if result_row.get("error"):
        reasons.append(str(result_row["error"]))
    for key in [
        "backtest_results",
        "oos_predictions",
        "leakage_report",
        "execution_trace_report",
        "metrics_report",
        "stress_report",
        "acceptance_report",
    ]:
        path = artifact_row.get(key)
        if path and not Path(path).exists():
            reasons.append(f"missing {key}: {path}")
    for col, checksum_col in [("prediction_prob", "pred_cs"), ("pnl", "pnl_cs")]:
        reason = _prediction_pnl_missing_reason(result_row, artifact_row, col, checksum_col)
        if reason:
            reasons.append(reason)
    if artifact_row.get("error") and artifact_row.get("error") not in reasons:
        reasons.append(str(artifact_row["error"]))
    return reasons


def _write_failure_reasons(rows: list[dict]) -> tuple[Path, Path]:
    out_dir = Path("reports") / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "failure_reasons.csv"
    json_path = out_dir / "failure_reasons.json"
    fields = [
        "symbol",
        "split",
        "status",
        "rc",
        "command",
        "exception_type",
        "exception_message",
        "stdout_tail",
        "stderr_tail",
        "pred_cs",
        "pnl_cs",
        "failure_reason",
        "backtest_results",
        "oos_predictions",
        "metrics_report",
        "acceptance_report",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


def _prepare_final_artifact_completeness(config: RootConfig, splits: list, files: list[Path]) -> dict:
    expected = _expected_runtime_rows(config, splits, files)
    expected_count = len(config.symbols) * len(splits)
    expected_by_key = {(r["symbol"], r["split"]): r for r in expected}
    result_by_key = {(r.get("symbol"), int(r.get("split", 0))): r for r in _VERIFICATION_TABLE}
    artifact_by_key = {(r.get("symbol"), int(r.get("split", 0))): r for r in _RUN_SPLIT_ARTIFACTS}

    for key, exp in expected_by_key.items():
        if key not in result_by_key:
            row = _failed_result_row(exp["split"], exp["symbol"], exp["backtest_results"], error="missing_result_row")
            row.update({k: exp.get(k) for k in ["backtest_results", "oos_predictions", "metrics_report", "acceptance_report"]})
            _record_split_result(row)
        if key not in artifact_by_key:
            _record_split_artifacts(
                exp["symbol"],
                exp["split"],
                exp["cli_command"],
                Path(exp["out_dir"]),
                None,
                None,
                None,
                None,
                "FAILED",
                error="missing_artifact_row",
            )

    failure_rows = []
    result_by_key = {(r.get("symbol"), int(r.get("split", 0))): r for r in _VERIFICATION_TABLE}
    artifact_by_key = {(r.get("symbol"), int(r.get("split", 0))): r for r in _RUN_SPLIT_ARTIFACTS}
    for key in sorted(expected_by_key, key=lambda x: (str(x[0]), int(x[1]))):
        result_row = result_by_key[key]
        artifact_row = {**artifact_by_key.get(key, {}), **expected_by_key[key]}
        reasons = _artifact_failure_reasons(result_row, artifact_row)
        result_row["failure_reason"] = "; ".join(reasons)
        result_row["backtest_results"] = artifact_row.get("backtest_results")
        result_row["oos_predictions"] = artifact_row.get("oos_predictions")
        result_row["metrics_report"] = artifact_row.get("metrics_report")
        result_row["acceptance_report"] = artifact_row.get("acceptance_report")
        for diag_key in ["rc", "command", "stdout_tail", "stderr_tail", "exception_type", "exception_message"]:
            if not result_row.get(diag_key) and artifact_row.get(diag_key):
                result_row[diag_key] = artifact_row.get(diag_key)
        if reasons or result_row.get("status") != "OK":
            failure_rows.append(
                {
                    **result_row,
                    "failure_reason": result_row["failure_reason"] or result_row.get("status", "FAILED"),
                }
            )

    assert len(_VERIFICATION_TABLE) == expected_count, (
        f"FINAL SUMMARY ROW COUNT FAIL: rows={len(_VERIFICATION_TABLE)} expected={expected_count} "
        f"symbols={list(config.symbols)} splits={len(splits)}"
    )
    keys = [(r.get("symbol"), int(r.get("split", 0))) for r in _VERIFICATION_TABLE]
    assert len(keys) == len(set(keys)), "FINAL SUMMARY DUPLICATE ROW FAIL: duplicate (symbol, split) rows present"
    csv_path, json_path = _write_failure_reasons(failure_rows)
    return {
        "status": "PASS" if not failure_rows else "FAIL",
        "expected_rows": expected_count,
        "actual_rows": len(_VERIFICATION_TABLE),
        "failure_count": len(failure_rows),
        "failure_reasons_csv": str(csv_path),
        "failure_reasons_json": str(json_path),
    }


def _final_safety_summary(config: RootConfig, total_splits: int, deployment_report: dict) -> dict:
    counts = {"ACCEPT": 0, "REJECT": 0, "WARN": 0, "MISSING": 0}
    for row in _RUN_SPLIT_ARTIFACTS:
        status = row.get("acceptance_status", "MISSING")
        counts[status if status in counts else "MISSING"] += 1
    successful = sum(1 for r in _RUN_SPLIT_ARTIFACTS if r.get("status") == "OK")
    expected_rows = len(config.symbols) * int(total_splits)
    failed = max(expected_rows - successful, 0)
    summary = {
        "run_id": _RUN_ID,
        "profile": _ns_cfg.ACTIVE_PROFILE,
        "symbols": list(config.symbols),
        "splits": total_splits,
        "expected_symbol_split_rows": expected_rows,
        "successful": successful,
        "failed": failed,
        "total_reports_written": sum(
            1
            for row in _RUN_SPLIT_ARTIFACTS
            for key in ["leakage_report", "execution_trace_report", "metrics_report", "stress_report", "acceptance_report"]
            if Path(row.get(key, "")).exists()
        ),
        "acceptance": counts,
        "deployment": deployment_report.get("status", "MISSING"),
        "deployment_mode": deployment_report.get("deployment", {}).get("mode"),
    }
    print("\n[FINAL SAFETY SUMMARY]", flush=True)
    print(f"run_id={summary['run_id']}", flush=True)
    print(f"profile={summary['profile']}", flush=True)
    print(f"symbols={','.join(summary['symbols'])}", flush=True)
    print(
        f"splits={summary['splits']} expected_rows={summary['expected_symbol_split_rows']} "
        f"successful={summary['successful']} failed={summary['failed']}",
        flush=True,
    )
    print(
        "acceptance: "
        f"ACCEPT={counts['ACCEPT']} REJECT={counts['REJECT']} WARN={counts['WARN']} MISSING={counts['MISSING']}",
        flush=True,
    )
    print(f"deployment={summary['deployment']} mode={summary['deployment_mode']}", flush=True)
    return summary
    if _PROGRESS.verbose:
        _PROGRESS.flush_stages()
        _print_split_result(row)
    else:
        _print_clean_split_result(row)


def _failed_result_row(
    split_idx: int,
    symbol: str,
    path: Path | str,
    *,
    mtime: float = 0.0,
    error: str | None = None,
    rc: int | None = None,
    command: str | None = None,
    stdout_tail: str | None = None,
    stderr_tail: str | None = None,
) -> dict:
    exception_type, exception_message = _exception_parts(error)
    return {
        'split': split_idx, 'symbol': symbol, 'path': str(path),
        'mtime': mtime, 'run_id': _RUN_ID, 'rows': 0,
        'ts_min': 'missing', 'ts_max': 'missing',
        'pnl_cs': 'missing', 'pred_cs': 'missing',
        'pnl': 0.0, 'sharpe': 0.0, 'trades': 0, 'turnover': 0.0,
        'hit_rate': 0.0, 'ic': 'NA', 'status': 'FAILED',
        'sharpe_annualized': 0.0, 'sharpe_per_bar': 0.0, 'bars_per_year': 'NA',
        'bar_hit_rate_all_bars': 0.0, 'bar_hit_rate_all_bars_n': 0,
        'bar_hit_rate_active_bars': 0.0, 'bar_hit_rate_active_bars_n': 0,
        'trade_hit_rate': 'NA', 'trade_hit_rate_n': 0,
        'position_change_events': 0, 'position_turnover': 0.0,
        'error': error,
        'rc': rc,
        'command': command,
        'stdout_tail': stdout_tail or '',
        'stderr_tail': stderr_tail or '',
        'exception_type': exception_type,
        'exception_message': exception_message,
    }


def _tracking_summary(*, saved: Path | str | None = None, failed: str | None = None) -> str:
    lines = []
    for path in _PROGRESS.context.get('stale_artifacts', []):
        lines.extend(['removed stale artifact:', str(path)])
    if saved is not None:
        lines.append(f'saved={saved}')
    if failed is not None:
        lines.append(failed)
    return '\n'.join(lines) if lines else 'not_observed'


def _hash_column(df: pl.DataFrame, col: str) -> str:
    if col not in df.columns:
        return 'missing'
    vals = df[col].to_numpy()
    mask = np.isfinite(vals)
    if mask.sum() == 0:
        return 'all_nan'
    h = hashlib.sha256(vals[mask].tobytes()).hexdigest()[:8]
    return h


def _summary_ic(bt: pl.DataFrame):
    if 'prediction_prob' not in bt.columns or 'ret_exec' not in bt.columns:
        return 'NA'
    try:
        from pipeline.analytics.aggregate import compute_ic
        result = compute_ic(bt['prediction_prob'].shift(1), bt['ret_exec'])
        val = result.get('spearman_ic')
        return 'NA' if val is None or not np.isfinite(float(val)) else f'{float(val):+.4f}'
    except Exception:
        return 'NA'


def _print_symbol_diagnostics(bt: pl.DataFrame, symbol: str, split_idx: int) -> None:
    from pipeline.analytics.aggregate import compute_backtest_metrics
    metrics = compute_backtest_metrics(bt)
    probs = bt['prediction_prob'].to_numpy().astype(np.float64) if 'prediction_prob' in bt.columns else None
    pmean = float(probs.mean()) if probs is not None else float('nan')
    pstd = float(probs.std()) if probs is not None else float('nan')
    gt055 = float((probs > 0.55).mean()) if probs is not None else float('nan')
    lt045 = float((probs < 0.45).mean()) if probs is not None else float('nan')
    gross_sharpe = 'missing'
    net_sharpe = f"{metrics['sharpe_annualized']:.3f}"
    cost_drag = 'missing'
    if 'gross_pnl' in bt.columns:
        from pipeline.analytics.aggregate import _sharpe_pair
        gp = bt['gross_pnl'].to_numpy().astype(np.float64)
        gross_sharpe = f'{_sharpe_pair(gp, metrics["bars_per_year"])[1]:.3f}'
    if 'pnl' in bt.columns:
        np_ = bt['pnl'].to_numpy().astype(np.float64)
        if 'gross_pnl' in bt.columns:
            cost_drag = f'{float(np_.sum() - gp.sum()):+.2f}'
    turnover = f'{metrics["position_turnover"]:.6g}'
    trades = f'{metrics["position_change_events"]}'
    pred_cs = _hash_column(bt, 'prediction_prob')
    pnl_cs = _hash_column(bt, 'pnl')
    sig_cs = _hash_column(bt, 'raw_signal') if 'raw_signal' in bt.columns else _hash_column(bt, 'target_exec')
    logger.info(
        '[DIAG] symbol=%s split=%d prob_mean=%.4f prob_std=%.4f gt055=%.3f lt045=%.3f '
        'gross_sharpe_annualized=%s net_sharpe_annualized=%s bars_per_year=%s cost_drag=%s '
        'position_turnover=%s position_change_events=%s '
        'bar_hit_rate_all=%0.4f denom_all=%d bar_hit_rate_active=%0.4f denom_active=%d trade_hit_rate=%s denom_trades=%d '
        'pred_cs=%s pnl_cs=%s sig_cs=%s',
        symbol, split_idx, pmean, pstd, gt055, lt045,
        gross_sharpe, net_sharpe, metrics['bars_per_year'], cost_drag, turnover, trades,
        metrics['bar_hit_rate_all_bars'], metrics['bar_hit_rate_all_bars_n'],
        metrics['bar_hit_rate_active_bars'], metrics['bar_hit_rate_active_bars_n'],
        metrics.get('trade_hit_rate'), metrics['trade_hit_rate_n'],
        pred_cs, pnl_cs, sig_cs,
    )


def _validate_backtest_output(out_dir: Path, symbol: str) -> None:
    bt_path = out_dir / 'backtest_results.parquet'
    if not bt_path.exists():
        raise RuntimeError('BACKTEST FAILURE: %s no output at %s' % (symbol, bt_path))
    try:
        df = pl.read_parquet(bt_path)
    except Exception as e:
        raise RuntimeError('BACKTEST FAILURE: %s cannot read %s: %s' % (symbol, bt_path, e))
    if df.height == 0:
        raise RuntimeError('BACKTEST FAILURE: %s empty output at %s' % (symbol, bt_path))
    if 'pnl' not in df.columns:
        raise RuntimeError('BACKTEST FAILURE: %s missing pnl column at %s' % (symbol, bt_path))
    pnl_sum = df['pnl'].sum()
    pnl_mean = df['pnl'].mean()
    pnl_std = df['pnl'].std()
    multiplier = get_contract_multiplier(symbol)
    notional = df['close'].mean() * multiplier if 'close' in df.columns else 0
    logger.info('[VALIDATE] %s: rows=%d pnl_sum=%.2f pnl_mean=%.6f pnl_std=%.4f notional=%.0f mult=%.6g',
                symbol, df.height, pnl_sum, pnl_mean, pnl_std, notional, multiplier)
    if abs(pnl_std) < 0.0001 and multiplier > 1:
        raise RuntimeError(
            'BACKTEST FAILURE: %s pnl std %.8f is zero-scale '
            '(multiplier=%.6g, notional=%.0f) -- PnL not in USD futures' %
            (symbol, pnl_std, multiplier, notional)
        )


def _contract_summary(symbol: str) -> str:
    try:
        import yaml
        cfg_path = Path('configs') / 'market_specs.yaml'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            market_cfg = (yaml.safe_load(f) or {}).get('markets', {}).get(symbol, {})
        multiplier = get_contract_multiplier(symbol)
        tick = market_cfg.get('contract_specs', {}).get('tick_size', 'NA')
        return f'symbol={symbol} multiplier={multiplier:g} tick={tick}'
    except Exception as e:
        return f'symbol={symbol} contract_metadata_error={e}'


def load_all_splits_for_year(year: int) -> list:
    """Load per-split manifests and backtest results for a given year."""
    import glob as _glob
    manifests = sorted(_glob.glob(f'output/manifest_{year}_split_*.json'))
    if not manifests:
        fallback = Path(f'output/manifest_{year}.json')
        if fallback.exists():
            manifests = [str(fallback)]
    summaries = []
    for mpath in manifests:
        with open(mpath, 'r', encoding='utf-8') as f:
            mf = json.load(f)
        out_path = mf.get('output_path')
        if out_path and Path(out_path).exists():
            try:
                df = pl.read_parquet(out_path)
                summaries.append({
                    'symbol': mf.get('symbol'),
                    'year': mf.get('year'),
                    'split_idx': mf.get('split_idx'),
                    'rows': df.height,
                })
            except Exception as e:
                summaries.append({'symbol': mf.get('symbol'), 'year': mf.get('year'),
                                  'split_idx': mf.get('split_idx'), 'error': str(e)})
        else:
            summaries.append({'symbol': mf.get('symbol'), 'year': mf.get('year'),
                              'split_idx': mf.get('split_idx'), 'error': 'backtest file missing'})
    return summaries


def process_split(train_years: list[int], test_years: list[int], files: list[Path],
                   config: RootConfig, split_idx: int, total_splits: int,
                   train_start=None, train_end=None, test_start=None, test_end=None) -> None:
    train_files = [f for f in files if int(f.stem) in train_years]
    test_files = [f for f in files if int(f.stem) in test_years]
    _raw_log(
        f'[SPLIT {split_idx}/{total_splits}] test_window=[{test_start}, {test_end}) run_id={_RUN_ID} '
        f'cwd={os.getcwd()} run_py_mtime={Path(__file__).stat().st_mtime:.0f} '
        f'cli_py_mtime={(Path(__file__).resolve().parent / "pipeline" / "cli.py").stat().st_mtime:.0f}'
    )
    if not train_files or not test_files:
        logger.warning('Empty train/test split — skipping')
        return
    # ---- Config safeguard: bootstrap discovery needs enough samples ----
    wf_train = getattr(config.walkforward, 'wf_train_days', 30)
    if wf_train < _MIN_TRAIN_DAYS:
        logger.warning(
            'wf_train_days=%d < %d — discovery may have insufficient samples. '
            'Consider wf_train_days >= %d for stable bootstrap.',
            wf_train, _MIN_TRAIN_DAYS, _MIN_TRAIN_DAYS
        )
    train_dir = Path('output') / f"train_{'_'.join(map(str, train_years))}_split_{split_idx}"
    train_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Preparing TRAIN dataset for years %s', train_years)
    for f in train_files:
        dst = train_dir / f'{f.parent.name}_{f.name}'
        if dst.exists():
            continue
        if os.name != 'nt':
            try:
                os.symlink(f.resolve(), dst)
                continue
            except Exception:
                pass
        shutil.copy2(f, dst)
    test_dir = Path('output') / f"test_{'_'.join(map(str, test_years))}_split_{split_idx}"
    test_dir.mkdir(parents=True, exist_ok=True)
    for f in test_files:
        dst = test_dir / f.parent.name / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if os.name != 'nt':
            try:
                os.symlink(f.resolve(), dst)
                continue
            except Exception:
                pass
        shutil.copy2(f, dst)
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    per_symbol = {}
    for symbol in config.symbols:
        symbol_test_files = sorted([p for p in test_files if p.parent.name == symbol])
        test_year_label = "_".join(str(p.stem) for p in symbol_test_files) or "missing"
        test_data_arg = str(test_dir / symbol / '*.parquet')
        test_label = (
            f'{test_start.isoformat()}→{test_end.isoformat()}'
            if test_start and test_end else test_year_label
        )
        _PROGRESS.set_context(
            symbol=symbol,
            split=split_idx,
            total_splits=total_splits,
            contract_summary=_contract_summary(symbol),
        )
        print(f'\n[SPLIT {split_idx}/{total_splits}] {symbol} test={test_label}', flush=True)
        _stage(1, f'{test_data_arg} found files={len(symbol_test_files)}')
        invalid_file = next((p for p in symbol_test_files if not _validate_symbol_data(p, config)), None)
        if not symbol_test_files or invalid_file is not None:
            err = 'missing_test_file' if not symbol_test_files else f'invalid_or_empty_raw_data: {invalid_file}'
            _stage(13, 'FAILED invalid_or_empty_raw_data')
            _record_split_result(_failed_result_row(
                split_idx, symbol, test_data_arg,
                mtime=invalid_file.stat().st_mtime if invalid_file and invalid_file.exists() else 0.0,
                error=err,
            ))
            continue
        train_glob = str(train_dir / f'{symbol}_*.parquet')
        manifest_path = Path('output') / f"manifest_{symbol}_{'_'.join(map(str, train_years))}_split_{split_idx}_{_ns_cfg.ACTIVE_PROFILE}.json"
        if config.pipeline.enable_discovery:
            symbol_train_files = sorted(train_dir.glob(f'{symbol}_*.parquet'))
            if not symbol_train_files:
                raise RuntimeError(
                    f'MULTI-SYMBOL DISCOVERY FAIL: no train files for symbol={symbol} '
                    f'in {train_dir}. Refusing to use a cross-symbol manifest.'
                )
            discover_cmd = [sys.executable, '-m', 'pipeline.cli', 'discover',
                            '--data', train_glob, '--out', str(manifest_path)]
            if train_start and train_end:
                discover_cmd.extend(['--start', train_start.isoformat(), '--end', train_end.isoformat()])
            _raw_log(f'[DISCOVERY-CMD] symbol={symbol} {" ".join(discover_cmd)}')
            _run_subprocess_streaming(discover_cmd, env)
            time.sleep(0.2)
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(manifest_path) + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'version': '1.0', 'feature_names': [], 'selection_seed': config.preprocessing.seed,
                           'selection_date': 'placeholder', 'discovery_status': 'disabled'}, f)
            os.replace(tmp, str(manifest_path))
        out_dir = Path('output') / symbol / f'{test_year_label}_split_{split_idx}_{_ns_cfg.ACTIVE_PROFILE}'
        out_dir.mkdir(parents=True, exist_ok=True)
        # Purge stale backtest files from previous runs
        for _stale_name in ('backtest_results_hmm.parquet', 'backtest_results.parquet'):
            _stale_path = out_dir / _stale_name
            if _stale_path.exists() and _stale_path.stat().st_mtime < _RUN_START:
                _PROGRESS.context.setdefault('stale_artifacts', []).append(str(_stale_path))
                _stage(13, _tracking_summary())
                logger.warning('[STALE] Removing pre-existing file: %s (mtime=%.0f < run_start=%.0f)',
                               _stale_path, _stale_path.stat().st_mtime, _RUN_START)
                _stale_path.unlink()
        hmm_enabled = bool(getattr(getattr(config, 'hmm', None), 'enabled', False))
        cli_action = 'run-hmm' if hmm_enabled else 'run'
        _raw_log(f'[RUN-MODE] symbol={symbol} split={split_idx} hmm_enabled={str(hmm_enabled).lower()} cli={cli_action}')
        cmd = [sys.executable, '-m', 'pipeline.cli', cli_action, '--data', test_data_arg,
               '--manifest', str(manifest_path), '--out', str(out_dir)]
        if train_start and train_end:
            cmd.extend(['--train-start', train_start.isoformat(), '--train-end', train_end.isoformat()])
        if test_start and test_end:
            cmd.extend(['--start', test_start.isoformat(), '--end', test_end.isoformat()])
        wf_mode = getattr(config, 'WF_MODE', '')
        if wf_mode == 'outer_split':
            assert train_start and train_end, 'outer_split mode but train_start/train_end not set'
            assert '--train-start' in cmd and '--train-end' in cmd, '--train-start/--train-end missing from subprocess command'
        _raw_log(f'[SUBPROCESS-CMD] {" ".join(cmd)}')
        logger.info('[SUBPROCESS] split=%d symbol=%s start=%s end=%s cmd=%s',
                    split_idx, symbol,
                    test_start.isoformat() if test_start else 'None',
                    test_end.isoformat() if test_end else 'None',
                    ' '.join(cmd))
        subprocess_failed = False
        try:
            _run_subprocess_streaming(cmd, env)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if not hmm_enabled:
                if _DEBUG:
                    import traceback
                    traceback.print_exc()
                stderr_text = e.stderr if hasattr(e, 'stderr') else ''
                stdout_text = e.stdout if hasattr(e, 'stdout') else ''
                if isinstance(stderr_text, bytes):
                    stderr_text = stderr_text.decode(errors='replace')
                if isinstance(stdout_text, bytes):
                    stdout_text = stdout_text.decode(errors='replace')
                run_error = _extract_error_summary(stderr_text, stdout_text)
                _stage(13, _tracking_summary(failed=f'FAILED\n{run_error}'))
                if _log_failures:
                    _log_subprocess_failure(cmd, getattr(e, 'returncode', -1), stderr_text, stdout_text,
                                            Path('output') / 'logs', symbol, split_idx, 'run')
                logger.error('run failed for %s split=%d (rc=%d)',
                             symbol, split_idx, getattr(e, 'returncode', -1))
                rc = getattr(e, 'returncode', -1)
                cmd_text = ' '.join(map(str, cmd))
                stdout_tail = _tail_text(stdout_text)
                stderr_tail = _tail_text(stderr_text)
                _record_split_result(_failed_result_row(
                    split_idx, symbol, out_dir, error=run_error,
                    rc=rc, command=cmd_text, stdout_tail=stdout_tail, stderr_tail=stderr_tail,
                ))
                _record_split_artifacts(
                    symbol, split_idx, cli_action, out_dir, train_start, train_end, test_start, test_end,
                    'FAILED', error=run_error, rc=rc, command_line=cmd_text,
                    stdout_tail=stdout_tail, stderr_tail=stderr_tail,
                )
                if os.environ.get('QUANT_ACCEPTANCE_GATE_REQUIRED') == '1':
                    raise RuntimeError(f'ACCEPTANCE REQUIRED FAIL: {run_error}') from e
                subprocess_failed = True
                continue
            if _DEBUG:
                import traceback
                traceback.print_exc()
            stderr_text = e.stderr if hasattr(e, 'stderr') else ''
            stdout_text = e.stdout if hasattr(e, 'stdout') else ''
            if isinstance(stderr_text, bytes):
                stderr_text = stderr_text.decode(errors='replace')
            if isinstance(stdout_text, bytes):
                stdout_text = stdout_text.decode(errors='replace')
            err_lines = stderr_text.strip().split('\n') if stderr_text else ['(no stderr)']
            hmm_error = _extract_error_summary(stderr_text, stdout_text)
            _stage(8, f'FAILED\n{hmm_error}\nFallback=non-HMM')
            if _PROGRESS.verbose:
                print(f'[WARNING] HMM failed on split {split_idx} ({symbol}). Last 3 lines of stderr:')
                for line in err_lines[-3:]:
                    print(f'  > {line}')
            if _log_failures:
                _log_subprocess_failure(cmd, getattr(e, 'returncode', -1), stderr_text, stdout_text,
                                        Path('output') / 'logs', symbol, split_idx, 'hmm')
            logger.warning(
                'run-hmm failed for %s split=%d (rc=%d)',
                symbol, split_idx, getattr(e, 'returncode', -1),
            )
            # Safe fallback: retry with plain run (no HMM)
            logger.info('Falling back to non-HMM run for %s split=%d', symbol, split_idx)
            cmd_fb = [sys.executable, '-m', 'pipeline.cli', 'run', '--data', str(f),
                       '--manifest', str(manifest_path), '--out', str(out_dir)]
            if train_start and train_end:
                cmd_fb.extend(['--train-start', train_start.isoformat(), '--train-end', train_end.isoformat()])
            if test_start and test_end:
                cmd_fb.extend(['--start', test_start.isoformat(), '--end', test_end.isoformat()])
            try:
                _run_subprocess_streaming(cmd_fb, env)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                if _DEBUG:
                    import traceback
                    traceback.print_exc()
                stderr2 = e2.stderr if hasattr(e2, 'stderr') else ''
                stdout2 = e2.stdout if hasattr(e2, 'stdout') else ''
                if isinstance(stderr2, bytes):
                    stderr2 = stderr2.decode(errors='replace')
                if isinstance(stdout2, bytes):
                    stdout2 = stdout2.decode(errors='replace')
                err2_lines = stderr2.strip().split('\n') if stderr2 else ['(no stderr)']
                fallback_error = _extract_error_summary(stderr2, stdout2)
                _stage(13, _tracking_summary(failed=f'FAILED\n{fallback_error}'))
                if _PROGRESS.verbose:
                    print(f'[ERROR] Fallback also failed on split {split_idx} ({symbol}). Last 3 lines:')
                    for line in err2_lines[-3:]:
                        print(f'  > {line}')
                if _log_failures:
                    _log_subprocess_failure(cmd_fb, getattr(e2, 'returncode', -1), stderr2, stdout2,
                                            Path('output') / 'logs', symbol, split_idx, 'fallback')
                logger.error('Fallback run also failed for %s split=%d', symbol, split_idx)
                rc2 = getattr(e2, 'returncode', -1)
                cmd_fb_text = ' '.join(map(str, cmd_fb))
                stdout2_tail = _tail_text(stdout2)
                stderr2_tail = _tail_text(stderr2)
                _record_split_result(_failed_result_row(
                    split_idx, symbol, out_dir, error=fallback_error,
                    rc=rc2, command=cmd_fb_text, stdout_tail=stdout2_tail, stderr_tail=stderr2_tail,
                ))
                _record_split_artifacts(
                    symbol, split_idx, 'run', out_dir, train_start, train_end, test_start, test_end,
                    'FAILED', error=fallback_error, rc=rc2, command_line=cmd_fb_text,
                    stdout_tail=stdout2_tail, stderr_tail=stderr2_tail,
                )
                if os.environ.get('QUANT_ACCEPTANCE_GATE_REQUIRED') == '1':
                    raise RuntimeError(f'ACCEPTANCE REQUIRED FAIL: {fallback_error}') from e2
                subprocess_failed = True
        if subprocess_failed:
            continue
        time.sleep(0.2)
        bt_path = out_dir / 'backtest_results_hmm.parquet'
        if not bt_path.exists():
            bt_path = out_dir / 'backtest_results.parquet'
        if bt_path.exists():
            _bt_mtime = bt_path.stat().st_mtime
            assert _bt_mtime >= _RUN_START, (
                f'STALE RESULT: {bt_path} mtime={_bt_mtime:.0f} < run_start={_RUN_START:.0f}. '
                f'Subprocess may have failed silently and an old file was read.'
            )
            _raw_log(f'[BACKTEST] {symbol} path={bt_path} mtime={_bt_mtime:.0f} run_id={_RUN_ID}')
            try:
                bt = pl.read_parquet(bt_path)
                t_min = str(bt['ts_event'].min()) if 'ts_event' in bt.columns else 'missing'
                t_max = str(bt['ts_event'].max()) if 'ts_event' in bt.columns else 'missing'
                _raw_log(f'[BACKTEST] {symbol} rows={bt.height} ts_min={t_min} ts_max={t_max}')
                # Assert timestamps are within the test window
                if test_start and test_end and 'ts_event' in bt.columns:
                    from datetime import datetime as _dt, timezone
                    ts_start = _dt.fromisoformat(test_start.isoformat()).replace(tzinfo=timezone.utc)
                    ts_end = _dt.fromisoformat(test_end.isoformat()).replace(tzinfo=timezone.utc)
                    bt_t_min = bt['ts_event'].min()
                    bt_t_max = bt['ts_event'].max()
                    assert bt_t_min >= ts_start, (
                        f'BOUNDARY VIOLATION: {symbol} split={split_idx} bt_t_min={bt_t_min} < test_start={ts_start}'
                    )
                    assert bt_t_max < ts_end, (
                        f'BOUNDARY VIOLATION: {symbol} split={split_idx} bt_t_max={bt_t_max} >= test_end={ts_end}'
                    )
                    horizon_min = int(getattr(_ns_cfg, 'TARGET_15M_HORIZON', 15)) * 5
                    safe_label_end = ts_end - timedelta(minutes=horizon_min)
                    assert bt_t_max < safe_label_end, (
                        f'TARGET BOUNDARY VIOLATION: {symbol} split={split_idx} '
                        f'bt_t_max={bt_t_max} >= test_end-horizon={safe_label_end}. '
                        f'Test labels would require data beyond the exclusive test_end.'
                    )
                pnl_cs = _hash_column(bt, 'pnl')
                pred_cs = _hash_column(bt, 'prediction_prob')
                from pipeline.analytics.aggregate import compute_backtest_metrics
                metrics = compute_backtest_metrics(bt)
                pnl = metrics['total_pnl']
                sharpe = metrics['sharpe_annualized']
                hit = metrics['bar_hit_rate_active_bars']
                _raw_log(
                    f'[VERIFY] {symbol} split={split_idx} pnl_cs={pnl_cs} pred_cs={pred_cs} '
                    f'sharpe_annualized={sharpe:.3f} sharpe_per_bar={metrics["sharpe_per_bar"]:.6f} '
                    f'bars_per_year={metrics["bars_per_year"]} pnl={pnl:.2f} '
                    f'bar_hit_rate_active={hit:.2%} denom_active={metrics["bar_hit_rate_active_bars_n"]} '
                    f'position_turnover={metrics["position_turnover"]:.6g}'
                )
                if symbol not in _PER_SYMBOL_PNL_CS:
                    _PER_SYMBOL_PNL_CS[symbol] = {}
                if pnl_cs != 'missing' and pnl_cs != 'all_nan':
                    for prior_split, prior_cs in _PER_SYMBOL_PNL_CS[symbol].items():
                        assert prior_cs != pnl_cs, (
                            f'IDENTICAL PNL OUTPUT: {symbol} split={split_idx} pnl_cs={pnl_cs} '
                            f'matches split={prior_split}. Stale cache or file reuse detected.'
                        )
                    _PER_SYMBOL_PNL_CS[symbol][split_idx] = pnl_cs
                # Collect for verification table
                trades = int(metrics['position_change_events'])
                turnover = float(metrics['position_turnover'])
                ic = _summary_ic(bt)
                _stage(10, f'position_change_events={trades} position_turnover={turnover:.6g}')
                _stage(11, f'rows={bt.height}')
                _stage(12, f'sharpe_annualized={sharpe:.3f} pnl={pnl:.2f} ic={ic} bar_hit_rate_active={hit:.2%}')
                _stage(13, _tracking_summary(saved=bt_path))
                result_row = {
                    'split': split_idx, 'symbol': symbol, 'path': str(bt_path),
                    'mtime': _bt_mtime, 'run_id': _RUN_ID, 'rows': bt.height,
                    'ts_min': str(t_min), 'ts_max': str(t_max),
                    'pnl_cs': pnl_cs, 'pred_cs': pred_cs, 'pnl': pnl, 'sharpe': sharpe,
                    'trades': trades, 'turnover': turnover, 'hit_rate': hit, 'ic': ic, 'status': 'OK',
                    'sharpe_annualized': metrics['sharpe_annualized'],
                    'sharpe_per_bar': metrics['sharpe_per_bar'],
                    'bars_per_year': metrics['bars_per_year'],
                    'bar_hit_rate_all_bars': metrics['bar_hit_rate_all_bars'],
                    'bar_hit_rate_all_bars_n': metrics['bar_hit_rate_all_bars_n'],
                    'bar_hit_rate_active_bars': metrics['bar_hit_rate_active_bars'],
                    'bar_hit_rate_active_bars_n': metrics['bar_hit_rate_active_bars_n'],
                    'trade_hit_rate': metrics.get('trade_hit_rate'),
                    'trade_hit_rate_n': metrics['trade_hit_rate_n'],
                    'position_change_events': metrics['position_change_events'],
                    'position_turnover': metrics['position_turnover'],
                }
                _record_split_artifacts(symbol, split_idx, cli_action, out_dir, train_start, train_end, test_start, test_end, 'OK')
                # HMM delta: compare with raw (non-HMM) baseline if available
                hmm_delta = 0.0
                raw_path = out_dir / 'backtest_results.parquet'
                if raw_path.exists() and raw_path != bt_path:
                    try:
                        raw = pl.read_parquet(raw_path)
                        raw_sharpe = compute_backtest_metrics(raw)['sharpe_annualized']
                        hmm_delta = sharpe - raw_sharpe
                    except Exception:
                        hmm_delta = 0.0
                per_symbol[symbol] = (sharpe, pnl, hit, hmm_delta)
                _print_symbol_diagnostics(bt, symbol, split_idx)
                _record_split_result(result_row)
            except Exception as e:
                print(f'[ERROR] Result validation failed for {symbol} split={split_idx}: {e}', flush=True)
                if _DEBUG:
                    import traceback
                    traceback.print_exc()
                per_symbol[symbol] = (0.0, 0.0, 0.0, 0.0)
                result_error = _extract_error_summary(str(e))
                _stage(13, _tracking_summary(failed=f'FAILED\n{result_error}'))
                _record_split_result(_failed_result_row(
                    split_idx, symbol, bt_path, mtime=_bt_mtime, error=result_error
                ))
                _record_split_artifacts(symbol, split_idx, cli_action, out_dir, train_start, train_end, test_start, test_end, 'FAILED', error=result_error)
        else:
            _stage(13, _tracking_summary(failed=f'missing_output={out_dir} status=FAILED'))
            _record_split_result(_failed_result_row(
                split_idx, symbol, out_dir, error='missing_output'
            ))
            _record_split_artifacts(symbol, split_idx, cli_action, out_dir, train_start, train_end, test_start, test_end, 'FAILED', error='missing_output')
    # ---- Master dashboard ----
    _print_split_dashboard(split_idx, total_splits, per_symbol)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--from-stage", dest="from_stage")
    parser.add_argument("--data-root", dest="data_root")
    args, _unknown = parser.parse_known_args()
    config = load_config(os.environ.get('CONFIG_ENV') or os.environ.get('QUANT_ENV'))
    start_stage = normalize_start_stage(args.from_stage or os.environ.get("QUANT_START_STAGE") or getattr(config.pipeline, "start_stage", "raw"))
    data_root_override = args.data_root or os.environ.get("QUANT_DATA_ROOT")
    if data_root_override:
        config.data.root = data_root_override
        config.pipeline.checkpoint_root = data_root_override
    auto_report = None
    if start_stage == "auto":
        source = data_root_override or config.data.root
        auto_report = classify_checkpoint(source, getattr(config.walkforward, "walkforward_target", "target_15m_ret"))
        inferred = canonical_stage(auto_report["inferred_stage"])
        print(
            f"[CHECKPOINT AUTO-DETECT]\nsource={source}\ninferred_stage={inferred}\n"
            f"reason={auto_report['reason']}",
            flush=True,
        )
        if inferred in {"unknown", "raw"}:
            raise SystemExit(f"CHECKPOINT GATE FAIL: stage=auto root={source} reason={auto_report['reason']}")
        start_stage = inferred
        next_num = START_STAGE_NUM[start_stage]
        next_name = next((s.name for s in stage_catalog() if s.number == next_num), "")
        print(f"continuing_from_stage={next_num} {next_name}", flush=True)
    config.pipeline.start_stage = start_stage
    config.pipeline.checkpoint_stage = None if start_stage == "raw" else start_stage
    _PROGRESS.rename_for_config(_ns_cfg.ACTIVE_PROFILE, config.symbols)
    data_dir = config.data.root if start_stage != "raw" else getattr(_ns_cfg, 'DATA_ROOT', 'data/L0_ohlcv_1m')
    checkpoint_report = None
    stage_plan = build_stage_plan(start_stage, config)
    if start_stage == "raw":
        try:
            preflight = validate_research_data_preflight(config)
            _stage(1, f"data_preflight={preflight['status']} root={preflight['data_root']}")
        except DatasetPreflightError as exc:
            raise SystemExit(f"DATA PREFLIGHT FAIL: {exc}") from exc
    else:
        if not getattr(config.pipeline, "allow_checkpoint_start", True):
            raise SystemExit(f"CHECKPOINT GATE FAIL: stage={start_stage} root={data_dir} reason=checkpoint starts disabled")
        # Advance intermediate checkpoints to the first WFA-capable root.
        if start_stage == "validated":
            from pipeline.session.normalize import session_normalize_root
            from pipeline.causal.gate import causal_gate_root
            checkpoint_report = validate_checkpoint_stage("validated", data_dir, config, config.symbols, config.start_year, config.end_year)
            if checkpoint_report["status"] != "PASS":
                raise SystemExit(f"CHECKPOINT GATE FAIL: stage=validated root={data_dir} reason={'; '.join(checkpoint_report['failures'][:3])}\n{checkpoint_report.get('remediation','')}")
            session_normalize_root(data_dir, config.data.session_normalized_root)
            causal_gate_root(config.data.session_normalized_root, config.data.causally_gated_root)
            data_dir = config.data.causally_gated_root
            start_stage = "causally_gated_normalized"
        elif start_stage == "session_normalized":
            from pipeline.causal.gate import causal_gate_root
            checkpoint_report = validate_checkpoint_stage("session_normalized", data_dir, config, config.symbols, config.start_year, config.end_year)
            if checkpoint_report["status"] != "PASS":
                raise SystemExit(f"CHECKPOINT GATE FAIL: stage=session_normalized root={data_dir} reason={'; '.join(checkpoint_report['failures'][:3])}\n{checkpoint_report.get('remediation','')}")
            causal_gate_root(data_dir, config.data.causally_gated_root)
            data_dir = config.data.causally_gated_root
            start_stage = "causally_gated_normalized"
        checkpoint_report = validate_checkpoint_stage(start_stage, data_dir, config, config.symbols, config.start_year, config.end_year)
        if checkpoint_report["status"] != "PASS":
            raise SystemExit(f"CHECKPOINT GATE FAIL: stage={start_stage} root={data_dir} reason={'; '.join(checkpoint_report['failures'][:3])}\n{checkpoint_report.get('remediation','')}")
        skipped = [s["stage_num"] for s in stage_plan if s["status"] == "SKIPPED_CHECKPOINT"]
        skipped_txt = f"{min(skipped)}-{max(skipped)}" if skipped else "none"
        next_stage = next((s for s in stage_plan if s["status"] == "PENDING"), {})
        print(
            f"[CHECKPOINT START]\nstart_stage={start_stage}\ncheckpoint_root={data_dir}\n"
            f"skipped_stages={skipped_txt}\ncheckpoint_gate={checkpoint_report['status']}\n"
            f"continuing_from_stage={next_stage.get('stage_num')} {next_stage.get('stage_name')}",
            flush=True,
        )
        if start_stage == "causally_gated_normalized":
            from pipeline.labels.generate import label_root
            from pipeline.features.baseline import baseline_feature_root
            from pipeline.features.expansion import expanded_feature_root
            label_root(data_dir, "data/labeled", config)
            baseline_feature_root("data/labeled", "data/feature_matrices/baseline", config)
            expanded_feature_root("data/feature_matrices/baseline", "data/feature_matrices/expanded", config)
        elif start_stage == "labeled":
            from pipeline.features.baseline import baseline_feature_root
            from pipeline.features.expansion import expanded_feature_root
            baseline_feature_root(data_dir, "data/feature_matrices/baseline", config)
            expanded_feature_root("data/feature_matrices/baseline", "data/feature_matrices/expanded", config)
        elif start_stage == "baseline_feature_matrix":
            from pipeline.features.expansion import expanded_feature_root
            expanded_feature_root(data_dir, "data/feature_matrices/expanded", config)
        config.data.root = data_dir
        _ns_cfg.DATA_ROOT = data_dir
    files = get_files(data_dir, config)
    if not files:
        logger.warning('No files found after filtering')
        sys.exit(0)
    if os.environ.get('QUANT_SKIP_DATASET_GATE', '0') != '1':
        try:
            validate_dataset_gate(
                files,
                symbols=config.symbols,
                manifest_path=os.environ.get('QUANT_AUDIT_MANIFEST', 'reports/validation/audit_manifest.json'),
                required=os.environ.get('QUANT_REQUIRE_DATASET_GATE', '1') == '1',
                check_hash=os.environ.get('QUANT_DATASET_GATE_HASH', '1') == '1',
            )
        except DatasetGateError as exc:
            raise SystemExit(str(exc)) from exc
    write_run_manifest(
        _RUN_ID,
        config,
        files,
        audit_paths={"research_data_preflight": "reports/validation/research_data_preflight.json"},
        checkpoint_start={"start_stage": start_stage, "checkpoint_root": data_dir, "checkpoint_gate": checkpoint_report, "stage_plan": stage_plan},
    )
    splits = generate_walkforward_splits(files, config)
    execution_plan_rows = _expected_runtime_rows(config, splits, files)
    _assert_execution_plan_unique(execution_plan_rows, list(config.symbols), splits)
    write_wfa_split_plan(splits, files, config)
    total = len(splits)
    print(
        f'[RUN] env={_ns_cfg.ACTIVE_PROFILE} profile={_ns_cfg.ACTIVE_PROFILE} '
        f'config={_ns_cfg.CONFIG_SOURCE} '
        f'symbols={",".join(config.symbols)} splits={total} log_mode={_LOG_MODE} '
        f'raw_log={_PROGRESS.log_path}',
        flush=True,
    )
    if _LOG_MODE == 'clean':
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger('quant').setLevel(logging.ERROR)
    elif _LOG_MODE == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        for i, split_data in enumerate(splits, 1):
            train, test = split_data[0], split_data[1]
            train_start = split_data[2] if len(split_data) > 2 else None
            train_end   = split_data[3] if len(split_data) > 3 else None
            test_start  = split_data[4] if len(split_data) > 4 else None
            test_end    = split_data[5] if len(split_data) > 5 else None
            process_split(train, test, files, config, i, total, train_start, train_end, test_start, test_end)
    finally:
        artifact_completeness = _prepare_final_artifact_completeness(config, splits, files)
        _print_final_summary_table()
        if artifact_completeness["status"] != "PASS":
            expected_rows = len(config.symbols) * len(splits)
            print(
                f"[FINAL SAFETY SUMMARY]\nrun_id={_RUN_ID}\nprofile={_ns_cfg.ACTIVE_PROFILE}\n"
                f"symbols={','.join(config.symbols)}\nsplits={len(splits)} expected_rows={expected_rows} "
                f"successful={sum(1 for r in _RUN_SPLIT_ARTIFACTS if r.get('status') == 'OK')} "
                f"failed={artifact_completeness['failure_count']}",
                flush=True,
            )
            print(
                f"ARTIFACT COMPLETENESS FAIL: failures={artifact_completeness['failure_count']} "
                f"details={artifact_completeness['failure_reasons_csv']}",
                flush=True,
            )
            raise RuntimeError(
                f"ARTIFACT COMPLETENESS FAIL: see {artifact_completeness['failure_reasons_csv']}"
            )
        deployment_report = run_deployment_readiness_gate(config)
        final_summary = _final_safety_summary(config, total, deployment_report)
        write_run_manifest(
            _RUN_ID,
            config,
            files,
            audit_paths={
                "research_data_preflight": "reports/validation/research_data_preflight.json",
                "deployment_readiness": "reports/acceptance/deployment_readiness.json",
            },
            splits=_RUN_SPLIT_ARTIFACTS,
            final_acceptance_summary=final_summary,
            deployment_readiness={
                "path": "reports/acceptance/deployment_readiness.json",
                "status": deployment_report.get("status"),
            },
            checkpoint_start={"start_stage": start_stage, "checkpoint_root": data_dir, "checkpoint_gate": checkpoint_report, "stage_plan": stage_plan},
        )
        _PROGRESS.close()
    logging.getLogger().setLevel(logging.INFO)
    logger.info('Done')
