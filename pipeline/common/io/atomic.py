"""
Atomic parquet write — Windows-safe single source of truth.
All parquet writes MUST use these functions, never direct write_parquet/sink_parquet.
"""
import os
import time
import json
import gc
import logging
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_ATOMIC_RETRY_DELAY = 0.2
_ATOMIC_MAX_RETRIES = 1


def atomic_write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    """
    Write DataFrame to parquet via temp file + os.replace (atomic on same fs).
    Never writes directly to final path — avoids NTFS lock contention.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Force release any mmap / lazy references
    df = df.clone()
    gc.collect()

    for attempt in range(_ATOMIC_MAX_RETRIES + 1):
        try:
            df.write_parquet(str(tmp_path))
            tmp_path.replace(path)
            gc.collect()
            return
        except OSError as e:
            if attempt < _ATOMIC_MAX_RETRIES:
                logger.warning(
                    "Parquet write retry %d/%d: %s",
                    attempt + 1, _ATOMIC_MAX_RETRIES, e,
                )
                time.sleep(_ATOMIC_RETRY_DELAY)
                gc.collect()
            else:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise


def atomic_write_canonical_parquet(
    df: pl.DataFrame,
    path: str | Path,
    row_group_size: int = 65536,
) -> None:
    """
    Canonical parquet writer (pyarrow table, sorted columns, snappy).
    Uses temp + atomic replace, same as atomic_write_parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    table = df.to_arrow()
    sorted_names = sorted(table.column_names)
    table = table.select(sorted_names)

    gc.collect()

    for attempt in range(_ATOMIC_MAX_RETRIES + 1):
        try:
            pq.write_table(
                table, str(tmp_path),
                version="2.6", compression="snappy",
                row_group_size=row_group_size,
                data_page_version="2.0",
                use_deprecated_int96_timestamps=False,
                coerce_timestamps="us",
            )
            tmp_path.replace(path)
            gc.collect()
            return
        except OSError as e:
            if attempt < _ATOMIC_MAX_RETRIES:
                logger.warning(
                    "Canonical parquet retry %d/%d: %s",
                    attempt + 1, _ATOMIC_MAX_RETRIES, e,
                )
                time.sleep(_ATOMIC_RETRY_DELAY)
                gc.collect()
            else:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise


def atomic_write_json(data: dict | list, path: str | Path) -> None:
    """Write JSON via temp + atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(str(tmp_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp_path.replace(path)
