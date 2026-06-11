#!/usr/bin/env python3
"""One-time Databento raw OHLCV download helper.

Default mode downloads raw Databento DBN/Zstd batch files under data/dbn. Use
--mode convert-parquet to stitch downloaded DBN files into data/raw, or --mode
all to download and convert in one run. Use --mode stream only when you
intentionally want immediate Parquet output from timeseries.get_range.
Use --convert-existing only for the legacy adjacent DBN-to-Parquet conversion.
Use --estimate-cost to estimate cost without downloading.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, TypedDict, cast
from uuid import uuid4

import pandas as pd


API_KEY_NAME = "DATABENTO_API_KEY"
API_KEY_FILE = Path(__file__).resolve().parents[2] / "databento.env"
CME_DATASET = "GLBX.MDP3"
ALLOWED_DATASETS = {CME_DATASET}
SCHEMA = "ohlcv-1m"
STYPE_IN = "continuous"
STYPE_OUT = "instrument_id"
START_YEAR = 2010
DEFAULT_RAW_OUT = "data/raw"
DEFAULT_STREAM_OUT = DEFAULT_RAW_OUT
DEFAULT_DBN_OUT = "data/dbn"
DEFAULT_BATCH_OUT = DEFAULT_DBN_OUT
DEFAULT_REPORTS_ROOT = "reports/raw_ingest"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
DATASET_AVAILABLE_START = {
    CME_DATASET: date(2010, 6, 6),
}
FATAL_ERROR_MARKERS = (
    "401",
    "402",
    "403",
    "auth_",
    "authentication failed",
    "forbidden",
    "payment required",
    "account has been locked",
    "account locked",
    "security reasons",
)
RETRYABLE_STREAM_ERROR_MARKERS = (
    "response ended prematurely",
    "streaming response",
    "connection reset",
    "read timed out",
    "timeout",
    "temporarily unavailable",
    "too many requests",
    "429",
    "503",
    "504",
)
DBN_DOWNLOAD_MODES = {"download-dbn", "all", "batch"}

CURRENT_20 = [
    "6B",
    "6E",
    "6J",
    "CL",
    "ES",
    "GC",
    "HE",
    "HG",
    "LE",
    "NG",
    "NQ",
    "RTY",
    "SI",
    "SR3",
    "YM",
    "ZB",
    "ZC",
    "ZN",
    "ZS",
    "ZW",
]

# Adds major CME/CBOT/NYMEX/COMEX products likely useful if this is truly one-shot.
# Excludes options and non-CME datasets.
EXTENDED_CME = sorted(
    set(
        CURRENT_20
        + [
            "6A",
            "6C",
            "6M",
            "6N",
            "6S",
            "E7",
            "J7",
            "PA",
            "PL",
            "QI",
            "QO",
            "RB",
            "HO",
            "UB",
            "ZF",
            "ZQ",
            "ZT",
        ]
    )
)
ALLOWED_PRODUCTS = set(EXTENDED_CME)

REQUIRED_OUTPUT_COLUMNS = {
    "ts_event",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rtype",
    "publisher_id",
    "instrument_id",
    "symbol",
}

QUALITY_OUTPUT_COLUMNS = [
    "data_quality_status",
    "data_quality_degraded",
]

REQUIRED_ARCHIVE_COLUMNS = REQUIRED_OUTPUT_COLUMNS | set(QUALITY_OUTPUT_COLUMNS)

ORDERED_OUTPUT_COLUMNS = [
    "ts_event",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rtype",
    "publisher_id",
    "instrument_id",
    "symbol",
    "data_quality_status",
    "data_quality_degraded",
]

PRICE_TYPE = "float"
PRICE_SCALE_POLICY = "databento_dbnstore_to_df_price_type_float"


@dataclass(frozen=True)
class DownloadTask:
    dataset: str
    product: str
    year: int
    start: str
    end: str
    symbol: str
    output_path: str
    schema: str = SCHEMA
    stype_in: str = STYPE_IN
    stype_out: str = STYPE_OUT
    chunk: str = "year"
    raw_format: str = "parquet"


@dataclass(frozen=True)
class DbnArchiveEntry:
    path: Path
    product: str
    year: int


class DatasetConditionInfo(TypedDict):
    raw: list[dict[str, object]]
    conditions: dict[str, str]
    degraded_dates: list[str]


class DatabentoMetadataClient(Protocol):
    def get_billable_size(self, **kwargs: object) -> object: ...

    def get_cost(self, **kwargs: object) -> float: ...

    def get_dataset_condition(self, **kwargs: object) -> list[dict[str, object]]: ...


class DatabentoTimeseriesClient(Protocol):
    def get_range(self, **kwargs: object) -> object: ...


class DatabentoBatchClient(Protocol):
    def submit_job(self, **kwargs: object) -> dict[str, Any]: ...

    def list_jobs(self, *args: object, **kwargs: object) -> list[dict[str, Any]]: ...

    def download(self, **kwargs: object) -> list[Path]: ...


class DatabentoMetadataHolder(Protocol):
    metadata: DatabentoMetadataClient


class DatabentoClient(DatabentoMetadataHolder, Protocol):
    batch: DatabentoBatchClient
    timeseries: DatabentoTimeseriesClient


class DatabentoStore(Protocol):
    def to_df(self, **kwargs: object) -> object: ...


def parse_symbols(value: str | None, universe: str) -> list[str]:
    if value:
        return sorted({item.strip().upper() for item in value.split(",") if item.strip()})
    if universe == "current20":
        return CURRENT_20
    if universe == "extended_cme":
        return EXTENDED_CME
    raise ValueError("--symbols is required when --universe custom")


def dataset_for_product(product: str) -> str:
    return CME_DATASET


def validate_allowed_dataset(dataset: str) -> str:
    if dataset not in ALLOWED_DATASETS:
        allowed = ", ".join(sorted(ALLOWED_DATASETS))
        raise ValueError(f"dataset {dataset!r} is not allowed; allowed datasets: {allowed}")
    return dataset


def validate_allowed_products(products: Iterable[str]) -> None:
    blocked = sorted({product for product in products if product not in ALLOWED_PRODUCTS})
    if blocked:
        raise ValueError(
            f"products outside the allowed {CME_DATASET} futures universe: {','.join(blocked)}"
        )


def symbol_for_product(product: str, stype_in: str) -> str:
    if stype_in == "continuous":
        return f"{product}.v.0"
    if stype_in == "parent":
        return f"{product}.FUT"
    return product


def next_year_start(value: date) -> date:
    return date(value.year + 1, 1, 1)


def next_month_start(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def iter_day_ranges(start: str, end: str) -> list[tuple[str, str]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    ranges: list[tuple[str, str]] = []
    current = start_date
    while current < end_date:
        day_end = min(current + timedelta(days=1), end_date)
        ranges.append((current.isoformat(), day_end.isoformat()))
        current = day_end
    return ranges


def iter_month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    ranges: list[tuple[str, str]] = []
    current = start_date
    while current < end_date:
        month_end = min(next_month_start(current), end_date)
        ranges.append((current.isoformat(), month_end.isoformat()))
        current = month_end
    return ranges


def iter_year_ranges(start: str, end: str) -> list[tuple[str, str]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    ranges: list[tuple[str, str]] = []
    current = start_date
    while current < end_date:
        year_end = min(next_year_start(current), end_date)
        ranges.append((current.isoformat(), year_end.isoformat()))
        current = year_end
    return ranges


def iter_chunk_ranges(start: str, end: str, chunk: str) -> list[tuple[str, str]]:
    if chunk == "day":
        return iter_day_ranges(start, end)
    if chunk == "month":
        return iter_month_ranges(start, end)
    if chunk == "year":
        return iter_year_ranges(start, end)
    raise ValueError(f"unsupported chunk: {chunk}")


def chunk_label(start: str, chunk: str) -> str:
    if chunk == "day":
        return start
    if chunk == "month":
        return start[:7]
    if chunk == "year":
        return start[:4]
    raise ValueError(f"unsupported chunk: {chunk}")


def task_output_path(
    output_root: Path,
    *,
    dataset: str,
    product: str,
    start: str,
    chunk: str,
    mode: str,
    raw_format: str,
) -> str:
    label = chunk_label(start, chunk)
    if mode in DBN_DOWNLOAD_MODES:
        year = str(date.fromisoformat(start).year)
        if chunk == "year":
            return (output_root / product / year).as_posix()
        return (output_root / product / year / label).as_posix()
    suffix = ".parquet"
    return (output_root / product / f"{label}{suffix}").as_posix()


def iter_range_tasks(
    products: Iterable[str],
    *,
    start: str,
    end: str,
    output_root: Path,
    chunk: str,
    mode: str = "stream",
    raw_format: str = "parquet",
    dataset: str | None = None,
    schema: str = SCHEMA,
    stype_in: str = STYPE_IN,
    stype_out: str = STYPE_OUT,
) -> list[DownloadTask]:
    final_end = pd.Timestamp(end).date()
    requested_start = pd.Timestamp(start).date()
    product_list = list(products)
    validate_allowed_products(product_list)
    if dataset is not None:
        validate_allowed_dataset(dataset)
    tasks: list[DownloadTask] = []
    for product in product_list:
        task_dataset = dataset or dataset_for_product(product)
        validate_allowed_dataset(task_dataset)
        dataset_start = DATASET_AVAILABLE_START.get(task_dataset, requested_start)
        range_start = max(requested_start, dataset_start)
        if range_start >= final_end:
            continue
        for task_start, task_end in iter_chunk_ranges(
            range_start.isoformat(),
            final_end.isoformat(),
            chunk,
        ):
            tasks.append(
                DownloadTask(
                    dataset=task_dataset,
                    product=product,
                    year=date.fromisoformat(task_start).year,
                    start=task_start,
                    end=task_end,
                    symbol=symbol_for_product(product, stype_in),
                    output_path=task_output_path(
                        output_root,
                        dataset=task_dataset,
                        product=product,
                        start=task_start,
                        chunk=chunk,
                        mode=mode,
                        raw_format=raw_format,
                    ),
                    schema=schema,
                    stype_in=stype_in,
                    stype_out=stype_out,
                    chunk=chunk,
                    raw_format=raw_format,
                )
            )
    return tasks


def iter_year_tasks(
    products: Iterable[str],
    *,
    start_year: int,
    end_year: int,
    end_date: str,
    output_root: Path,
) -> list[DownloadTask]:
    final_end = pd.Timestamp(end_date).date()
    product_list = list(products)
    validate_allowed_products(product_list)
    tasks: list[DownloadTask] = []
    for product in product_list:
        dataset = dataset_for_product(product)
        validate_allowed_dataset(dataset)
        dataset_start = DATASET_AVAILABLE_START.get(dataset, date(start_year, 1, 1))
        for year in range(start_year, end_year + 1):
            start = max(date(year, 1, 1), dataset_start)
            end = min(date(year + 1, 1, 1), final_end)
            if start >= end:
                continue
            tasks.append(
                DownloadTask(
                    dataset=dataset,
                    product=product,
                    year=year,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    symbol=symbol_for_product(product, STYPE_IN),
                    output_path=(output_root / product / f"{year}.parquet").as_posix(),
                )
            )
    return tasks


def normalize_api_key(value: str | None) -> str:
    if not value:
        return ""
    key = value.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    return key


def load_databento_api_key_from_file(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            return normalize_api_key(text)
        name, value = text.split("=", 1)
        if name.strip() == API_KEY_NAME:
            return normalize_api_key(value)
    return ""


def resolve_databento_api_key() -> str:
    return load_databento_api_key_from_file(API_KEY_FILE)


def get_client() -> DatabentoClient:
    key = resolve_databento_api_key()
    if not key:
        raise SystemExit(
            f"Set {API_KEY_NAME} in {API_KEY_FILE.name} at the project root."
        )
    import databento as db

    return cast(DatabentoClient, db.Historical(key))


def is_fatal_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in FATAL_ERROR_MARKERS)


def result_has_fatal_error(result: dict[str, object]) -> bool:
    return result.get("status") == "download_error" and is_fatal_error(
        RuntimeError(str(result.get("error", "")))
    )


def is_retryable_stream_error(exc: Exception) -> bool:
    if is_fatal_error(exc):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in RETRYABLE_STREAM_ERROR_MARKERS)


def first_pending_download(tasks: list[DownloadTask], *, overwrite: bool) -> DownloadTask | None:
    if overwrite:
        return tasks[0] if tasks else None
    for task in tasks:
        if not has_non_empty_output(Path(task.output_path)):
            return task
    return None


def preflight_auth(
    client: DatabentoMetadataHolder,
    tasks: list[DownloadTask],
    *,
    overwrite: bool,
) -> None:
    task = first_pending_download(tasks, overwrite=overwrite)
    if task is None:
        return
    try:
        client.metadata.get_billable_size(
            dataset=task.dataset,
            symbols=task.symbol,
            schema=task.schema,
            stype_in=task.stype_in,
            start=task.start,
            end=task.end,
        )
    except Exception as exc:
        if not is_fatal_error(exc):
            raise
        raise SystemExit(
            f"Databento rejected preflight request for {task.dataset} "
            f"{task.product} {task.year} symbol={task.symbol} "
            f"{task.start}->{task.end}. "
            "The prior OK_EXISTING lines, if any, were local file checks only. "
            f"{API_KEY_NAME} may be invalid for that dataset/symbol/date range. "
            f"Databento error: {exc}"
        ) from exc


def condition_is_degraded(status: object) -> bool:
    text = str(status).strip().lower()
    if not text or text in {"available", "ok", "normal"}:
        return False
    return True


def _condition_date(row: dict[str, object]) -> str | None:
    for key in ("date", "day", "d", "start_date"):
        value = row.get(key)
        if value:
            return pd.Timestamp(value).date().isoformat()
    return None


def _condition_status(row: dict[str, object]) -> str:
    for key in ("condition", "status", "quality", "state"):
        value = row.get(key)
        if value:
            return str(value)
    return "available"


def normalize_dataset_conditions(rows: list[dict[str, object]]) -> dict[str, str]:
    conditions: dict[str, str] = {}
    for row in rows:
        day = _condition_date(row)
        if day:
            conditions[day] = _condition_status(row)
    return conditions


def fetch_dataset_conditions(
    client: DatabentoMetadataHolder,
    task: DownloadTask,
) -> DatasetConditionInfo:
    start = date.fromisoformat(task.start)
    end_inclusive = date.fromisoformat(task.end) - timedelta(days=1)
    if start > end_inclusive:
        return {"raw": [], "conditions": {}, "degraded_dates": []}

    rows = client.metadata.get_dataset_condition(
        dataset=task.dataset,
        start_date=start.isoformat(),
        end_date=end_inclusive.isoformat(),
    )
    conditions = normalize_dataset_conditions(rows)
    degraded_dates = sorted(
        day for day, status in conditions.items() if condition_is_degraded(status)
    )
    return {"raw": rows, "conditions": conditions, "degraded_dates": degraded_dates}


def store_to_required_dataframe(
    store: DatabentoStore,
    condition_by_date: dict[str, str] | None = None,
    default_quality_status: str = "available",
) -> pd.DataFrame:
    df = store.to_df(price_type=PRICE_TYPE, pretty_ts=True, map_symbols=True)
    if not isinstance(df, pd.DataFrame):
        df = pd.concat(df, ignore_index=False)

    df = df.copy()
    if "ts_event" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            first_col = df.columns[0]
            if first_col != "ts_event":
                df = df.rename(columns={first_col: "ts_event"})
        else:
            raise ValueError("downloaded data has no ts_event column or DatetimeIndex")

    missing = sorted(REQUIRED_OUTPUT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"downloaded data missing required columns: {missing}")

    condition_by_date = condition_by_date or {}
    event_dates = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.date.astype(str)
    df["data_quality_status"] = event_dates.map(condition_by_date).fillna(default_quality_status)
    df["data_quality_degraded"] = df["data_quality_status"].map(condition_is_degraded).astype(bool)

    return df[ORDERED_OUTPUT_COLUMNS].sort_values("ts_event", kind="mergesort")


def write_store_parquet(
    store: DatabentoStore,
    path: Path,
    condition_by_date: dict[str, str] | None = None,
) -> None:
    df = store_to_required_dataframe(store, condition_by_date)
    write_required_dataframe_parquet(df, path)


def write_required_dataframe_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_parquet(tmp_path, index=False)
    check = validate_download(tmp_path)
    if not check["valid"]:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"temporary parquet failed validation: {check['errors']}")
    tmp_path.replace(path)


def validate_download(path: Path) -> dict[str, object]:
    df = pd.read_parquet(path)
    cols = set(df.columns)
    missing = sorted(REQUIRED_ARCHIVE_COLUMNS - cols)
    errors: list[str] = []
    warnings: list[str] = []
    rows = int(len(df))

    if rows == 0:
        errors.append("empty_file")
    if missing:
        errors.append(f"missing_columns:{','.join(missing)}")

    duplicate_ts_count = 0
    invalid_ts_count = 0
    timestamp_monotonic = False
    if "ts_event" in cols:
        ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        invalid_ts_count = int(ts.isna().sum())
        duplicate_ts_count = int(ts.duplicated().sum())
        timestamp_monotonic = bool(ts.is_monotonic_increasing)
        if invalid_ts_count:
            errors.append("invalid_ts_event")
        if duplicate_ts_count:
            errors.append("duplicate_ts_event")
        if not timestamp_monotonic:
            errors.append("non_monotonic_ts_event")
    else:
        errors.append("missing_ts_event")

    bad_ohlc_count = 0
    negative_volume_count = 0
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    if ohlcv_cols <= cols:
        open_ = pd.to_numeric(df["open"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        bad_ohlc = high.lt(pd.concat([open_, close], axis=1).max(axis=1)) | low.gt(
            pd.concat([open_, close], axis=1).min(axis=1)
        )
        bad_ohlc_count = int(bad_ohlc.fillna(True).sum())
        negative_volume_count = int(volume.lt(0).fillna(True).sum())
        if bad_ohlc_count:
            errors.append("bad_ohlc")
        if negative_volume_count:
            errors.append("negative_volume")

    metadata_null_counts: dict[str, int] = {}
    for column in ("rtype", "publisher_id", "instrument_id", "symbol"):
        if column in cols:
            null_count = int(df[column].isna().sum())
            metadata_null_counts[column] = null_count
            if null_count:
                errors.append(f"null_metadata:{column}")

    instrument_id_nonnull = int(df["instrument_id"].notna().sum()) if "instrument_id" in cols else 0
    symbol_nonnull = int(df["symbol"].notna().sum()) if "symbol" in cols else 0
    blank_symbol_count = 0
    if "symbol" in cols:
        blank_symbol_count = int(
            df["symbol"]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
            .sum()
        )
        if blank_symbol_count:
            errors.append("blank_symbol")
    degraded_bar_count = (
        int(df["data_quality_degraded"].fillna(False).astype(bool).sum())
        if "data_quality_degraded" in cols
        else 0
    )

    return {
        "path": path.as_posix(),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": rows,
        "timestamp_ok": "ts_event" in cols,
        "timestamp_monotonic": timestamp_monotonic,
        "invalid_ts_count": invalid_ts_count,
        "duplicate_ts_count": duplicate_ts_count,
        "missing_columns": missing,
        "bad_ohlc_count": bad_ohlc_count,
        "negative_volume_count": negative_volume_count,
        "metadata_null_counts": metadata_null_counts,
        "instrument_id_nonnull": instrument_id_nonnull,
        "symbol_nonnull": symbol_nonnull,
        "blank_symbol_count": blank_symbol_count,
        "degraded_bar_count": degraded_bar_count,
    }


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(int(item.stat().st_size) for item in path.rglob("*") if item.is_file())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_non_empty_output(path: Path) -> bool:
    return path.exists() and path_size_bytes(path) > 0


def format_speed(bytes_count: int, elapsed_seconds: float) -> tuple[float, float | None]:
    mb = bytes_count / (1024 * 1024)
    mbps = mb / elapsed_seconds if elapsed_seconds > 0 and bytes_count else None
    return mb, mbps


def log_chunk_result(
    *,
    status: str,
    task: DownloadTask,
    output_path: Path,
    elapsed_seconds: float,
    rows: int | None = None,
    bytes_count: int | None = None,
    extra: str = "",
) -> None:
    actual_bytes = path_size_bytes(output_path) if bytes_count is None else bytes_count
    mb, mbps = format_speed(actual_bytes, elapsed_seconds)
    rows_text = "unknown" if rows is None else str(rows)
    mbps_text = "unknown" if mbps is None else f"{mbps:.3f}"
    suffix = f" {extra}" if extra else ""
    print(
        f"{status.upper()} {task.dataset} {task.product} symbol={task.symbol} "
        f"{task.start}->{task.end} output={output_path.as_posix()} "
        f"rows={rows_text} bytes={actual_bytes} mb={mb:.3f} "
        f"elapsed_s={elapsed_seconds:.3f} mbps={mbps_text}{suffix}"
    )


def run_with_retries(
    action: Callable[[], dict[str, object]],
    *,
    task: DownloadTask,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, object]:
    attempt = 0
    while True:
        try:
            return action()
        except Exception as exc:
            if is_fatal_error(exc) or not is_retryable_stream_error(exc) or attempt >= max_retries:
                raise
            attempt += 1
            sleep_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"RETRY {task.dataset} {task.product} {task.start}->{task.end} "
                f"attempt={attempt}/{max_retries} sleep_s={sleep_seconds:.1f}: {exc}"
            )
            time.sleep(sleep_seconds)


def batch_split_duration_for_chunk(chunk: str) -> str:
    if chunk in {"day", "month", "year"}:
        return chunk
    return "day"


def batch_encoding_and_compression(raw_format: str) -> tuple[str, str]:
    if raw_format == "dbn-zstd":
        return "dbn", "zstd"
    if raw_format == "parquet":
        raise ValueError("Databento DBN download writes raw DBN/Zstd; use --mode convert-parquet separately")
    raise ValueError(f"unsupported raw format: {raw_format}")


def batch_job_id(job: dict[str, Any]) -> str:
    for key in ("id", "job_id", "jobId"):
        value = job.get(key)
        if value:
            return str(value)
    raise ValueError(f"Databento batch response did not include a job id: {job}")


def batch_job_state(job: dict[str, Any]) -> str:
    for key in ("state", "status"):
        value = job.get(key)
        if value:
            return str(value).lower()
    return ""


def list_batch_jobs(batch: DatabentoBatchClient) -> list[dict[str, Any]]:
    try:
        return batch.list_jobs(states=None)
    except TypeError:
        return batch.list_jobs()


def wait_for_batch_job(
    batch: DatabentoBatchClient,
    *,
    job_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any] | None:
    if timeout_seconds <= 0:
        return None
    deadline = time.monotonic() + timeout_seconds
    while True:
        jobs = list_batch_jobs(batch)
        for job in jobs:
            if str(job.get("id") or job.get("job_id") or job.get("jobId")) == job_id:
                state = batch_job_state(job)
                if state in {"done", "completed", "complete"}:
                    return job
                if state in {"failed", "expired", "cancelled", "canceled"}:
                    raise RuntimeError(f"Databento batch job {job_id} ended in state {state}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Databento batch job {job_id}")
        time.sleep(poll_seconds)


def is_dbn_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".dbn.zst") or path.name.endswith(".dbn"))


def non_empty_dbn_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if is_dbn_file(root) and path_size_bytes(root) > 0 else []
    return sorted(
        path
        for path in root.rglob("*")
        if is_dbn_file(path) and path_size_bytes(path) > 0
    )


def has_non_empty_dbn_output(path: Path) -> bool:
    return bool(non_empty_dbn_files(path))


def iter_dbn_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"DBN input root does not exist: {root}")
    if root.is_file():
        return [root] if is_dbn_file(root) else []
    return sorted(path for path in root.rglob("*") if is_dbn_file(path))


def dbn_parquet_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".parquet")


def validate_converted_parquet(
    path: Path,
    *,
    required_quality_status: str | None = None,
) -> None:
    check = validate_download(path)
    if not check["valid"]:
        raise ValueError(f"converted parquet failed validation: {check['errors']}")
    if required_quality_status is not None:
        df = pd.read_parquet(path, columns=["data_quality_status"])
        statuses = set(df["data_quality_status"].dropna().astype(str))
        if statuses != {required_quality_status}:
            raise ValueError(
                "existing converted parquet has ambiguous data_quality_status; "
                "rerun conversion with --overwrite"
            )


def convert_dbn_files_to_parquet(
    paths: list[Path],
    *,
    overwrite: bool = False,
    condition_by_date: dict[str, str] | None = None,
    default_quality_status: str = "available",
) -> list[Path]:
    import databento as db

    converted: list[Path] = []
    required_quality_status = (
        default_quality_status
        if condition_by_date is None and default_quality_status != "available"
        else None
    )
    for path in paths:
        if not is_dbn_file(path) or path_size_bytes(path) <= 0:
            continue
        parquet_path = dbn_parquet_path(path)
        if has_non_empty_output(parquet_path) and not overwrite:
            validate_converted_parquet(
                parquet_path,
                required_quality_status=required_quality_status,
            )
            converted.append(parquet_path)
            continue
        store = db.DBNStore.from_file(path)
        df = store_to_required_dataframe(
            cast(DatabentoStore, store),
            condition_by_date,
            default_quality_status=default_quality_status,
        )
        write_required_dataframe_parquet(df, parquet_path)
        validate_converted_parquet(
            parquet_path,
            required_quality_status=required_quality_status,
        )
        converted.append(parquet_path)
    return converted


def convert_existing_dbn_tree(root: Path, *, overwrite: bool = False) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    dbn_paths = iter_dbn_files(root)
    for path in dbn_paths:
        out = dbn_parquet_path(path)
        started = time.monotonic()
        try:
            skipped = has_non_empty_output(out) and not overwrite
            converted = convert_dbn_files_to_parquet(
                [path],
                overwrite=overwrite,
                default_quality_status="metadata_unavailable",
            )
            status = "ok_existing" if skipped else "ok"
            bytes_count = path_size_bytes(converted[0]) if converted else 0
            elapsed = time.monotonic() - started
            print(
                f"CONVERT_{status.upper()} input={path.as_posix()} "
                f"output={out.as_posix()} bytes={bytes_count} elapsed_s={elapsed:.3f}"
            )
            results.append(
                {
                    "status": status,
                    "input": path.as_posix(),
                    "output": out.as_posix(),
                    "bytes": bytes_count,
                    "elapsed_seconds": elapsed,
                }
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(
                f"CONVERT_ERROR input={path.as_posix()} "
                f"output={out.as_posix()} elapsed_s={elapsed:.3f}: {exc}"
            )
            results.append(
                {
                    "status": "convert_error",
                    "input": path.as_posix(),
                    "output": out.as_posix(),
                    "error": str(exc),
                    "elapsed_seconds": elapsed,
                }
            )
    return results


def infer_dbn_archive_entry(path: Path, dbn_root: Path) -> DbnArchiveEntry:
    try:
        parts = path.relative_to(dbn_root).parts
    except ValueError:
        parts = path.parts
    if len(parts) >= 2 and parts[1][:4].isdigit():
        return DbnArchiveEntry(path=path, product=parts[0], year=int(parts[1][:4]))
    if len(parts) >= 3 and parts[0] in ALLOWED_DATASETS and parts[2][:4].isdigit():
        return DbnArchiveEntry(path=path, product=parts[1], year=int(parts[2][:4]))
    raise ValueError(
        "cannot infer market/year from DBN path; expected "
        "data/dbn/{market}/{year}/...dbn.zst"
    )


def archive_entries_for_paths(
    paths: Iterable[Path],
    dbn_root: Path,
    *,
    products: set[str] | None = None,
) -> list[DbnArchiveEntry]:
    entries: list[DbnArchiveEntry] = []
    for path in sorted(paths):
        if not is_dbn_file(path) or path_size_bytes(path) <= 0:
            continue
        entry = infer_dbn_archive_entry(path, dbn_root)
        if products is not None and entry.product not in products:
            continue
        entries.append(entry)
    return entries


def dbn_paths_for_tasks(tasks: Iterable[DownloadTask]) -> list[Path]:
    paths: list[Path] = []
    for task in tasks:
        out = Path(task.output_path)
        if out.exists():
            paths.extend(non_empty_dbn_files(out))
    return sorted(set(paths))


def fetch_conditions_by_group(
    client: DatabentoMetadataHolder,
    tasks: Iterable[DownloadTask],
) -> dict[tuple[str, int], dict[str, str]]:
    conditions_by_group: dict[tuple[str, int], dict[str, str]] = {}
    for task in tasks:
        info = fetch_dataset_conditions(client, task)
        key = (task.product, task.year)
        conditions_by_group.setdefault(key, {}).update(info["conditions"])
    return conditions_by_group


def raw_parquet_summary(path: Path) -> dict[str, object]:
    df = pd.read_parquet(
        path,
        columns=["ts_event", "symbol", "data_quality_status", "data_quality_degraded"],
    )
    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    quality_counts = (
        df["data_quality_status"].fillna("missing").astype(str).value_counts().to_dict()
    )
    return {
        "row_count": int(len(df)),
        "first_ts": ts.min().isoformat() if len(ts) and not pd.isna(ts.min()) else None,
        "last_ts": ts.max().isoformat() if len(ts) and not pd.isna(ts.max()) else None,
        "decoded_symbols": sorted(
            str(value) for value in df["symbol"].dropna().unique().tolist()
        ),
        "data_quality_status_counts": {
            str(key): int(value) for key, value in quality_counts.items()
        },
        "degraded_bar_count": int(
            df["data_quality_degraded"].fillna(False).astype(bool).sum()
        ),
    }


def convert_dbn_archive_to_raw(
    dbn_root: Path,
    raw_root: Path,
    *,
    overwrite: bool = False,
    paths: Iterable[Path] | None = None,
    products: set[str] | None = None,
    condition_by_group: dict[tuple[str, int], dict[str, str]] | None = None,
    default_quality_status: str = "metadata_unavailable",
) -> list[dict[str, object]]:
    source_paths = list(paths) if paths is not None else iter_dbn_files(dbn_root)
    entries = archive_entries_for_paths(source_paths, dbn_root, products=products)
    groups: dict[tuple[str, int], list[Path]] = {}
    for entry in entries:
        groups.setdefault((entry.product, entry.year), []).append(entry.path)

    if not groups:
        return []

    results: list[dict[str, object]] = []
    for (product, year), group_paths in sorted(groups.items()):
        out = raw_root / product / f"{year}.parquet"
        started = time.monotonic()
        input_hashes = {path.as_posix(): file_sha256(path) for path in group_paths}
        conditions = (condition_by_group or {}).get((product, year))
        vendor_quality_available = conditions is not None
        data_quality_source = (
            "databento_metadata.get_dataset_condition"
            if vendor_quality_available
            else "metadata_unavailable"
        )
        quality_default = "available" if vendor_quality_available else default_quality_status
        try:
            if not vendor_quality_available:
                raise ValueError(
                    "missing dataset-condition metadata for canonical raw conversion"
                )
            skipped = has_non_empty_output(out) and not overwrite
            if skipped:
                check = validate_download(out)
                if not check["valid"]:
                    raise ValueError(f"existing raw parquet failed validation: {check['errors']}")
            else:
                import databento as db

                frames = []
                for path in group_paths:
                    store = db.DBNStore.from_file(path)
                    frames.append(
                        store_to_required_dataframe(
                            cast(DatabentoStore, store),
                            conditions,
                            default_quality_status=quality_default,
                        )
                    )
                if not frames:
                    raise ValueError("no DBN frames converted")
                df = pd.concat(frames, ignore_index=True).sort_values(
                    "ts_event",
                    kind="mergesort",
                )
                write_required_dataframe_parquet(df[ORDERED_OUTPUT_COLUMNS], out)
                check = validate_download(out)
                if not check["valid"]:
                    raise ValueError(f"raw parquet failed validation: {check['errors']}")

            summary = raw_parquet_summary(out)
            elapsed = time.monotonic() - started
            status = "ok_existing" if skipped else "ok"
            print(
                f"CONVERT_{status.upper()} market={product} year={year} "
                f"inputs={len(group_paths)} output={out.as_posix()} "
                f"rows={summary['row_count']} elapsed_s={elapsed:.3f}"
            )
            results.append(
                {
                    "status": status,
                    "market": product,
                    "year": year,
                    "input_paths": [path.as_posix() for path in group_paths],
                    "input_hashes": input_hashes,
                    "output_path": out.as_posix(),
                    "output_hash": file_sha256(out),
                    "schema": SCHEMA,
                    "required_schema_columns": ORDERED_OUTPUT_COLUMNS,
                    "raw_schema_variant": "databento_full",
                    "price_type": PRICE_TYPE,
                    "price_scale_policy": PRICE_SCALE_POLICY,
                    "data_quality_source": data_quality_source,
                    "vendor_quality_available": vendor_quality_available,
                    "validation": check,
                    "elapsed_seconds": elapsed,
                    **summary,
                }
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(
                f"CONVERT_ERROR market={product} year={year} "
                f"inputs={len(group_paths)} output={out.as_posix()} "
                f"elapsed_s={elapsed:.3f}: {exc}"
            )
            results.append(
                {
                    "status": "convert_error",
                    "market": product,
                    "year": year,
                    "input_paths": [path.as_posix() for path in group_paths],
                    "input_hashes": input_hashes,
                    "output_path": out.as_posix(),
                    "schema": SCHEMA,
                    "required_schema_columns": ORDERED_OUTPUT_COLUMNS,
                    "price_type": PRICE_TYPE,
                    "price_scale_policy": PRICE_SCALE_POLICY,
                    "data_quality_source": data_quality_source,
                    "vendor_quality_available": vendor_quality_available,
                    "error": str(exc),
                    "elapsed_seconds": elapsed,
                }
            )
    return results


def build_raw_ingest_manifest(
    results: list[dict[str, object]],
    *,
    mode: str,
    dbn_root: Path,
    raw_root: Path,
    run_id: object | None = None,
    plan_hash: object | None = None,
) -> dict[str, object]:
    output_hashes = {
        str(item["output_path"]): item.get("output_hash")
        for item in results
        if item.get("output_path")
    }
    input_hashes: dict[str, object] = {}
    for item in results:
        input_hashes.update(cast(dict[str, object], item.get("input_hashes", {})))
    failed = [item for item in results if item.get("status") == "convert_error"]
    data_quality_sources = sorted(
        {
            str(item.get("data_quality_source"))
            for item in results
            if item.get("data_quality_source")
        }
    )
    decoded_symbols = sorted(
        {
            str(symbol)
            for item in results
            for symbol in cast(list[object], item.get("decoded_symbols", []))
        }
    )
    return {
        "stage": "raw_ingest",
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "plan_hash": plan_hash,
        "dbn_root": dbn_root.as_posix(),
        "raw_root": raw_root.as_posix(),
        "schema": SCHEMA,
        "required_schema_columns": ORDERED_OUTPUT_COLUMNS,
        "price_type": PRICE_TYPE,
        "price_scale_policy": PRICE_SCALE_POLICY,
        "data_quality_fields": QUALITY_OUTPUT_COLUMNS,
        "data_quality_sources": data_quality_sources,
        "vendor_quality_available": (
            all(bool(item.get("vendor_quality_available")) for item in results)
            if results
            else False
        ),
        "decoded_symbols": decoded_symbols,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "output_count": len(output_hashes),
        "failure_count": len(failed),
        "outputs": results,
    }


def request_range_dataframe(
    client: DatabentoClient,
    task: DownloadTask,
    *,
    start: str,
    end: str,
    condition_by_date: dict[str, str],
) -> pd.DataFrame:
    data = client.timeseries.get_range(
        dataset=task.dataset,
        symbols=task.symbol,
        schema=task.schema,
        stype_in=task.stype_in,
        stype_out=task.stype_out,
        start=start,
        end=end,
    )
    return store_to_required_dataframe(cast(DatabentoStore, data), condition_by_date)


def range_midpoint(start: str, end: str) -> str | None:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    span_days = (end_date - start_date).days
    if span_days <= 1:
        return None
    return (start_date + timedelta(days=span_days // 2)).isoformat()


def concat_download_dataframes(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True).sort_values("ts_event", kind="mergesort")


def download_dataframe_with_split_retry(
    client: DatabentoClient,
    task: DownloadTask,
    *,
    condition_by_date: dict[str, str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    range_start = start or task.start
    range_end = end or task.end
    try:
        return request_range_dataframe(
            client,
            task,
            start=range_start,
            end=range_end,
            condition_by_date=condition_by_date,
        )
    except Exception as exc:
        midpoint_text = range_midpoint(range_start, range_end)
        if not is_retryable_stream_error(exc) or midpoint_text is None:
            raise

        print(
            f"RETRY_SPLIT {task.dataset} {task.product} {task.year} "
            f"{range_start}->{range_end} at {midpoint_text}: {exc}"
        )
        left = download_dataframe_with_split_retry(
            client,
            task,
            condition_by_date=condition_by_date,
            start=range_start,
            end=midpoint_text,
        )
        right = download_dataframe_with_split_retry(
            client,
            task,
            condition_by_date=condition_by_date,
            start=midpoint_text,
            end=range_end,
        )
        return concat_download_dataframes([left, right])


def download_dataframe_in_months(
    client: DatabentoClient,
    task: DownloadTask,
    *,
    condition_by_date: dict[str, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month_start, month_end in iter_month_ranges(task.start, task.end):
        print(
            f"DOWNLOAD_MONTH {task.dataset} {task.product} {task.year} "
            f"{month_start}->{month_end}"
        )
        frames.append(
            download_dataframe_with_split_retry(
                client,
                task,
                condition_by_date=condition_by_date,
                start=month_start,
                end=month_end,
            )
        )
    return concat_download_dataframes(frames)


def estimate_cost(
    client: DatabentoMetadataHolder,
    tasks: list[DownloadTask],
) -> list[dict[str, object]]:
    estimates: list[dict[str, object]] = []
    for task in tasks:
        try:
            cost = client.metadata.get_cost(
                dataset=task.dataset,
                symbols=task.symbol,
                schema=task.schema,
                stype_in=task.stype_in,
                start=task.start,
                end=task.end,
            )
            size = client.metadata.get_billable_size(
                dataset=task.dataset,
                symbols=task.symbol,
                schema=task.schema,
                stype_in=task.stype_in,
                start=task.start,
                end=task.end,
            )
        except Exception as exc:
            estimates.append({**asdict(task), "status": "estimate_error", "error": str(exc)})
            print(f"ESTIMATE_ERROR {task.dataset} {task.product} {task.year}: {exc}")
            if is_fatal_error(exc):
                print("FATAL authentication error. Stopping cost estimate run.")
                break
            continue
        estimates.append(
            {
                **asdict(task),
                "status": "ok",
                "estimated_cost_usd": cost,
                "billable_size": size,
            }
        )
        print(f"ESTIMATE {task.dataset} {task.product} {task.year}: ${cost:.4f} size={size}")
    return estimates


def execute_download(
    client: DatabentoClient,
    tasks: list[DownloadTask],
    *,
    overwrite: bool,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    condition_cache: dict[tuple[str, str, str], DatasetConditionInfo] = {}
    for task in tasks:
        out = Path(task.output_path)
        task_started = time.monotonic()
        if has_non_empty_output(out) and not overwrite:
            try:
                check = validate_download(out)
                status = "ok_existing" if check["valid"] else "bad_existing"
                results.append({**asdict(task), "status": status, "validation": check})
                log_chunk_result(
                    status=status,
                    task=task,
                    output_path=out,
                    elapsed_seconds=time.monotonic() - task_started,
                    rows=int(check["rows"]),
                )
            except Exception as exc:
                results.append({**asdict(task), "status": "bad_existing", "error": str(exc)})
                log_chunk_result(
                    status="bad_existing",
                    task=task,
                    output_path=out,
                    elapsed_seconds=time.monotonic() - task_started,
                    extra=str(exc),
                )
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            def download_once() -> dict[str, object]:
                condition_key = (task.dataset, task.start, task.end)
                if condition_key not in condition_cache:
                    condition_cache[condition_key] = fetch_dataset_conditions(client, task)
                condition_info = condition_cache[condition_key]
                df = download_dataframe_in_months(
                    client,
                    task,
                    condition_by_date=condition_info["conditions"],
                )
                write_required_dataframe_parquet(df, out)
                check = validate_download(out)
                status = "ok" if check["valid"] else "bad_schema"
                return {
                    "status": status,
                    "validation": check,
                    "dataset_condition": {
                        "degraded_dates": condition_info["degraded_dates"],
                        "degraded_date_count": len(condition_info["degraded_dates"]),
                    },
                }

            payload = run_with_retries(
                download_once,
                task=task,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            check = cast(dict[str, object], payload["validation"])
            status = str(payload["status"])
            results.append(
                {
                    **asdict(task),
                    "status": status,
                    "validation": check,
                    "dataset_condition": payload["dataset_condition"],
                }
            )
            degraded = cast(dict[str, object], payload["dataset_condition"])[
                "degraded_date_count"
            ]
            log_chunk_result(
                status=status,
                task=task,
                output_path=out,
                elapsed_seconds=time.monotonic() - task_started,
                rows=int(check["rows"]),
                extra=f"degraded_dates={degraded}",
            )
        except Exception as exc:
            results.append({**asdict(task), "status": "download_error", "error": str(exc)})
            log_chunk_result(
                status="download_error",
                task=task,
                output_path=out,
                elapsed_seconds=time.monotonic() - task_started,
                extra=str(exc),
            )
            if is_fatal_error(exc):
                print("FATAL authentication error. Stopping download run.")
                break
    return results


def execute_stream_downloads(
    tasks: list[DownloadTask],
    *,
    overwrite: bool,
    workers: int,
    client_factory: Callable[[], DatabentoClient],
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> list[dict[str, object]]:
    if workers <= 1:
        return execute_download(
            client_factory(),
            tasks,
            overwrite=overwrite,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    def run_one(task: DownloadTask) -> list[dict[str, object]]:
        return execute_download(
            client_factory(),
            [task],
            overwrite=overwrite,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    results_by_index: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
                results_by_index[index] = result[0] if result else {
                    **asdict(tasks[index]),
                    "status": "download_error",
                    "error": "worker returned no result",
                }
            except Exception as exc:
                results_by_index[index] = {
                    **asdict(tasks[index]),
                    "status": "download_error",
                    "error": str(exc),
                }
    return [results_by_index[index] for index in range(len(tasks))]


def execute_batch_task(
    client: DatabentoClient,
    task: DownloadTask,
    *,
    overwrite: bool,
    convert_parquet: bool,
    batch_wait_timeout_seconds: float,
    batch_poll_seconds: float,
    raise_errors: bool = False,
) -> dict[str, object]:
    out = Path(task.output_path)
    started = time.monotonic()
    if has_non_empty_dbn_output(out) and not overwrite:
        log_chunk_result(
            status="ok_existing",
            task=task,
            output_path=out,
            elapsed_seconds=time.monotonic() - started,
        )
        return {
            **asdict(task),
            "status": "ok_existing",
            "output_role": "archive_only",
            "pipeline_raw_ready": False,
            "bytes": path_size_bytes(out),
        }

    encoding, compression = batch_encoding_and_compression(task.raw_format)
    tmp_dir = out.with_name(f"{out.name}.tmp-{time.time_ns()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    try:
        job = client.batch.submit_job(
            dataset=task.dataset,
            symbols=task.symbol,
            schema=task.schema,
            stype_in=task.stype_in,
            stype_out=task.stype_out,
            start=task.start,
            end=task.end,
            encoding=encoding,
            compression=compression,
            delivery="download",
            split_duration=batch_split_duration_for_chunk(task.chunk),
        )
        job_id = batch_job_id(job)
        waited_job = wait_for_batch_job(
            client.batch,
            job_id=job_id,
            timeout_seconds=batch_wait_timeout_seconds,
            poll_seconds=batch_poll_seconds,
        )
        downloaded = client.batch.download(job_id=job_id, output_dir=tmp_dir)
        downloaded_paths = [Path(path) for path in downloaded]
        dbn_paths = [path for path in downloaded_paths if is_dbn_file(path) and path_size_bytes(path) > 0]
        if not dbn_paths:
            raise RuntimeError("Databento batch download produced no non-empty DBN files")
        condition_info = fetch_dataset_conditions(client, task) if convert_parquet else None
        converted_paths = (
            convert_dbn_files_to_parquet(
                dbn_paths,
                overwrite=True,
                condition_by_date=condition_info["conditions"] if condition_info else None,
            )
            if convert_parquet
            else []
        )

        if out.exists():
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.replace(out)

        bytes_count = path_size_bytes(out)
        elapsed = time.monotonic() - started
        log_chunk_result(
            status="ok",
            task=task,
            output_path=out,
            elapsed_seconds=elapsed,
            bytes_count=bytes_count,
            extra=f"job_id={job_id} files={len(downloaded_paths)} dbn_files={len(dbn_paths)}",
        )
        return {
            **asdict(task),
            "status": "ok",
            "output_role": "archive_only",
            "pipeline_raw_ready": False,
            "job": waited_job or job,
            "job_id": job_id,
            "downloaded_files": [path.as_posix() for path in downloaded_paths],
            "downloaded_dbn_files": [path.as_posix() for path in dbn_paths],
            "converted_parquet_files": [
                (out / path.relative_to(tmp_dir)).as_posix() for path in converted_paths
            ],
            "dataset_condition": (
                {
                    "degraded_dates": condition_info["degraded_dates"],
                    "degraded_date_count": len(condition_info["degraded_dates"]),
                }
                if condition_info
                else None
            ),
            "bytes": bytes_count,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        log_chunk_result(
            status="download_error",
            task=task,
            output_path=out,
            elapsed_seconds=time.monotonic() - started,
            extra=str(exc),
        )
        if raise_errors:
            raise
        return {**asdict(task), "status": "download_error", "error": str(exc)}


def execute_batch_downloads(
    tasks: list[DownloadTask],
    *,
    overwrite: bool,
    workers: int,
    client_factory: Callable[[], DatabentoClient],
    convert_parquet: bool,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    batch_wait_timeout_seconds: float = 3600.0,
    batch_poll_seconds: float = 30.0,
) -> list[dict[str, object]]:
    def run_task(task: DownloadTask) -> dict[str, object]:
        try:
            return run_with_retries(
                lambda: execute_batch_task(
                    client_factory(),
                    task,
                    overwrite=overwrite,
                    convert_parquet=convert_parquet,
                    batch_wait_timeout_seconds=batch_wait_timeout_seconds,
                    batch_poll_seconds=batch_poll_seconds,
                    raise_errors=True,
                ),
                task=task,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        except Exception as exc:
            return {**asdict(task), "status": "download_error", "error": str(exc)}

    if workers <= 1:
        results: list[dict[str, object]] = []
        for task in tasks:
            result = run_task(task)
            results.append(result)
            if result_has_fatal_error(result):
                print("FATAL Databento account/auth error. Stopping batch download run.")
                break
        return results

    results_by_index: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_task, task): index for index, task in enumerate(tasks)}
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results_by_index[index] = result
            if result_has_fatal_error(result):
                print("FATAL Databento account/auth error. Cancelling pending batch jobs.")
                for pending in futures:
                    pending.cancel()
                break
    return [results_by_index[index] for index in sorted(results_by_index)]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_role_for_run(mode: str, raw_format: str, output_root: Path) -> str:
    if mode in {"download-dbn", "batch"}:
        return "dbn_archive"
    if mode == "all":
        return "dbn_archive_and_pipeline_raw_parquet"
    if mode == "convert-parquet":
        return "pipeline_raw_parquet"
    if mode == "stream" and raw_format == "parquet" and output_root == Path(DEFAULT_STREAM_OUT):
        return "pipeline_raw_parquet"
    if mode == "stream" and raw_format == "parquet":
        return "stream_parquet_noncanonical"
    return "archive_only"


def pipeline_raw_ready_for_run(mode: str, raw_format: str, output_root: Path) -> bool:
    return output_role_for_run(mode, raw_format, output_root) in {
        "pipeline_raw_parquet",
        "dbn_archive_and_pipeline_raw_parquet",
    }


def dry_run_plan_path(plan_out: Path) -> Path:
    return plan_out.with_name(f"{plan_out.stem}_dry_run{plan_out.suffix}")


def finalize_plan_provenance(plan: dict[str, object], *, run_kind: str) -> dict[str, object]:
    plan["generated_at"] = datetime.now(timezone.utc).isoformat()
    plan["run_id"] = uuid4().hex
    plan["run_kind"] = run_kind
    plan["plan_hash"] = canonical_json_hash(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    return plan


def add_result_provenance(
    results: list[dict[str, object]],
    plan: dict[str, object],
) -> list[dict[str, object]]:
    return [
        {
            **item,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "mode": plan.get("mode"),
            "chunk": plan.get("chunk"),
            "raw_format": plan.get("raw_format"),
            "output_role": plan.get("output_role"),
            "pipeline_raw_ready": plan.get("pipeline_raw_ready"),
        }
        for item in results
    ]


def effective_date_range(args: argparse.Namespace) -> tuple[str, str]:
    start = pd.Timestamp(args.start).date() if args.start else date(args.start_year, 1, 1)
    if args.end:
        end = pd.Timestamp(args.end).date()
    else:
        end = min(pd.Timestamp(args.end_date).date(), date(args.end_year + 1, 1, 1))
    if start >= end:
        raise SystemExit(f"Invalid date range: start {start.isoformat()} must be before end {end.isoformat()}")
    return start.isoformat(), end.isoformat()


def effective_raw_format(args: argparse.Namespace) -> str:
    if args.raw_format:
        return str(args.raw_format)
    return "parquet" if args.mode == "stream" else "dbn-zstd"


def effective_output_root(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    if args.mode in DBN_DOWNLOAD_MODES:
        return Path(args.dbn_root or DEFAULT_DBN_OUT)
    if args.mode == "convert-parquet":
        return Path(args.dbn_root or DEFAULT_DBN_OUT)
    return Path(args.raw_root or DEFAULT_RAW_OUT)


def effective_raw_root(args: argparse.Namespace) -> Path:
    return Path(args.raw_root or DEFAULT_RAW_OUT)


def effective_reports_root(args: argparse.Namespace) -> Path:
    return Path(args.reports_root or DEFAULT_REPORTS_ROOT)


def effective_plan_out(args: argparse.Namespace) -> Path:
    if args.plan_out:
        return Path(args.plan_out)
    return effective_reports_root(args) / "databento_download_plan.json"


def report_path(args: argparse.Namespace, name: str) -> Path:
    if args.plan_out:
        return Path(args.plan_out).with_name(name)
    return effective_reports_root(args) / name


def print_dry_run(tasks: list[DownloadTask]) -> None:
    print(f"DRY_RUN total_planned_chunks={len(tasks)}")
    for task in tasks:
        print(
            f"DRY_RUN_JOB dataset={task.dataset} market={task.product} "
            f"symbol={task.symbol} schema={task.schema} stype_in={task.stype_in} "
            f"start={task.start} end={task.end} chunk={task.chunk} "
            f"raw_format={task.raw_format} output={task.output_path}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Default raw DBN/Zstd batch output under data/dbn.
  python scripts\\phase1A_download\\download_databento_raw.py --markets ES,NQ --start 2023-01-01 --end 2024-01-01 --chunk month --workers 1 --resume

  # Fast planning check with monthly chunks and no API calls.
  python scripts\\phase1A_download\\download_databento_raw.py --markets ES,NQ --start 2023-01-01 --end 2023-03-01 --chunk month --dry-run

  # Convert already-downloaded DBN/Zstd files to data/raw/{market}/{year}.parquet.
  python scripts\\phase1B_convert\\convert_databento_raw.py --dbn-root data/dbn --raw-root data/raw

  # Download DBN/Zstd, then convert to canonical raw Parquet.
  python scripts\\phase1A_download\\download_databento_raw.py --mode all --markets ES,NQ --start 2023-01-01 --end 2024-01-01

  # Intentional old behavior: immediate yearly Parquet stream output under data/raw.
  python scripts\\phase1A_download\\download_databento_raw.py --mode stream --raw-format parquet --markets ES,NQ --start-year 2023 --end-year 2025
""",
    )
    parser.add_argument(
        "--universe",
        choices=["current20", "extended_cme", "custom"],
        default="extended_cme",
    )
    parser.add_argument("--symbols", "--markets", dest="symbols", help="Comma-separated product roots, e.g. ES,NQ,CL")
    parser.add_argument("--dataset", help=f"Override dataset for every requested market; only {CME_DATASET} is allowed")
    parser.add_argument("--schema", default=SCHEMA)
    parser.add_argument("--stype-in", default=STYPE_IN, help="Default continuous. Use parent for symbols like ES.FUT.")
    parser.add_argument("--stype-out", default=STYPE_OUT)
    parser.add_argument("--start", help="Inclusive start date, e.g. 2023-01-01. Overrides --start-year.")
    parser.add_argument("--end", help="Exclusive end date, e.g. 2026-01-01. Overrides --end-year/--end-date.")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--dbn-root", default=DEFAULT_DBN_OUT)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_OUT)
    parser.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--out", help="Legacy output root override; prefer --dbn-root or --raw-root.")
    parser.add_argument("--plan-out", help="Override download plan path; defaults under --reports-root.")
    parser.add_argument("--chunk", choices=["day", "month", "year"], default="year")
    parser.add_argument(
        "--mode",
        choices=["download-dbn", "convert-parquet", "all", "stream", "batch"],
        default="download-dbn",
    )
    parser.add_argument("--raw-format", choices=["parquet", "dbn-zstd"])
    parser.add_argument("--workers", type=int, default=1, help="Bounded concurrent market/chunk jobs. Use 3-4 for this machine.")
    parser.add_argument("--resume", action="store_true", help="Explicitly keep skip/resume behavior; existing non-empty final outputs are skipped unless --overwrite is set.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs and exit without API calls.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--batch-wait-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--batch-poll-seconds", type=float, default=30.0)
    parser.add_argument("--convert-parquet", action="store_true", help="Legacy batch option: also write adjacent parquet conversions after DBN download.")
    parser.add_argument("--convert-existing", action="store_true", help="Convert existing local .dbn/.dbn.zst files to adjacent Parquet files and exit without API calls.")
    parser.add_argument("--convert-in", help="Input file or root directory for legacy --convert-existing. Defaults to --dbn-root.")
    parser.add_argument("--estimate-cost", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.batch_poll_seconds <= 0:
        raise SystemExit("--batch-poll-seconds must be > 0")

    if args.convert_existing:
        convert_in = Path(args.convert_in or args.dbn_root)
        results = convert_existing_dbn_tree(convert_in, overwrite=args.overwrite)
        write_json(report_path(args, "databento_convert_results.json"), results)
        failed = [item for item in results if item.get("status") == "convert_error"]
        print(f"CONVERT_EXISTING total={len(results)} failed={len(failed)}")
        return 1 if failed else 0

    if args.mode == "convert-parquet":
        products = None
        if args.symbols:
            try:
                products = set(parse_symbols(args.symbols, "custom"))
                validate_allowed_products(products)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        dbn_root = effective_output_root(args)
        raw_root = effective_raw_root(args)
        results = convert_dbn_archive_to_raw(
            dbn_root,
            raw_root,
            overwrite=args.overwrite,
            products=products,
        )
        write_json(report_path(args, "databento_convert_results.json"), results)
        write_json(
            report_path(args, "raw_ingest_manifest.json"),
            build_raw_ingest_manifest(
                results,
                mode=args.mode,
                dbn_root=dbn_root,
                raw_root=raw_root,
            ),
        )
        failed = [item for item in results if item.get("status") == "convert_error"]
        print(f"CONVERT_PARQUET total={len(results)} failed={len(failed)}")
        return 1 if failed else 0

    try:
        products = parse_symbols(args.symbols, args.universe)
        validate_allowed_products(products)
        if args.dataset:
            validate_allowed_dataset(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    start, end = effective_date_range(args)
    raw_format = effective_raw_format(args)
    if args.mode == "stream" and raw_format != "parquet":
        raise SystemExit("--mode stream requires --raw-format parquet")
    if args.mode in DBN_DOWNLOAD_MODES and raw_format != "dbn-zstd":
        raise SystemExit(f"--mode {args.mode} currently requires --raw-format dbn-zstd")

    output_root = effective_output_root(args)
    raw_root = effective_raw_root(args)
    reports_root = effective_reports_root(args)
    plan_out = effective_plan_out(args)
    output_role = output_role_for_run(args.mode, raw_format, output_root)
    pipeline_raw_ready = pipeline_raw_ready_for_run(args.mode, raw_format, output_root)
    tasks = iter_range_tasks(
        products,
        start=start,
        end=end,
        output_root=output_root,
        chunk=args.chunk,
        mode=args.mode,
        raw_format=raw_format,
        dataset=args.dataset,
        schema=args.schema,
        stype_in=args.stype_in,
        stype_out=args.stype_out,
    )

    plan = {
        "mode": args.mode,
        "chunk": args.chunk,
        "raw_format": raw_format,
        "schema": args.schema,
        "stype_in": args.stype_in,
        "stype_out": args.stype_out,
        "start": start,
        "end": end,
        "universe": args.universe,
        "product_count": len(products),
        "task_count": len(tasks),
        "datasets": sorted({task.dataset for task in tasks}),
        "products": products,
        "workers": args.workers,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "dbn_root": output_root.as_posix() if args.mode in DBN_DOWNLOAD_MODES else None,
        "raw_root": raw_root.as_posix(),
        "reports_root": reports_root.as_posix(),
        "required_schema_columns": ORDERED_OUTPUT_COLUMNS,
        "data_quality_fields": QUALITY_OUTPUT_COLUMNS,
        "price_type": PRICE_TYPE,
        "price_scale_policy": PRICE_SCALE_POLICY,
        "output_role": output_role,
        "pipeline_raw_ready": pipeline_raw_ready,
        "archive_only": output_role in {"archive_only", "dbn_archive"},
        "tasks": [asdict(task) for task in tasks],
    }
    plan = finalize_plan_provenance(
        plan,
        run_kind="dry_run" if args.dry_run else ("estimate" if args.estimate_cost else "download"),
    )
    print(
        f"PLAN mode={args.mode} chunk={args.chunk} products={len(products)} "
        f"tasks={len(tasks)} out={output_root.as_posix()} workers={args.workers} "
        f"output_role={output_role} pipeline_raw_ready={pipeline_raw_ready}"
    )

    if args.dry_run:
        write_json(dry_run_plan_path(plan_out), plan)
        print_dry_run(tasks)
        return 0

    write_json(plan_out, plan)

    client = get_client()
    if args.estimate_cost:
        estimates = estimate_cost(client, tasks)
        estimates = add_result_provenance(estimates, plan)
        write_json(report_path(args, "databento_cost_estimate.json"), estimates)
        total = sum(float(item.get("estimated_cost_usd", 0.0)) for item in estimates)
        errors = sum(1 for item in estimates if item.get("status") == "estimate_error")
        print(f"TOTAL_ESTIMATED_COST_USD {total:.4f}")
        print(f"TOTAL_ESTIMATE_ERRORS {errors}")
        return 0

    preflight_auth(client, tasks, overwrite=args.overwrite)
    if args.mode == "stream":
        if args.workers <= 1:
            results = execute_download(
                client,
                tasks,
                overwrite=args.overwrite,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
        else:
            results = execute_stream_downloads(
                tasks,
                overwrite=args.overwrite,
                workers=args.workers,
                client_factory=get_client,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
    else:
        results = execute_batch_downloads(
            tasks,
            overwrite=args.overwrite,
            workers=args.workers,
            client_factory=get_client,
            convert_parquet=args.convert_parquet and args.mode != "all",
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            batch_wait_timeout_seconds=args.batch_wait_timeout_seconds,
            batch_poll_seconds=args.batch_poll_seconds,
        )
    results = add_result_provenance(results, plan)
    write_json(report_path(args, "databento_download_results.json"), results)
    failed = [item for item in results if item.get("status") not in {"ok", "ok_existing"}]
    if args.mode == "all" and not failed:
        conditions = fetch_conditions_by_group(client, tasks)
        convert_results = convert_dbn_archive_to_raw(
            output_root,
            raw_root,
            overwrite=args.overwrite,
            paths=dbn_paths_for_tasks(tasks),
            condition_by_group=conditions,
        )
        convert_results = add_result_provenance(convert_results, plan)
        write_json(report_path(args, "databento_convert_results.json"), convert_results)
        write_json(
            report_path(args, "raw_ingest_manifest.json"),
            build_raw_ingest_manifest(
                convert_results,
                mode=args.mode,
                dbn_root=output_root,
                raw_root=raw_root,
                run_id=plan["run_id"],
                plan_hash=plan["plan_hash"],
            ),
        )
        failed.extend(
            item for item in convert_results if item.get("status") == "convert_error"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
