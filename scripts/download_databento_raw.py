#!/usr/bin/env python3
"""One-time Databento raw OHLCV download helper.

Default mode is offline plan only. It does not spend credits unless
--estimate-cost or --execute is passed.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Protocol, TypedDict, cast

import pandas as pd


CME_DATASET = "GLBX.MDP3"
CFE_DATASET = "XCBF.PITCH"
SCHEMA = "ohlcv-1m"
STYPE_IN = "continuous"
STYPE_OUT = "instrument_id"
START_YEAR = 2010
DATASET_AVAILABLE_START = {
    CME_DATASET: date(2010, 6, 6),
}
FATAL_ERROR_MARKERS = (
    "401",
    "auth_authentication_failed",
    "authentication failed",
)

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
            "M2K",
            "MCL",
            "MES",
            "MGC",
            "MNQ",
            "MYM",
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

CFE_VIX = ["VX", "VXM"]

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


@dataclass(frozen=True)
class DownloadTask:
    dataset: str
    product: str
    year: int
    start: str
    end: str
    symbol: str
    output_path: str


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


class DatabentoMetadataHolder(Protocol):
    metadata: DatabentoMetadataClient


class DatabentoClient(DatabentoMetadataHolder, Protocol):
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
    if universe == "extended_cme_vix":
        return EXTENDED_CME + CFE_VIX
    if universe == "vix":
        return CFE_VIX
    raise ValueError("--symbols is required when --universe custom")


def dataset_for_product(product: str) -> str:
    if product in CFE_VIX:
        return CFE_DATASET
    return CME_DATASET


def iter_year_tasks(
    products: Iterable[str],
    *,
    start_year: int,
    end_year: int,
    end_date: str,
    output_root: Path,
) -> list[DownloadTask]:
    final_end = pd.Timestamp(end_date).date()
    tasks: list[DownloadTask] = []
    for product in products:
        dataset = dataset_for_product(product)
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
                    symbol=f"{product}.v.0",
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


def get_client() -> DatabentoClient:
    key = normalize_api_key(os.environ.get("DATABENTO_API_KEY"))
    if not key:
        raise SystemExit("Set DATABENTO_API_KEY in the environment. Do not store it in files.")
    import databento as db

    return cast(DatabentoClient, db.Historical(key))


def offline_mode_message(*, key_set: bool) -> str:
    key_status = "set" if key_set else "not set"
    return (
        "Nothing downloaded. A plain run only writes the download plan. "
        f"DATABENTO_API_KEY is {key_status}. Rerun with --execute to download raw data. "
        "Use --estimate-cost first if you want to review Databento cost before downloading."
    )


def is_fatal_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in FATAL_ERROR_MARKERS)


def first_pending_download(tasks: list[DownloadTask], *, overwrite: bool) -> DownloadTask | None:
    if overwrite:
        return tasks[0] if tasks else None
    for task in tasks:
        if not Path(task.output_path).exists():
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
            schema=SCHEMA,
            stype_in=STYPE_IN,
            start=task.start,
            end=task.end,
        )
    except Exception as exc:
        if not is_fatal_error(exc):
            raise
        raise SystemExit(
            "Databento rejected DATABENTO_API_KEY before download. "
            "The prior OK_EXISTING lines, if any, were local file checks only. "
            "Set a valid key in this PowerShell session and rerun with --execute."
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
) -> pd.DataFrame:
    df = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
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
    df["data_quality_status"] = event_dates.map(condition_by_date).fillna("available")
    df["data_quality_degraded"] = df["data_quality_status"].map(condition_is_degraded).astype(bool)

    return df[ORDERED_OUTPUT_COLUMNS].sort_values("ts_event", kind="mergesort")


def write_store_parquet(
    store: DatabentoStore,
    path: Path,
    condition_by_date: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = store_to_required_dataframe(store, condition_by_date)
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

    instrument_id_nonnull = int(df["instrument_id"].notna().sum()) if "instrument_id" in cols else 0
    symbol_nonnull = int(df["symbol"].notna().sum()) if "symbol" in cols else 0
    if not instrument_id_nonnull:
        warnings.append("instrument_id_missing_or_null")
    if not symbol_nonnull:
        warnings.append("symbol_missing_or_null")
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
        "instrument_id_nonnull": instrument_id_nonnull,
        "symbol_nonnull": symbol_nonnull,
        "degraded_bar_count": degraded_bar_count,
    }


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
                schema=SCHEMA,
                stype_in=STYPE_IN,
                start=task.start,
                end=task.end,
            )
            size = client.metadata.get_billable_size(
                dataset=task.dataset,
                symbols=task.symbol,
                schema=SCHEMA,
                stype_in=STYPE_IN,
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
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    condition_cache: dict[tuple[str, str, str], DatasetConditionInfo] = {}
    for task in tasks:
        out = Path(task.output_path)
        if out.exists() and not overwrite:
            try:
                check = validate_download(out)
                status = "ok_existing" if check["valid"] else "bad_existing"
                results.append({**asdict(task), "status": status, "validation": check})
                print(f"{status.upper()} {task.product} {task.year}: rows={check['rows']}")
            except Exception as exc:
                results.append({**asdict(task), "status": "bad_existing", "error": str(exc)})
                print(f"BAD_EXISTING {task.product} {task.year}: {exc}")
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            condition_key = (task.dataset, task.start, task.end)
            if condition_key not in condition_cache:
                condition_cache[condition_key] = fetch_dataset_conditions(client, task)
            condition_info = condition_cache[condition_key]
            data = client.timeseries.get_range(
                dataset=task.dataset,
                symbols=task.symbol,
                schema=SCHEMA,
                stype_in=STYPE_IN,
                stype_out=STYPE_OUT,
                start=task.start,
                end=task.end,
            )
            write_store_parquet(cast(DatabentoStore, data), out, condition_info["conditions"])
            check = validate_download(out)
            status = "ok" if check["valid"] else "bad_schema"
            results.append(
                {
                    **asdict(task),
                    "status": status,
                    "validation": check,
                    "dataset_condition": {
                        "degraded_dates": condition_info["degraded_dates"],
                        "degraded_date_count": len(condition_info["degraded_dates"]),
                    },
                }
            )
            degraded = len(condition_info["degraded_dates"])
            print(
                f"{status.upper()} {task.dataset} {task.product} {task.year}: "
                f"rows={check['rows']} degraded_dates={degraded}"
            )
        except Exception as exc:
            results.append({**asdict(task), "status": "download_error", "error": str(exc)})
            print(f"DOWNLOAD_ERROR {task.dataset} {task.product} {task.year}: {exc}")
            if is_fatal_error(exc):
                print("FATAL authentication error. Stopping download run.")
                break
    return results


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        choices=["current20", "extended_cme", "extended_cme_vix", "vix", "custom"],
        default="extended_cme_vix",
    )
    parser.add_argument("--symbols", help="Comma-separated product roots, e.g. ES,NQ,CL")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--plan-out", default="reports/databento_download_plan.json")
    parser.add_argument("--estimate-cost", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    products = parse_symbols(args.symbols, args.universe)
    tasks = iter_year_tasks(
        products,
        start_year=args.start_year,
        end_year=args.end_year,
        end_date=args.end_date,
        output_root=Path(args.out),
    )

    plan = {
        "schema": SCHEMA,
        "stype_in": STYPE_IN,
        "stype_out": STYPE_OUT,
        "universe": args.universe,
        "product_count": len(products),
        "task_count": len(tasks),
        "datasets": sorted({task.dataset for task in tasks}),
        "products": products,
        "tasks": [asdict(task) for task in tasks],
    }
    write_json(Path(args.plan_out), plan)
    print(f"PLAN products={len(products)} tasks={len(tasks)} out={args.out}")

    if not args.estimate_cost and not args.execute:
        print(offline_mode_message(key_set=bool(os.environ.get("DATABENTO_API_KEY"))))
        return 0

    client = get_client()
    if args.estimate_cost:
        estimates = estimate_cost(client, tasks)
        write_json(Path(args.plan_out).with_name("databento_cost_estimate.json"), estimates)
        total = sum(float(item.get("estimated_cost_usd", 0.0)) for item in estimates)
        errors = sum(1 for item in estimates if item.get("status") == "estimate_error")
        print(f"TOTAL_ESTIMATED_COST_USD {total:.4f}")
        print(f"TOTAL_ESTIMATE_ERRORS {errors}")

    if args.execute:
        preflight_auth(client, tasks, overwrite=args.overwrite)
        results = execute_download(client, tasks, overwrite=args.overwrite)
        write_json(Path(args.plan_out).with_name("databento_download_results.json"), results)
        failed = [item for item in results if item.get("status") not in {"ok", "ok_existing"}]
        return 1 if failed else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
