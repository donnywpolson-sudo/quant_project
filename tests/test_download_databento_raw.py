from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.download_databento_raw import (
    CFE_DATASET,
    CFE_VIX,
    CME_DATASET,
    CURRENT_20,
    EXTENDED_CME,
    STYPE_IN,
    STYPE_OUT,
    DownloadTask,
    build_arg_parser,
    condition_is_degraded,
    dataset_for_product,
    execute_download,
    estimate_cost,
    first_pending_download,
    is_fatal_error,
    iter_year_tasks,
    normalize_api_key,
    offline_mode_message,
    parse_symbols,
    preflight_auth,
    store_to_required_dataframe,
    validate_download,
    write_store_parquet,
)


class FakeStore:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def to_df(self, **kwargs: object) -> pd.DataFrame:
        return self.df


class FailingTimeseries:
    def get_range(self, **kwargs: object) -> object:
        raise RuntimeError("401 auth_authentication_failed Authentication failed.")


class FailingMetadata:
    def get_dataset_condition(self, **kwargs: object) -> list[dict[str, object]]:
        return []

    def get_billable_size(self, **kwargs: object) -> object:
        return 0

    def get_cost(self, **kwargs: object) -> float:
        return 0.0


class FailingClient:
    metadata = FailingMetadata()
    timeseries = FailingTimeseries()


class AuthFailingEstimateMetadata:
    def get_cost(self, **kwargs: object) -> float:
        raise RuntimeError("401 auth_authentication_failed Authentication failed.")

    def get_billable_size(self, **kwargs: object) -> object:
        raise AssertionError("get_billable_size should not be called after auth failure")

    def get_dataset_condition(self, **kwargs: object) -> list[dict[str, object]]:
        return []


class AuthFailingEstimateClient:
    metadata = AuthFailingEstimateMetadata()


class AuthFailingPreflightMetadata:
    def get_billable_size(self, **kwargs: object) -> object:
        raise RuntimeError("401 auth_authentication_failed Authentication failed.")

    def get_cost(self, **kwargs: object) -> float:
        return 0.0

    def get_dataset_condition(self, **kwargs: object) -> list[dict[str, object]]:
        return []


class AuthFailingPreflightClient:
    metadata = AuthFailingPreflightMetadata()


def test_parse_symbols_current_and_extended() -> None:
    assert parse_symbols(None, "current20") == CURRENT_20
    assert "ES" in parse_symbols(None, "extended_cme")
    assert "MES" in parse_symbols(None, "extended_cme")
    assert "VX" in parse_symbols(None, "extended_cme_vix")
    assert "VXM" in parse_symbols(None, "extended_cme_vix")
    assert parse_symbols(None, "vix") == CFE_VIX
    assert len(EXTENDED_CME) > len(CURRENT_20)


def test_parse_symbols_custom_normalizes_and_sorts() -> None:
    assert parse_symbols(" es,CL, es ", "custom") == ["CL", "ES"]


def test_default_output_is_pipeline_raw_root() -> None:
    args = build_arg_parser().parse_args([])
    assert args.out == "data/raw"


def test_continuous_requests_use_supported_output_symbology() -> None:
    assert STYPE_IN == "continuous"
    assert STYPE_OUT == "instrument_id"


def test_normalize_api_key_strips_wrapping_noise() -> None:
    assert normalize_api_key(None) == ""
    assert normalize_api_key("  db-test  ") == "db-test"
    assert normalize_api_key('"db-test"') == "db-test"
    assert normalize_api_key("'db-test'") == "db-test"


def test_offline_mode_message_says_nothing_downloaded() -> None:
    assert "Nothing downloaded" in offline_mode_message(key_set=True)
    assert "--execute" in offline_mode_message(key_set=True)
    assert "DATABENTO_API_KEY is set" in offline_mode_message(key_set=True)
    assert "DATABENTO_API_KEY is not set" in offline_mode_message(key_set=False)


def test_condition_is_degraded_classifies_quality_status() -> None:
    assert condition_is_degraded("available") is False
    assert condition_is_degraded("degraded") is True
    assert condition_is_degraded("pending") is True
    assert condition_is_degraded("missing") is True
    assert condition_is_degraded("partial") is True


def test_is_fatal_error_detects_auth_failure() -> None:
    assert is_fatal_error(RuntimeError("401 auth_authentication_failed")) is True
    assert is_fatal_error(RuntimeError("422 data_start_before_available_start")) is False


def test_dataset_for_product_routes_vix_to_cfe() -> None:
    assert dataset_for_product("ES") == CME_DATASET
    assert dataset_for_product("VX") == CFE_DATASET
    assert dataset_for_product("VXM") == CFE_DATASET


def test_iter_year_tasks_clips_final_year_to_end_date(tmp_path: Path) -> None:
    tasks = iter_year_tasks(
        ["ES", "VX"],
        start_year=2024,
        end_year=2026,
        end_date="2026-06-10",
        output_root=tmp_path / "raw",
    )

    assert [(task.product, task.year, task.start, task.end) for task in tasks] == [
        ("ES", 2024, "2024-01-01", "2025-01-01"),
        ("ES", 2025, "2025-01-01", "2026-01-01"),
        ("ES", 2026, "2026-01-01", "2026-06-10"),
        ("VX", 2024, "2024-01-01", "2025-01-01"),
        ("VX", 2025, "2025-01-01", "2026-01-01"),
        ("VX", 2026, "2026-01-01", "2026-06-10"),
    ]
    assert [(task.product, task.dataset) for task in tasks] == [
        ("ES", CME_DATASET),
        ("ES", CME_DATASET),
        ("ES", CME_DATASET),
        ("VX", CFE_DATASET),
        ("VX", CFE_DATASET),
        ("VX", CFE_DATASET),
    ]
    assert [(task.year, task.start, task.end) for task in tasks[:3]] == [
        (2024, "2024-01-01", "2025-01-01"),
        (2025, "2025-01-01", "2026-01-01"),
        (2026, "2026-01-01", "2026-06-10"),
    ]
    assert tasks[0].symbol == "ES.v.0"
    assert tasks[0].output_path.endswith("ES/2024.parquet")


def test_iter_year_tasks_clips_glbx_to_available_start(tmp_path: Path) -> None:
    tasks = iter_year_tasks(
        ["ES"],
        start_year=2010,
        end_year=2010,
        end_date="2011-01-01",
        output_root=tmp_path / "raw",
    )

    assert len(tasks) == 1
    assert tasks[0].start == "2010-06-06"
    assert tasks[0].end == "2011-01-01"


def test_first_pending_download_skips_existing_files(tmp_path: Path) -> None:
    existing = tmp_path / "raw" / "ES" / "2024.parquet"
    missing = tmp_path / "raw" / "ES" / "2025.parquet"
    existing.parent.mkdir(parents=True)
    existing.write_text("placeholder", encoding="utf-8")
    tasks = [
        DownloadTask(
            CME_DATASET,
            "ES",
            2024,
            "2024-01-01",
            "2025-01-01",
            "ES.v.0",
            existing.as_posix(),
        ),
        DownloadTask(
            CME_DATASET,
            "ES",
            2025,
            "2025-01-01",
            "2026-01-01",
            "ES.v.0",
            missing.as_posix(),
        ),
    ]

    assert first_pending_download(tasks, overwrite=False) == tasks[1]
    assert first_pending_download(tasks, overwrite=True) == tasks[0]


def test_preflight_auth_fails_fast_on_auth_error(tmp_path: Path) -> None:
    task = DownloadTask(
        CME_DATASET,
        "ES",
        2024,
        "2024-01-01",
        "2025-01-01",
        "ES.v.0",
        (tmp_path / "raw" / "ES" / "2024.parquet").as_posix(),
    )

    with pytest.raises(SystemExit, match="Databento rejected DATABENTO_API_KEY"):
        preflight_auth(AuthFailingPreflightClient(), [task], overwrite=False)


def test_store_to_required_dataframe_resets_datetime_index_to_ts_event() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10],
            "rtype": [33],
            "publisher_id": [1],
            "instrument_id": [100],
            "symbol": ["ESH4"],
        },
        index=pd.DatetimeIndex(["2024-01-02T15:00:00Z"], name="ts_event"),
    )

    out = store_to_required_dataframe(FakeStore(df))

    assert out.columns.tolist() == [
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
    assert out.loc[0, "instrument_id"] == 100
    assert out.loc[0, "data_quality_status"] == "available"
    assert out.loc[0, "data_quality_degraded"] == False


def test_store_to_required_dataframe_marks_degraded_dates() -> None:
    df = pd.DataFrame(
        {
            "ts_event": [
                pd.Timestamp("2024-01-02T15:00:00Z"),
                pd.Timestamp("2024-01-03T15:00:00Z"),
            ],
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [10, 10],
            "rtype": [33, 33],
            "publisher_id": [1, 1],
            "instrument_id": [100, 100],
            "symbol": ["ESH4", "ESH4"],
        }
    )

    out = store_to_required_dataframe(
        FakeStore(df),
        {"2024-01-03": "degraded"},
    )

    assert out["data_quality_degraded"].tolist() == [False, True]


def test_store_to_required_dataframe_fails_missing_metadata() -> None:
    df = pd.DataFrame(
        {
            "ts_event": ["2024-01-02T15:00:00Z"],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        store_to_required_dataframe(FakeStore(df))


def test_write_store_parquet_writes_visible_ts_event(tmp_path: Path) -> None:
    path = tmp_path / "ES" / "2024.parquet"
    path.parent.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "ts_event": [pd.Timestamp("2024-01-02T15:00:00Z")],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10],
            "rtype": [33],
            "publisher_id": [1],
            "instrument_id": [100],
            "symbol": ["ESH4"],
        }
    )

    write_store_parquet(FakeStore(df), path)
    check = validate_download(path)

    assert check["timestamp_ok"] is True
    assert check["valid"] is True
    assert check["missing_columns"] == []
    assert check["instrument_id_nonnull"] == 1
    assert check["degraded_bar_count"] == 0


def test_write_store_parquet_removes_tmp_file_after_atomic_replace(tmp_path: Path) -> None:
    path = tmp_path / "ES" / "2024.parquet"
    df = pd.DataFrame(
        {
            "ts_event": [pd.Timestamp("2024-01-02T15:00:00Z")],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10],
            "rtype": [33],
            "publisher_id": [1],
            "instrument_id": [100],
            "symbol": ["ESH4"],
        }
    )

    write_store_parquet(FakeStore(df), path)

    assert path.exists()
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_validate_download_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=list(REQUIRED_TEST_COLUMNS())).to_parquet(path, index=False)

    check = validate_download(path)

    assert check["valid"] is False
    assert "empty_file" in check["errors"]


def test_validate_download_rejects_duplicate_timestamp_bad_ohlc_and_negative_volume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame(
        {
            "ts_event": [
                pd.Timestamp("2024-01-02T15:00:00Z"),
                pd.Timestamp("2024-01-02T15:00:00Z"),
            ],
            "open": [10.0, 10.0],
            "high": [9.0, 11.0],
            "low": [9.5, 9.0],
            "close": [10.0, 10.5],
            "volume": [1, -1],
            "rtype": [33, 33],
            "publisher_id": [1, 1],
            "instrument_id": [100, 100],
            "symbol": ["ESH4", "ESH4"],
            "data_quality_status": ["available", "available"],
            "data_quality_degraded": [False, False],
        }
    ).to_parquet(path, index=False)

    check = validate_download(path)

    assert check["valid"] is False
    assert "duplicate_ts_event" in check["errors"]
    assert "bad_ohlc" in check["errors"]
    assert "negative_volume" in check["errors"]


def test_validate_download_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "unsorted.parquet"
    pd.DataFrame(
        {
            "ts_event": [
                pd.Timestamp("2024-01-02T15:01:00Z"),
                pd.Timestamp("2024-01-02T15:00:00Z"),
            ],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.5, 10.5],
            "volume": [1, 1],
            "rtype": [33, 33],
            "publisher_id": [1, 1],
            "instrument_id": [100, 100],
            "symbol": ["ESH4", "ESH4"],
            "data_quality_status": ["available", "available"],
            "data_quality_degraded": [False, False],
        }
    ).to_parquet(path, index=False)

    check = validate_download(path)

    assert check["valid"] is False
    assert "non_monotonic_ts_event" in check["errors"]


def test_execute_download_validates_existing_files_as_ok(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "ES" / "2024.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_event": [pd.Timestamp("2024-01-02T15:00:00Z")],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10],
            "rtype": [33],
            "publisher_id": [1],
            "instrument_id": [100],
            "symbol": ["ESH4"],
            "data_quality_status": ["available"],
            "data_quality_degraded": [False],
        }
    ).to_parquet(path, index=False)

    results = execute_download(
        client=FailingClient(),
        tasks=[
            DownloadTask(
                dataset=CME_DATASET,
                product="ES",
                year=2024,
                start="2024-01-01",
                end="2025-01-01",
                symbol="ES.v.0",
                output_path=path.as_posix(),
            )
        ],
        overwrite=False,
    )

    assert results[0]["status"] == "ok_existing"
    assert results[0]["validation"]["valid"] is True


def test_execute_download_stops_on_auth_failure(tmp_path: Path) -> None:
    tasks = [
        DownloadTask(
            dataset=CME_DATASET,
            product="ES",
            year=2024,
            start="2024-01-01",
            end="2025-01-01",
            symbol="ES.v.0",
            output_path=(tmp_path / "raw" / "ES" / "2024.parquet").as_posix(),
        ),
        DownloadTask(
            dataset=CME_DATASET,
            product="ES",
            year=2025,
            start="2025-01-01",
            end="2026-01-01",
            symbol="ES.v.0",
            output_path=(tmp_path / "raw" / "ES" / "2025.parquet").as_posix(),
        ),
    ]

    results = execute_download(FailingClient(), tasks, overwrite=False)

    assert len(results) == 1
    assert results[0]["status"] == "download_error"


def test_estimate_cost_stops_on_auth_failure(tmp_path: Path) -> None:
    tasks = [
        DownloadTask(
            dataset=CME_DATASET,
            product="ES",
            year=2024,
            start="2024-01-01",
            end="2025-01-01",
            symbol="ES.v.0",
            output_path=(tmp_path / "raw" / "ES" / "2024.parquet").as_posix(),
        ),
        DownloadTask(
            dataset=CME_DATASET,
            product="ES",
            year=2025,
            start="2025-01-01",
            end="2026-01-01",
            symbol="ES.v.0",
            output_path=(tmp_path / "raw" / "ES" / "2025.parquet").as_posix(),
        ),
    ]

    results = estimate_cost(AuthFailingEstimateClient(), tasks)

    assert len(results) == 1
    assert results[0]["status"] == "estimate_error"


def REQUIRED_TEST_COLUMNS() -> list[str]:
    return [
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
