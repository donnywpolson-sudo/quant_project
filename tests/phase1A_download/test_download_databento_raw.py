from __future__ import annotations

import json
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.phase1A_download.download_databento_raw import (
    CME_DATASET,
    CURRENT_20,
    DbnArchiveEntry,
    EXTENDED_CME,
    STYPE_IN,
    STYPE_OUT,
    DownloadTask,
    add_result_provenance,
    build_raw_ingest_manifest,
    build_arg_parser,
    build_dbn_download_manifest,
    batch_split_duration_for_chunk,
    build_raw_file_manifest,
    condition_is_degraded,
    convert_dbn_archive_to_raw,
    convert_dbn_files_to_parquet,
    dataset_for_product,
    dbn_chunk_manifest_rows,
    dbn_parquet_path,
    dry_run_plan_path,
    effective_output_root,
    effective_raw_format,
    execute_download,
    execute_batch_downloads,
    estimate_cost,
    fetch_conditions_for_archive_entries,
    finalize_plan_provenance,
    first_pending_download,
    iter_range_tasks,
    is_fatal_error,
    iter_month_ranges,
    iter_year_tasks,
    main,
    load_databento_api_key_from_file,
    normalize_api_key,
    output_role_for_run,
    pipeline_raw_ready_for_run,
    parse_symbols,
    preflight_auth,
    resolve_databento_api_key,
    symbol_for_product,
    store_to_required_dataframe,
    validate_download,
    validate_raw_file_manifest,
    write_json,
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


class SplitRetryTimeseries:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_range(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if kwargs["start"] == "2014-02-01" and kwargs["end"] == "2014-03-01":
            raise RuntimeError("Error streaming response: Response ended prematurely")
        start = str(kwargs["start"])
        df = pd.DataFrame(
            {
                "ts_event": [pd.Timestamp(f"{start}T15:00:00Z")],
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [10],
                "rtype": [33],
                "publisher_id": [1],
                "instrument_id": [100],
                "symbol": ["6BM4"],
            }
        )
        return FakeStore(df)


class SplitRetryClient:
    def __init__(self) -> None:
        self.metadata = FailingMetadata()
        self.timeseries = SplitRetryTimeseries()


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


class FakeBatch:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []
        self.downloads: list[dict[str, object]] = []

    def submit_job(self, **kwargs: object) -> dict[str, object]:
        self.submissions.append(kwargs)
        return {"id": "job-test", "state": "queued"}

    def list_jobs(self, **kwargs: object) -> list[dict[str, object]]:
        return [{"id": "job-test", "state": "done"}]

    def download(self, **kwargs: object) -> list[Path]:
        self.downloads.append(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "job-test.dbn.zst"
        path.write_bytes(b"dbn-zstd-placeholder")
        return [path]


class EmptyBatch(FakeBatch):
    def download(self, **kwargs: object) -> list[Path]:
        self.downloads.append(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        return []


class NonDbnBatch(FakeBatch):
    def download(self, **kwargs: object) -> list[Path]:
        self.downloads.append(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "job-test.txt"
        path.write_text("not dbn", encoding="utf-8")
        return [path]


class FakeBatchClient:
    def __init__(self) -> None:
        self.batch = FakeBatch()
        self.metadata = FailingMetadata()
        self.timeseries = FailingTimeseries()


class DegradedMetadata(FailingMetadata):
    def get_dataset_condition(self, **kwargs: object) -> list[dict[str, object]]:
        return [{"date": "2024-01-03", "condition": "degraded"}]


def install_fake_databento_store(
    monkeypatch: pytest.MonkeyPatch,
    df: pd.DataFrame,
) -> None:
    class FakeDBNStore:
        @classmethod
        def from_file(cls, path: Path) -> FakeStore:
            return FakeStore(df.copy())

    monkeypatch.setitem(
        sys.modules,
        "databento",
        types.SimpleNamespace(DBNStore=FakeDBNStore),
    )


def _write_raw_manifest(path: Path, *, schema: str, market: str = "ES", year: int = 2024) -> None:
    task = DownloadTask(
        dataset=CME_DATASET,
        product=market,
        year=year,
        start=f"{year}-01-01",
        end=f"{year + 1}-01-01",
        symbol=f"{market}.v.0" if schema == "ohlcv-1m" else f"{market}.FUT",
        output_path=path.as_posix(),
        schema=schema,
        stype_in="continuous" if schema == "ohlcv-1m" else "parent",
        stype_out="instrument_id",
        chunk="year",
        raw_format="dbn-zstd",
    )
    write_json(
        path.with_name(f"{path.name}.manifest.json"),
        build_raw_file_manifest(task, path, job_id="job-test", request_status="ok"),
    )


def test_parse_symbols_current_and_extended() -> None:
    assert parse_symbols(None, "current20") == CURRENT_20
    assert "ES" in parse_symbols(None, "extended_cme")
    assert len(EXTENDED_CME) > len(CURRENT_20)


def test_parse_symbols_custom_normalizes_and_sorts() -> None:
    assert parse_symbols(" es,CL, es ", "custom") == ["CL", "ES"]


def test_default_roots_match_public_raw_ingest_contract() -> None:
    args = build_arg_parser().parse_args([])
    assert args.mode == "download-dbn"
    assert args.chunk == "year"
    assert args.workers == 4
    assert args.end_date == (date.today() - timedelta(days=1)).isoformat()
    assert args.dbn_root == "data/dbn/ohlcv_1m"
    assert args.raw_root == "data/raw"
    assert args.reports_root == "reports/raw_ingest"
    assert effective_output_root(args) == Path("data/dbn/ohlcv_1m")


def test_default_batch_plan_is_archive_only_market_year_dbn() -> None:
    args = build_arg_parser().parse_args([])
    raw_format = effective_raw_format(args)
    output_root = effective_output_root(args)

    assert args.universe == "extended_cme"
    assert args.mode == "download-dbn"
    assert raw_format == "dbn-zstd"
    assert output_role_for_run(args.mode, raw_format, output_root) == "dbn_archive"
    assert pipeline_raw_ready_for_run(args.mode, raw_format, output_root) is False


def test_public_raw_ingest_modes_parse() -> None:
    parser = build_arg_parser()
    assert parser.parse_args(["--mode", "download-dbn"]).mode == "download-dbn"
    assert parser.parse_args(["--mode", "convert-parquet"]).mode == "convert-parquet"
    args = parser.parse_args(["--mode", "all"])
    assert effective_raw_format(args) == "dbn-zstd"
    assert pipeline_raw_ready_for_run(args.mode, effective_raw_format(args), effective_output_root(args)) is True


def test_phase1b_entry_defaults_to_convert_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.phase1B_convert import convert_databento_raw

    captured: dict[str, list[str]] = {}

    def fake_main() -> int:
        captured["argv"] = sys.argv.copy()
        return 0

    monkeypatch.setattr(convert_databento_raw, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["convert_databento_raw.py", "--dbn-root", "data/raw"])

    assert convert_databento_raw.phase1b_main() == 0
    assert captured["argv"][1:3] == ["--mode", "convert-parquet"]


def test_new_speedup_args_parse_without_breaking_existing_defaults() -> None:
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "GLBX.MDP3",
            "--schema",
            "ohlcv-1m",
            "--markets",
            "ES,NQ",
            "--start",
            "2023-01-01",
            "--end",
            "2023-03-01",
            "--chunk",
            "month",
            "--mode",
            "batch",
            "--workers",
            "4",
            "--raw-format",
            "dbn-zstd",
            "--resume",
        ]
    )

    assert args.symbols == "ES,NQ"
    assert args.dataset == "GLBX.MDP3"
    assert args.chunk == "month"
    assert args.mode == "batch"
    assert args.workers == 4
    assert args.raw_format == "dbn-zstd"
    assert args.resume is True


def test_continuous_requests_use_supported_output_symbology() -> None:
    assert STYPE_IN == "continuous"
    assert STYPE_OUT == "instrument_id"


def test_symbol_for_product_preserves_continuous_default_and_supports_parent() -> None:
    assert symbol_for_product("ES", "continuous") == "ES.v.0"
    assert symbol_for_product("ES", "parent") == "ES.FUT"
    assert symbol_for_product("ESM4", "raw_symbol") == "ESM4"


def test_normalize_api_key_strips_wrapping_noise() -> None:
    assert normalize_api_key(None) == ""
    assert normalize_api_key("  db-test  ") == "db-test"
    assert normalize_api_key('"db-test"') == "db-test"
    assert normalize_api_key("'db-test'") == "db-test"


def test_load_databento_api_key_from_project_databento_env(tmp_path: Path) -> None:
    key_file = tmp_path / "databento.env"
    key_file.write_text(
        "# local Databento key\nDATABENTO_API_KEY='db-file-test'\n",
        encoding="utf-8",
    )

    assert load_databento_api_key_from_file(key_file) == "db-file-test"


def test_load_databento_api_key_accepts_raw_key_in_project_databento_env(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "databento.env"
    key_file.write_text("  db-raw-test  \n", encoding="utf-8")

    assert load_databento_api_key_from_file(key_file) == "db-raw-test"


def test_resolve_databento_api_key_uses_project_databento_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "databento.env"
    key_file.write_text("DATABENTO_API_KEY=db-file-test\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.phase1A_download.download_databento_raw.API_KEY_FILE",
        key_file,
    )
    monkeypatch.setattr(
        "scripts.phase1A_download.download_databento_raw.API_KEY_FILES",
        [key_file],
    )

    assert resolve_databento_api_key() == "db-file-test"


def test_resolve_databento_api_key_prefers_secrets_databento_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_key_file = tmp_path / "databento.env"
    secrets_key_file = tmp_path / "secrets" / "databento.env"
    root_key_file.write_text("DATABENTO_API_KEY=db-root-test\n", encoding="utf-8")
    secrets_key_file.parent.mkdir(parents=True)
    secrets_key_file.write_text("DATABENTO_API_KEY=db-secrets-test\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.phase1A_download.download_databento_raw.API_KEY_FILES",
        [secrets_key_file, root_key_file],
    )

    assert resolve_databento_api_key() == "db-secrets-test"


def test_condition_is_degraded_classifies_quality_status() -> None:
    assert condition_is_degraded("available") is False
    assert condition_is_degraded("degraded") is True
    assert condition_is_degraded("pending") is True
    assert condition_is_degraded("missing") is True
    assert condition_is_degraded("partial") is True


def test_is_fatal_error_detects_auth_failure() -> None:
    assert is_fatal_error(RuntimeError("401 auth_authentication_failed")) is True
    assert is_fatal_error(RuntimeError("422 data_start_before_available_start")) is False


def test_dataset_for_product_uses_glbx_dataset() -> None:
    assert dataset_for_product("ES") == CME_DATASET


def test_iter_year_tasks_clips_final_year_to_end_date(tmp_path: Path) -> None:
    tasks = iter_year_tasks(
        ["ES"],
        start_year=2024,
        end_year=2026,
        end_date="2026-06-10",
        output_root=tmp_path / "raw",
    )

    assert [(task.product, task.year, task.start, task.end) for task in tasks] == [
        ("ES", 2024, "2024-01-01", "2025-01-01"),
        ("ES", 2025, "2025-01-01", "2026-01-01"),
        ("ES", 2026, "2026-01-01", "2026-06-10"),
    ]
    assert [(task.product, task.dataset) for task in tasks] == [
        ("ES", CME_DATASET),
        ("ES", CME_DATASET),
        ("ES", CME_DATASET),
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


def test_iter_year_tasks_clips_products_to_product_available_start(tmp_path: Path) -> None:
    tasks = iter_year_tasks(
        ["RTY", "SR3"],
        start_year=2010,
        end_year=2018,
        end_date="2019-01-01",
        output_root=tmp_path / "raw",
    )

    assert [(task.product, task.year, task.start, task.end) for task in tasks] == [
        ("RTY", 2017, "2017-06-05", "2018-01-01"),
        ("RTY", 2018, "2018-01-01", "2019-01-01"),
        ("SR3", 2018, "2018-04-23", "2019-01-01"),
    ]


def test_iter_month_ranges_uses_calendar_months_and_clips_edges() -> None:
    assert iter_month_ranges("2024-01-15", "2024-04-10") == [
        ("2024-01-15", "2024-02-01"),
        ("2024-02-01", "2024-03-01"),
        ("2024-03-01", "2024-04-01"),
        ("2024-04-01", "2024-04-10"),
    ]


def test_iter_range_tasks_builds_month_stream_jobs_without_daily_requests(
    tmp_path: Path,
) -> None:
    tasks = iter_range_tasks(
        ["ES"],
        start="2024-01-15",
        end="2024-04-10",
        output_root=tmp_path / "raw",
        chunk="month",
        schema="ohlcv-1m",
        stype_in="continuous",
        stype_out="instrument_id",
    )

    assert [(task.start, task.end) for task in tasks] == [
        ("2024-01-15", "2024-02-01"),
        ("2024-02-01", "2024-03-01"),
        ("2024-03-01", "2024-04-01"),
        ("2024-04-01", "2024-04-10"),
    ]
    assert [Path(task.output_path).name for task in tasks] == [
        "2024-01.parquet",
        "2024-02.parquet",
        "2024-03.parquet",
        "2024-04.parquet",
    ]


def test_iter_range_tasks_builds_market_year_dbn_files(tmp_path: Path) -> None:
    tasks = iter_range_tasks(
        ["ES", "NQ"],
        start="2024-01-01",
        end="2025-01-01",
        output_root=tmp_path / "raw",
        chunk="year",
        mode="download-dbn",
        raw_format="dbn-zstd",
        dataset="GLBX.MDP3",
        stype_in="continuous",
    )

    assert len(tasks) == 2
    assert [(task.product, task.start, task.end, task.chunk) for task in tasks] == [
        ("ES", "2024-01-01", "2025-01-01", "year"),
        ("NQ", "2024-01-01", "2025-01-01", "year"),
    ]
    assert Path(tasks[0].output_path).parts[-3:] == (
        "ES",
        "2024",
        "2024-01-01_2025-01-01.dbn.zst",
    )
    assert batch_split_duration_for_chunk("year") == "year"


def test_iter_range_tasks_clips_products_to_product_available_start(tmp_path: Path) -> None:
    tasks = iter_range_tasks(
        ["RTY", "SR3"],
        start="2010-01-01",
        end="2019-01-01",
        output_root=tmp_path / "raw",
        chunk="year",
        mode="download-dbn",
        raw_format="dbn-zstd",
        dataset="GLBX.MDP3",
        stype_in="continuous",
    )

    assert [(task.product, task.year, task.start, task.end) for task in tasks] == [
        ("RTY", 2017, "2017-06-05", "2018-01-01"),
        ("RTY", 2018, "2018-01-01", "2019-01-01"),
        ("SR3", 2018, "2018-04-23", "2019-01-01"),
    ]


def test_iter_range_tasks_builds_batch_dbn_zstd_jobs_with_parent_symbols(
    tmp_path: Path,
) -> None:
    tasks = iter_range_tasks(
        ["ES", "NQ"],
        start="2024-01-01",
        end="2024-03-01",
        output_root=tmp_path / "dbn",
        chunk="month",
        mode="download-dbn",
        raw_format="dbn-zstd",
        dataset="GLBX.MDP3",
        stype_in="parent",
    )

    assert len(tasks) == 4
    assert {task.dataset for task in tasks} == {"GLBX.MDP3"}
    assert tasks[0].symbol == "ES.FUT"
    assert tasks[0].raw_format == "dbn-zstd"
    assert Path(tasks[0].output_path).parts[-3:] == (
        "ES",
        "2024",
        "2024-01-01_2024-02-01.dbn.zst",
    )


def test_iter_range_tasks_rejects_non_glbx_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset 'NOT.GLBX' is not allowed"):
        iter_range_tasks(
            ["ES"],
            start="2024-01-01",
            end="2024-02-01",
            output_root=tmp_path / "raw_databento",
            chunk="month",
            mode="batch",
            raw_format="dbn-zstd",
            dataset="NOT.GLBX",
        )


def test_iter_range_tasks_rejects_products_outside_allowed_glbx_universe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="products outside the allowed GLBX.MDP3 futures universe"):
        iter_range_tasks(
            ["NOT_A_PRODUCT"],
            start="2024-01-01",
            end="2024-02-01",
            output_root=tmp_path / "raw_databento",
            chunk="month",
            mode="batch",
            raw_format="dbn-zstd",
        )


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


def test_first_pending_download_does_not_skip_empty_final_file(tmp_path: Path) -> None:
    empty = tmp_path / "raw" / "ES" / "2024.parquet"
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")
    task = DownloadTask(
        CME_DATASET,
        "ES",
        2024,
        "2024-01-01",
        "2025-01-01",
        "ES.v.0",
        empty.as_posix(),
    )

    assert first_pending_download([task], overwrite=False) == task


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

    with pytest.raises(SystemExit, match="Databento rejected preflight request for GLBX.MDP3"):
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


def test_store_to_required_dataframe_can_mark_unknown_conversion_quality() -> None:
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

    out = store_to_required_dataframe(
        FakeStore(df),
        default_quality_status="metadata_unavailable",
    )

    assert out.loc[0, "data_quality_status"] == "metadata_unavailable"
    assert out.loc[0, "data_quality_degraded"] == True


def test_convert_dbn_files_validates_existing_converted_parquet(tmp_path: Path) -> None:
    dbn_path = tmp_path / "job-test.dbn.zst"
    dbn_path.write_bytes(b"dbn")
    pd.DataFrame({"bad": [1]}).to_parquet(dbn_parquet_path(dbn_path), index=False)

    with pytest.raises(ValueError, match="converted parquet failed validation"):
        convert_dbn_files_to_parquet([dbn_path], overwrite=False)


def test_convert_existing_requires_non_available_quality_on_skipped_parquet(
    tmp_path: Path,
) -> None:
    dbn_path = tmp_path / "job-test.dbn.zst"
    dbn_path.write_bytes(b"dbn")
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
    ).to_parquet(dbn_parquet_path(dbn_path), index=False)

    with pytest.raises(ValueError, match="ambiguous data_quality_status"):
        convert_dbn_files_to_parquet(
            [dbn_path],
            overwrite=False,
            default_quality_status="metadata_unavailable",
        )


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


def test_validate_download_rejects_null_or_blank_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad_metadata.parquet"
    pd.DataFrame(
        {
            "ts_event": [
                pd.Timestamp("2024-01-02T15:00:00Z"),
                pd.Timestamp("2024-01-02T15:01:00Z"),
            ],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.5, 10.5],
            "volume": [1, 1],
            "rtype": [None, 33],
            "publisher_id": [1, None],
            "instrument_id": [100, None],
            "symbol": [None, " "],
            "data_quality_status": ["available", "available"],
            "data_quality_degraded": [False, False],
        }
    ).to_parquet(path, index=False)

    check = validate_download(path)

    assert check["valid"] is False
    assert "null_metadata:rtype" in check["errors"]
    assert "null_metadata:publisher_id" in check["errors"]
    assert "null_metadata:instrument_id" in check["errors"]
    assert "null_metadata:symbol" in check["errors"]
    assert "blank_symbol" in check["errors"]


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


def test_execute_download_downloads_months_and_splits_retryable_month_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw" / "6B" / "2014.parquet"
    client = SplitRetryClient()
    task = DownloadTask(
        dataset=CME_DATASET,
        product="6B",
        year=2014,
        start="2014-01-01",
        end="2015-01-01",
        symbol="6B.v.0",
        output_path=path.as_posix(),
    )

    results = execute_download(client, [task], overwrite=False)

    assert results[0]["status"] == "ok"
    assert results[0]["validation"]["rows"] == 13
    call_ranges = [(call["start"], call["end"]) for call in client.timeseries.calls]
    assert call_ranges[:4] == [
        ("2014-01-01", "2014-02-01"),
        ("2014-02-01", "2014-03-01"),
        ("2014-02-01", "2014-02-15"),
        ("2014-02-15", "2014-03-01"),
    ]
    assert call_ranges[-1] == ("2014-12-01", "2015-01-01")
    df = pd.read_parquet(path)
    assert df["ts_event"].iloc[0] == pd.Timestamp("2014-01-01T15:00:00Z")
    assert df["ts_event"].iloc[-1] == pd.Timestamp("2014-12-01T15:00:00Z")


def test_execute_batch_download_writes_temp_dir_then_final_dbn_file(tmp_path: Path) -> None:
    client = FakeBatchClient()
    final_file = tmp_path / "raw" / "ES" / "2024.dbn.zst"
    task = DownloadTask(
        dataset=CME_DATASET,
        product="ES",
        year=2024,
        start="2024-01-01",
        end="2024-02-01",
        symbol="ES.v.0",
        output_path=final_file.as_posix(),
        chunk="month",
        raw_format="dbn-zstd",
    )

    results = execute_batch_downloads(
        [task],
        overwrite=False,
        workers=1,
        client_factory=lambda: client,
        convert_parquet=False,
        batch_wait_timeout_seconds=1.0,
        batch_poll_seconds=0.01,
    )

    assert results[0]["status"] == "ok"
    assert final_file.read_bytes() == b"dbn-zstd-placeholder"
    assert not list(final_file.parent.glob("*.tmp-*"))
    assert client.batch.submissions[0]["encoding"] == "dbn"
    assert client.batch.submissions[0]["compression"] == "zstd"
    assert client.batch.submissions[0]["delivery"] == "download"
    assert client.batch.submissions[0]["split_duration"] == "month"
    manifest_path = final_file.with_name(f"{final_file.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["vendor"] == "databento"
    assert manifest["dataset"] == "GLBX.MDP3"
    assert manifest["schema"] == "ohlcv-1m"
    assert manifest["encoding"] == "dbn"
    assert manifest["compression"] == "zstd"
    assert manifest["file_size_bytes"] == len(b"dbn-zstd-placeholder")
    assert not validate_raw_file_manifest(
        final_file,
        expected_schema="ohlcv-1m",
        expected_market="ES",
        expected_year=2024,
    )


def test_raw_file_manifest_validation_fails_on_schema_mismatch(tmp_path: Path) -> None:
    dbn_path = tmp_path / "raw" / "ES" / "2024.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")
    _write_raw_manifest(dbn_path, schema="definition")

    failures = validate_raw_file_manifest(
        dbn_path,
        expected_schema="ohlcv-1m",
        expected_market="ES",
        expected_year=2024,
    )

    assert "manifest schema mismatch" in failures


def test_raw_file_manifest_validation_fails_on_checksum_mismatch(tmp_path: Path) -> None:
    dbn_path = tmp_path / "raw" / "ES" / "2024.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")
    _write_raw_manifest(dbn_path, schema="ohlcv-1m")
    dbn_path.write_bytes(b"changed")

    failures = validate_raw_file_manifest(
        dbn_path,
        expected_schema="ohlcv-1m",
        expected_market="ES",
        expected_year=2024,
    )

    assert "checksum mismatch" in failures


def test_existing_batch_file_with_valid_manifest_is_not_overwritten(tmp_path: Path) -> None:
    client = FakeBatchClient()
    final_file = tmp_path / "raw" / "ES" / "2024.dbn.zst"
    final_file.parent.mkdir(parents=True)
    final_file.write_bytes(b"existing-dbn")
    task = DownloadTask(
        dataset=CME_DATASET,
        product="ES",
        year=2024,
        start="2024-01-01",
        end="2025-01-01",
        symbol="ES.v.0",
        output_path=final_file.as_posix(),
        chunk="year",
        raw_format="dbn-zstd",
    )
    write_json(
        final_file.with_name(f"{final_file.name}.manifest.json"),
        build_raw_file_manifest(task, final_file, job_id="job-existing", request_status="ok"),
    )

    results = execute_batch_downloads(
        [task],
        overwrite=False,
        workers=1,
        client_factory=lambda: client,
        convert_parquet=False,
        max_retries=0,
        batch_wait_timeout_seconds=1.0,
        batch_poll_seconds=0.01,
    )

    assert results[0]["status"] == "ok_existing"
    assert final_file.read_bytes() == b"existing-dbn"
    assert not client.batch.submissions


def test_existing_batch_directory_without_dbn_is_not_ok_existing(tmp_path: Path) -> None:
    client = FakeBatchClient()
    final_file = tmp_path / "raw" / "ES" / "2024.dbn.zst"
    final_file.mkdir(parents=True)
    (final_file / "note.txt").write_text("not dbn", encoding="utf-8")
    task = DownloadTask(
        dataset=CME_DATASET,
        product="ES",
        year=2024,
        start="2024-01-01",
        end="2024-02-01",
        symbol="ES.v.0",
        output_path=final_file.as_posix(),
        chunk="month",
        raw_format="dbn-zstd",
    )

    results = execute_batch_downloads(
        [task],
        overwrite=False,
        workers=1,
        client_factory=lambda: client,
        convert_parquet=False,
        max_retries=0,
        batch_wait_timeout_seconds=1.0,
        batch_poll_seconds=0.01,
    )

    assert results[0]["status"] == "ok"
    assert client.batch.submissions
    assert final_file.is_file()
    assert final_file.read_bytes() == b"dbn-zstd-placeholder"


def test_batch_download_without_non_empty_dbn_files_is_not_ok(tmp_path: Path) -> None:
    client = FakeBatchClient()
    client.batch = NonDbnBatch()
    task = DownloadTask(
        dataset=CME_DATASET,
        product="ES",
        year=2024,
        start="2024-01-01",
        end="2024-02-01",
        symbol="ES.v.0",
        output_path=(tmp_path / "raw" / "ES" / "2024.dbn.zst").as_posix(),
        chunk="month",
        raw_format="dbn-zstd",
    )

    results = execute_batch_downloads(
        [task],
        overwrite=False,
        workers=1,
        client_factory=lambda: client,
        convert_parquet=False,
        max_retries=0,
        batch_wait_timeout_seconds=1.0,
        batch_poll_seconds=0.01,
    )

    assert results[0]["status"] == "download_error"
    assert "no non-empty DBN files" in str(results[0]["error"])


def test_empty_batch_download_is_not_ok(tmp_path: Path) -> None:
    client = FakeBatchClient()
    client.batch = EmptyBatch()
    task = DownloadTask(
        dataset=CME_DATASET,
        product="ES",
        year=2024,
        start="2024-01-01",
        end="2024-02-01",
        symbol="ES.v.0",
        output_path=(tmp_path / "raw" / "ES" / "2024.dbn.zst").as_posix(),
        chunk="month",
        raw_format="dbn-zstd",
    )

    results = execute_batch_downloads(
        [task],
        overwrite=False,
        workers=1,
        client_factory=lambda: client,
        convert_parquet=False,
        max_retries=0,
        batch_wait_timeout_seconds=1.0,
        batch_poll_seconds=0.01,
    )

    assert results[0]["status"] == "download_error"
    assert "no non-empty DBN files" in str(results[0]["error"])


def test_batch_convert_parquet_preserves_degraded_dataset_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted_df = pd.DataFrame(
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
    install_fake_databento_store(monkeypatch, converted_df)
    client = FakeBatchClient()
    client.metadata = DegradedMetadata()
    task = DownloadTask(
        dataset=CME_DATASET,
        product="ES",
        year=2024,
        start="2024-01-01",
        end="2024-02-01",
        symbol="ES.v.0",
        output_path=(tmp_path / "raw" / "ES" / "2024.dbn.zst").as_posix(),
        chunk="month",
        raw_format="dbn-zstd",
    )

    results = execute_batch_downloads(
        [task],
        overwrite=False,
        workers=1,
        client_factory=lambda: client,
        convert_parquet=True,
        max_retries=0,
        batch_wait_timeout_seconds=1.0,
        batch_poll_seconds=0.01,
    )

    assert results[0]["status"] == "ok"
    final_parquet = Path(task.output_path).with_suffix(".zst.parquet")
    out = pd.read_parquet(final_parquet)
    assert out["data_quality_status"].tolist() == ["available", "degraded"]
    assert out["data_quality_degraded"].tolist() == [False, True]
    assert results[0]["dataset_condition"]["degraded_date_count"] == 1


def test_convert_dbn_archive_fails_without_quality_metadata(tmp_path: Path) -> None:
    dbn_root = tmp_path / "raw"
    raw_root = dbn_root
    dbn_path = dbn_root / "ES" / "2024.dbn.zst"
    definition_path = dbn_root / "definition" / "ES" / "2024.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    definition_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")
    definition_path.write_bytes(b"definition-placeholder")
    _write_raw_manifest(dbn_path, schema="ohlcv-1m")
    _write_raw_manifest(definition_path, schema="definition")

    results = convert_dbn_archive_to_raw(dbn_root, raw_root)

    output_path = raw_root / "ES" / "2024.parquet"
    assert results[0]["status"] == "convert_error"
    assert "missing dataset-condition metadata" in str(results[0]["error"])
    assert results[0]["data_quality_source"] == "metadata_unavailable"
    assert results[0]["vendor_quality_available"] is False
    assert not output_path.exists()


def test_fetch_conditions_for_archive_entries_uses_manifest_dates(tmp_path: Path) -> None:
    dbn_path = tmp_path / "raw" / "ES" / "2024.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")
    _write_raw_manifest(dbn_path, schema="ohlcv-1m")
    client = FakeBatchClient()
    client.metadata = DegradedMetadata()

    conditions = fetch_conditions_for_archive_entries(
        client,
        [DbnArchiveEntry(path=dbn_path, product="ES", year=2024)],
    )

    assert conditions == {("ES", 2024): {"2024-01-03": "degraded"}}


def test_convert_dbn_archive_fails_when_definition_file_missing(tmp_path: Path) -> None:
    dbn_root = tmp_path / "raw"
    dbn_path = dbn_root / "ES" / "2024.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")
    _write_raw_manifest(dbn_path, schema="ohlcv-1m")

    results = convert_dbn_archive_to_raw(
        dbn_root,
        dbn_root,
        condition_by_group={("ES", 2024): {"2024-01-03": "available"}},
    )

    assert results[0]["status"] == "convert_error"
    assert "missing definition file" in str(results[0]["error"])


def test_convert_dbn_archive_fails_when_definition_mapping_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted_df = pd.DataFrame(
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
            "min_price_increment": [0.25],
        }
    )
    install_fake_databento_store(monkeypatch, converted_df)
    dbn_root = tmp_path / "raw"
    dbn_path = dbn_root / "ES" / "2024.dbn.zst"
    definition_path = dbn_root / "definition" / "ES" / "2024.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    definition_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")
    definition_path.write_bytes(b"definition-placeholder")
    _write_raw_manifest(dbn_path, schema="ohlcv-1m")
    _write_raw_manifest(definition_path, schema="definition")

    results = convert_dbn_archive_to_raw(
        dbn_root,
        dbn_root,
        condition_by_group={("ES", 2024): {"2024-01-02": "available"}},
    )

    assert results[0]["status"] == "convert_error"
    assert "definition missing required fields: raw_symbol" in str(results[0]["error"])
    assert not (dbn_root / "ES" / "2024.parquet").exists()


def test_convert_dbn_archive_groups_multiple_canonical_ohlcv_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted_df = pd.DataFrame(
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
            "raw_symbol": ["ESH4", "ESH4"],
            "min_price_increment": [0.25, 0.25],
            "contract_multiplier": [50.0, 50.0],
        }
    )
    dbn_root = tmp_path / "data" / "dbn" / "ohlcv_1m"
    raw_root = tmp_path / "data" / "raw"
    dbn_paths = [
        dbn_root / "ES" / "2024" / "2024-01-01_2024-02-01.dbn.zst",
        dbn_root / "ES" / "2024" / "2024-02-01_2024-03-01.dbn.zst",
    ]
    definition_path = (
        tmp_path
        / "data"
        / "dbn"
        / "definition"
        / "ES"
        / "2024"
        / "2024-01-01_2025-01-01.dbn.zst"
    )
    for path in [*dbn_paths, definition_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dbn-zstd-placeholder")
    for path in dbn_paths:
        _write_raw_manifest(path, schema="ohlcv-1m")
    _write_raw_manifest(definition_path, schema="definition")

    class FakeDBNStore:
        @classmethod
        def from_file(cls, path: Path) -> FakeStore:
            df = converted_df.copy()
            if path == dbn_paths[1]:
                df["ts_event"] = df["ts_event"] + pd.Timedelta(days=31)
            return FakeStore(df)

    monkeypatch.setitem(
        sys.modules,
        "databento",
        types.SimpleNamespace(DBNStore=FakeDBNStore),
    )

    results = convert_dbn_archive_to_raw(
        dbn_root,
        raw_root,
        condition_by_group={("ES", 2024): {"2024-01-03": "degraded"}},
    )

    output_path = raw_root / "ES" / "2024.parquet"
    assert results[0]["status"] == "ok"
    assert sorted(results[0]["input_paths"]) == sorted(path.as_posix() for path in dbn_paths)
    assert results[0]["definition_paths"] == [definition_path.as_posix()]
    assert output_path.exists()
    assert len(pd.read_parquet(output_path)) == 4
    assert not (raw_root / "definition" / "ES" / "2024.parquet").exists()


def test_convert_dbn_archive_discovers_legacy_flat_dbn_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted_df = pd.DataFrame(
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
            "raw_symbol": ["ESH4"],
            "min_price_increment": [0.25],
            "contract_multiplier": [50.0],
        }
    )
    install_fake_databento_store(monkeypatch, converted_df)
    dbn_root = tmp_path / "data" / "dbn" / "ohlcv_1m"
    raw_root = tmp_path / "data" / "raw"
    legacy_dbn = raw_root / "ES" / "2024.dbn.zst"
    legacy_definition = raw_root / "definition" / "ES" / "2024.dbn.zst"
    legacy_dbn.parent.mkdir(parents=True)
    legacy_definition.parent.mkdir(parents=True)
    legacy_dbn.write_bytes(b"dbn-zstd-placeholder")
    legacy_definition.write_bytes(b"definition-placeholder")
    _write_raw_manifest(legacy_dbn, schema="ohlcv-1m")
    _write_raw_manifest(legacy_definition, schema="definition")

    results = convert_dbn_archive_to_raw(
        dbn_root,
        raw_root,
        condition_by_group={("ES", 2024): {"2024-01-02": "available"}},
    )

    assert results[0]["status"] == "ok"
    assert results[0]["input_paths"] == [legacy_dbn.as_posix()]
    assert results[0]["definition_paths"] == [legacy_definition.as_posix()]
    assert (raw_root / "ES" / "2024.parquet").exists()


def test_convert_dbn_archive_rejects_nested_chunk_layout(tmp_path: Path) -> None:
    dbn_root = tmp_path / "raw"
    dbn_path = dbn_root / "ES" / "2024" / "2024-01.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")

    with pytest.raises(ValueError, match=r"data/raw/\{market\}/\{year\}.dbn.zst"):
        convert_dbn_archive_to_raw(dbn_root, dbn_root)


def test_convert_dbn_archive_writes_canonical_raw_parquet_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted_df = pd.DataFrame(
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
            "raw_symbol": ["ESH4", "ESH4"],
            "min_price_increment": [0.25, 0.25],
            "contract_multiplier": [50.0, 50.0],
            "expiration": ["2024-03-15", "2024-03-15"],
            "maturity_year": [2024, 2024],
            "maturity_month": [3, 3],
        }
    )
    install_fake_databento_store(monkeypatch, converted_df)
    dbn_root = tmp_path / "raw"
    raw_root = dbn_root
    dbn_path = dbn_root / "ES" / "2024.dbn.zst"
    definition_path = dbn_root / "definition" / "ES" / "2024.dbn.zst"
    dbn_path.parent.mkdir(parents=True)
    definition_path.parent.mkdir(parents=True)
    dbn_path.write_bytes(b"dbn-zstd-placeholder")
    definition_path.write_bytes(b"definition-placeholder")
    _write_raw_manifest(dbn_path, schema="ohlcv-1m")
    _write_raw_manifest(definition_path, schema="definition")

    results = convert_dbn_archive_to_raw(
        dbn_root,
        raw_root,
        condition_by_group={("ES", 2024): {"2024-01-03": "degraded"}},
    )

    output_path = raw_root / "ES" / "2024.parquet"
    assert results[0]["status"] == "ok"
    assert output_path.exists()
    out = pd.read_parquet(output_path)
    assert out["data_quality_status"].tolist() == ["available", "degraded"]
    assert out["data_quality_degraded"].tolist() == [False, True]
    assert results[0]["output_path"] == output_path.as_posix()
    assert results[0]["input_hashes"][dbn_path.as_posix()]
    assert results[0]["input_hashes"][definition_path.as_posix()]
    assert results[0]["output_hash"]
    assert results[0]["schema"] == "ohlcv-1m"
    assert results[0]["price_scale_policy"]
    assert results[0]["data_quality_source"] == "databento_metadata.get_dataset_condition"
    assert results[0]["vendor_quality_available"] is True
    assert results[0]["definition_point_in_time_enforced"] is False
    assert out["raw_symbol"].tolist() == ["ESH4", "ESH4"]
    assert out["tick_size"].tolist() == [0.25, 0.25]

    manifest = build_raw_ingest_manifest(
        results,
        mode="convert-parquet",
        dbn_root=dbn_root,
        raw_root=raw_root,
    )
    assert manifest["schema"] == "ohlcv-1m"
    assert manifest["required_schema_columns"]
    assert manifest["data_quality_fields"] == ["data_quality_status", "data_quality_degraded"]
    assert manifest["data_quality_sources"] == ["databento_metadata.get_dataset_condition"]
    assert manifest["vendor_quality_available"] is True
    assert manifest["decoded_symbols"] == ["ESH4"]
    assert manifest["price_scale_policy"]
    assert manifest["input_hashes"][dbn_path.as_posix()]
    assert manifest["output_hashes"][output_path.as_posix()]


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


def test_plan_and_results_share_run_id_and_plan_hash() -> None:
    plan = finalize_plan_provenance(
        {
            "mode": "stream",
            "chunk": "year",
            "raw_format": "parquet",
            "output_role": "pipeline_raw_parquet",
            "pipeline_raw_ready": True,
            "tasks": [],
        },
        run_kind="download",
    )

    results = add_result_provenance([{"status": "ok"}], plan)

    assert plan["run_id"]
    assert plan["plan_hash"]
    assert results[0]["run_id"] == plan["run_id"]
    assert results[0]["plan_hash"] == plan["plan_hash"]
    assert results[0]["output_role"] == "pipeline_raw_parquet"
    assert results[0]["pipeline_raw_ready"] is True


def test_dbn_download_manifest_and_chunk_rows_contain_expected_fields(
    tmp_path: Path,
) -> None:
    result = {
        "status": "ok",
        "schema": "ohlcv-1m",
        "product": "ES",
        "year": 2024,
        "start": "2024-01-01",
        "end": "2025-01-01",
        "chunk": "year",
        "output_path": (tmp_path / "data" / "dbn" / "ohlcv_1m" / "ES" / "2024" / "2024-01-01_2025-01-01.dbn.zst").as_posix(),
        "manifest_path": (tmp_path / "manifest.json").as_posix(),
        "bytes": 10,
        "job_id": "job-test",
        "run_id": "run-test",
        "plan_hash": "hash-test",
    }

    manifest = build_dbn_download_manifest(
        [result],
        mode="download-dbn",
        dbn_root=tmp_path / "data" / "dbn" / "ohlcv_1m",
        raw_root=tmp_path / "data" / "raw",
        run_id="run-test",
        plan_hash="hash-test",
    )
    rows = dbn_chunk_manifest_rows([result])

    assert manifest["stage"] == "raw_ingest"
    assert manifest["dbn_root"].endswith("data/dbn/ohlcv_1m")
    assert manifest["raw_root"].endswith("data/raw")
    assert manifest["schemas"] == ["ohlcv-1m"]
    assert manifest["chunk_count"] == 1
    assert manifest["status_counts"] == {"ok": 1}
    assert rows[0]["output_path"].endswith("2024-01-01_2025-01-01.dbn.zst")
    assert rows[0]["manifest_path"].endswith("manifest.json")


def test_dry_run_uses_separate_plan_path_and_no_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_out = tmp_path / "reports" / "databento_download_plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_databento_raw.py",
            "--markets",
            "ES",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--dry-run",
            "--plan-out",
            plan_out.as_posix(),
        ],
    )

    assert main() == 0

    dry_plan = dry_run_plan_path(plan_out)
    assert not plan_out.exists()
    assert dry_plan.exists()
    assert not (plan_out.parent / "databento_download_results.json").exists()
    payload = json.loads(dry_plan.read_text(encoding="utf-8"))
    assert payload["run_kind"] == "dry_run"
    assert payload["run_id"]
    assert payload["plan_hash"]
    assert payload["output_role"] == "dbn_archive"
    assert payload["pipeline_raw_ready"] is False
    assert payload["schema"] == "all"
    assert payload["schemas"] == ["ohlcv-1m", "definition"]
    assert payload["task_count"] == 2
    planned = payload["tasks"]
    assert [task["schema"] for task in planned] == ["ohlcv-1m", "definition"]
    assert planned[0]["output_path"].endswith(
        "data/dbn/ohlcv_1m/ES/2024/2024-01-01_2024-01-02.dbn.zst"
    )
    assert planned[1]["output_path"].endswith(
        "data/dbn/definition/ES/2024/2024-01-01_2024-01-02.dbn.zst"
    )
    assert planned[1]["symbol"] == "ES.FUT"
    assert planned[1]["stype_in"] == "parent"


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
