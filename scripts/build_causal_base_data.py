#!/usr/bin/env python3
"""Build Phase 2 causal base parquet files from raw 1-minute futures bars."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PROFILE = "all_raw"
DISCOVERY_PROFILES = {"all_raw", "all_raw_data"}

# Discovery profiles process every top-level data/raw/{market}/{year}.parquet file.
# Static profiles are optional limited subsets for faster smoke tests.
STATIC_PROFILE_MARKETS = {
    "tier_1_CL_ES_ZN": ["CL", "ES", "ZN"],
}

STATIC_PROFILE_YEARS = {
    "tier_1_CL_ES_ZN": [2023, 2024, 2025],
}

REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
AUDIT_RAW_COLUMNS = ["rtype", "publisher_id", "instrument_id", "symbol"]

OUTPUT_COLUMNS = [
    "ts",
    "market",
    "year",
    "symbol",
    "instrument_id",
    "publisher_id",
    "rtype",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "raw_row_present",
    "is_synthetic",
    "valid_ohlcv",
    "inside_session",
    "causal_valid",
    "session_id",
    "session_date",
    "session_segment_id",
    "session_template",
    "is_session_open",
    "is_session_close",
    "minutes_since_session_open",
    "minutes_until_session_close",
    "session_progress",
    "minute_of_day",
    "day_of_week",
    "roll_boundary_flag",
    "symbol_change_flag",
    "instrument_id_change_flag",
    "bars_since_roll",
    "bars_until_roll",
    "roll_window_flag",
    "source_path",
    "source_file_hash",
    "source_row_number",
    "raw_schema_variant",
    "timestamp_source",
    "metadata_available",
    "roll_detection_available",
    "roll_detection_source",
    "roll_policy_status",
]

SESSION_TEMPLATE = "cme_globex_17_16_ct"
EXCHANGE_TZ = "America/Chicago"
DEFAULT_ROLL_WINDOW_BARS = 15
DEFAULT_MAX_SYNTHETIC_GAP_MINUTES = 120


@dataclass
class ValidationResult:
    profile: str
    market: str
    year: int
    input_path: str
    output_path: str
    source_file_hash: str | None = None
    raw_rows: int = 0
    output_rows: int = 0
    synthetic_rows: int = 0
    outside_session_rows: int = 0
    roll_boundary_rows: int = 0
    roll_window_rows: int = 0
    raw_schema_variant: str | None = None
    timestamp_source: str | None = None
    metadata_available: bool = False
    roll_detection_available: bool = False
    roll_detection_source: str = "unavailable"
    roll_policy_status: str = "unavailable_metadata"
    symbol_nonnull_count: int = 0
    instrument_id_nonnull_count: int = 0
    instrument_id_nunique: int = 0
    missing_required_raw_cols: list[str] = field(default_factory=list)
    missing_audit_cols: list[str] = field(default_factory=list)
    duplicate_timestamps: int = 0
    null_ts: int = 0
    invalid_ohlcv_rows: int = 0
    negative_volume_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict[str, object]:
        data = self.__dict__.copy()
        data["status"] = self.status
        data["warning_count"] = len(self.warnings)
        data["failure_count"] = len(self.failures)
        data["roll_boundary_count"] = self.roll_boundary_rows
        data["roll_window_count"] = self.roll_window_rows
        return data

    def to_csv_row(self) -> dict[str, object]:
        data = self.to_dict()
        data["missing_required_raw_cols"] = ";".join(self.missing_required_raw_cols)
        data["missing_audit_cols"] = ";".join(self.missing_audit_cols)
        data["warnings"] = ";".join(self.warnings)
        data["failures"] = ";".join(self.failures)
        return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_source_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def discover_raw_inputs(raw_root: Path) -> list[tuple[str, int, Path]]:
    """Discover top-level data/raw/{market}/{year}.parquet files only."""
    if not raw_root.exists():
        raise SystemExit(f"Raw root does not exist: {raw_root}")

    inputs: list[tuple[str, int, Path]] = []
    for market_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for parquet_path in sorted(market_dir.glob("*.parquet")):
            if not parquet_path.stem.isdigit():
                continue
            inputs.append((market_dir.name, int(parquet_path.stem), parquet_path))

    if not inputs:
        raise SystemExit(f"No raw year parquet files found under {raw_root}")
    return inputs


def resolve_profile_inputs(profile: str, raw_root: Path) -> list[tuple[str, int, Path]]:
    if profile in DISCOVERY_PROFILES:
        return discover_raw_inputs(raw_root)

    if profile not in STATIC_PROFILE_MARKETS:
        known = ", ".join(sorted([*STATIC_PROFILE_MARKETS, *DISCOVERY_PROFILES]))
        raise SystemExit(f"Unknown profile {profile!r}. Known profiles: {known}")

    inputs: list[tuple[str, int, Path]] = []
    for market in STATIC_PROFILE_MARKETS[profile]:
        for year in STATIC_PROFILE_YEARS[profile]:
            inputs.append((market, year, raw_root / market / f"{year}.parquet"))
    return inputs


def infer_market_year(path: Path) -> tuple[str, int]:
    try:
        return path.parent.name, int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Cannot infer market/year from {path}") from exc


def _session_metadata(ts: pd.Series) -> pd.DataFrame:
    local = ts.dt.tz_convert(EXCHANGE_TZ)
    dow = local.dt.dayofweek
    minutes = local.dt.hour * 60 + local.dt.minute

    after_open = minutes >= 17 * 60
    before_close = minutes < 16 * 60
    inside = (after_open & dow.isin([6, 0, 1, 2, 3])) | (
        before_close & dow.isin([0, 1, 2, 3, 4])
    )

    local_midnight = local.dt.normalize().dt.tz_localize(None)
    session_date = local_midnight.where(before_close, local_midnight + pd.Timedelta(days=1))
    session_date = pd.to_datetime(session_date.dt.date)

    open_naive = session_date - pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    close_naive = session_date + pd.Timedelta(hours=16)
    open_local = open_naive.dt.tz_localize(EXCHANGE_TZ)
    close_local = close_naive.dt.tz_localize(EXCHANGE_TZ)

    since_open = (local - open_local).dt.total_seconds() / 60.0
    until_close = (close_local - local).dt.total_seconds() / 60.0
    denom = since_open + until_close
    progress = np.where(denom > 0, since_open / denom, np.nan)

    session_date_str = session_date.dt.strftime("%Y-%m-%d")
    session_id = session_date_str.where(inside, pd.NA)

    metadata = pd.DataFrame(
        {
            "inside_session": inside.astype(bool),
            "session_date": session_date_str.where(inside, pd.NA),
            "session_id": session_id.where(session_id.isna(), "session_" + session_id),
            "session_template": np.where(inside, SESSION_TEMPLATE, pd.NA),
            "minutes_since_session_open": np.where(inside, since_open, np.nan),
            "minutes_until_session_close": np.where(inside, until_close, np.nan),
            "session_progress": np.where(inside, progress, np.nan),
            "minute_of_day": minutes.astype("int64"),
            "day_of_week": dow.astype("int64"),
        },
        index=ts.index,
    )
    return metadata


def _valid_ohlcv(df: pd.DataFrame) -> pd.Series:
    price_cols = ["open", "high", "low", "close"]
    prices = df[price_cols].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    return (
        prices.notna().all(axis=1)
        & volume.notna()
        & (prices["high"] >= prices["low"])
        & (prices["open"] <= prices["high"])
        & (prices["open"] >= prices["low"])
        & (prices["close"] <= prices["high"])
        & (prices["close"] >= prices["low"])
        & (volume >= 0)
    )


def _timestamp_from_raw(raw: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    if "ts_event" in raw.columns:
        return pd.to_datetime(raw["ts_event"], utc=True, errors="coerce"), "ts_event_column"
    if isinstance(raw.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(raw.index, utc=True, errors="coerce"), index=raw.index), "dataframe_index"
    return None, None


def _schema_variant(raw: pd.DataFrame, timestamp_source: str) -> str:
    metadata_cols_present = all(col in raw.columns for col in AUDIT_RAW_COLUMNS)
    if metadata_cols_present and timestamp_source == "ts_event_column":
        return "databento_full"
    if metadata_cols_present and timestamp_source == "dataframe_index":
        return "metadata_no_ts_event"
    return "ohlcv_only"


def _build_synthetic_rows(
    df: pd.DataFrame,
    market: str,
    year: int,
    source_path: str,
    source_hash: str,
    max_gap_minutes: int,
) -> pd.DataFrame:
    synthetic: list[dict[str, object]] = []
    inside = df[df["inside_session"]].sort_values("ts")

    for _, group in inside.groupby("session_id", sort=False):
        group = group.sort_values("ts")
        if len(group) < 2:
            continue
        prev_rows = group.iloc[:-1]
        next_rows = group.iloc[1:]
        gaps = (next_rows["ts"].to_numpy() - prev_rows["ts"].to_numpy()).astype(
            "timedelta64[m]"
        ).astype(int)

        for prev, gap in zip(prev_rows.itertuples(index=False), gaps):
            if gap <= 1 or gap > max_gap_minutes:
                continue
            prev_close = getattr(prev, "close")
            if pd.isna(prev_close):
                continue
            for offset in range(1, int(gap)):
                ts = getattr(prev, "ts") + pd.Timedelta(minutes=offset)
                synthetic.append(
                    {
                        "ts": ts,
                        "market": market,
                        "year": year,
                        "symbol": getattr(prev, "symbol"),
                        "instrument_id": getattr(prev, "instrument_id"),
                        "publisher_id": getattr(prev, "publisher_id"),
                        "rtype": getattr(prev, "rtype"),
                        "open": prev_close,
                        "high": prev_close,
                        "low": prev_close,
                        "close": prev_close,
                        "volume": 0,
                        "raw_row_present": False,
                        "is_synthetic": True,
                        "valid_ohlcv": True,
                        "source_path": source_path,
                        "source_file_hash": source_hash,
                        "source_row_number": pd.NA,
                        "raw_schema_variant": getattr(prev, "raw_schema_variant"),
                        "timestamp_source": getattr(prev, "timestamp_source"),
                        "metadata_available": getattr(prev, "metadata_available"),
                        "roll_detection_available": getattr(prev, "roll_detection_available"),
                        "roll_detection_source": getattr(prev, "roll_detection_source"),
                        "roll_policy_status": getattr(prev, "roll_policy_status"),
                    }
                )

    if not synthetic:
        return pd.DataFrame(columns=df.columns)

    synth_df = pd.DataFrame(synthetic)
    synth_meta = _session_metadata(synth_df["ts"])
    return pd.concat([synth_df, synth_meta], axis=1)


def _add_roll_fields(df: pd.DataFrame, roll_window_bars: int) -> pd.DataFrame:
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)

    prev_symbol = df["symbol"].shift(1)
    prev_instrument = df["instrument_id"].shift(1)
    same_context = df["inside_session"] & df["inside_session"].shift(1, fill_value=False)

    symbol_known = df["symbol"].notna() & prev_symbol.notna()
    instrument_known = df["instrument_id"].notna() & prev_instrument.notna()
    df["symbol_change_flag"] = (same_context & symbol_known & (df["symbol"] != prev_symbol)).astype(bool)
    if bool(df["roll_detection_available"].fillna(False).any()):
        df["instrument_id_change_flag"] = (
            same_context & instrument_known & (df["instrument_id"] != prev_instrument)
        ).astype(bool)
        df["roll_boundary_flag"] = df["instrument_id_change_flag"].astype(bool)
    else:
        df["instrument_id_change_flag"] = False
        df["roll_boundary_flag"] = False

    roll_positions = np.flatnonzero(df["roll_boundary_flag"].to_numpy())
    n = len(df)
    if len(roll_positions) == 0:
        df["bars_since_roll"] = pd.Series([pd.NA] * n, dtype="Int64")
        df["bars_until_roll"] = pd.Series([pd.NA] * n, dtype="Int64")
        df["roll_window_flag"] = False
    else:
        positions = np.arange(n)
        last_roll_idx = np.searchsorted(roll_positions, positions, side="right") - 1
        next_roll_idx = np.searchsorted(roll_positions, positions, side="left")

        since = np.full(n, np.nan)
        has_last = last_roll_idx >= 0
        since[has_last] = positions[has_last] - roll_positions[last_roll_idx[has_last]]

        until = np.full(n, np.nan)
        has_next = next_roll_idx < len(roll_positions)
        until[has_next] = roll_positions[next_roll_idx[has_next]] - positions[has_next]

        df["bars_since_roll"] = pd.Series(since).round().astype("Int64")
        df["bars_until_roll"] = pd.Series(until).round().astype("Int64")
        df["roll_window_flag"] = (
            df["bars_since_roll"].le(roll_window_bars).fillna(False)
            | df["bars_until_roll"].le(roll_window_bars).fillna(False)
        ).astype(bool)

    segment_number = df.groupby("session_id", dropna=False)["roll_boundary_flag"].cumsum()
    df["session_segment_id"] = np.where(
        df["inside_session"],
        df["session_id"].astype("string") + "_seg" + segment_number.astype("int64").astype(str),
        pd.NA,
    )
    return df


def _add_session_edge_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_session_open"] = False
    df["is_session_close"] = False
    inside = df["inside_session"] & df["session_segment_id"].notna()
    if inside.any():
        first_idx = df.loc[inside].groupby("session_segment_id", sort=False)["ts"].idxmin()
        last_idx = df.loc[inside].groupby("session_segment_id", sort=False)["ts"].idxmax()
        df.loc[first_idx, "is_session_open"] = True
        df.loc[last_idx, "is_session_close"] = True
    return df


def _coerce_output_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    bool_cols = [
        "raw_row_present",
        "is_synthetic",
        "valid_ohlcv",
        "inside_session",
        "causal_valid",
        "is_session_open",
        "is_session_close",
        "roll_boundary_flag",
        "symbol_change_flag",
        "instrument_id_change_flag",
        "roll_window_flag",
        "metadata_available",
        "roll_detection_available",
    ]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)

    int_nullable_cols = [
        "instrument_id",
        "publisher_id",
        "rtype",
        "source_row_number",
        "bars_since_roll",
        "bars_until_roll",
    ]
    for col in int_nullable_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype("int64")
    df["minute_of_day"] = pd.to_numeric(df["minute_of_day"], errors="coerce").astype("Int64")
    df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").astype("Int64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[OUTPUT_COLUMNS]


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    profile: str,
    roll_window_bars: int = DEFAULT_ROLL_WINDOW_BARS,
    max_synthetic_gap_minutes: int = DEFAULT_MAX_SYNTHETIC_GAP_MINUTES,
) -> ValidationResult:
    market, year = infer_market_year(input_path)
    result = ValidationResult(
        profile=profile,
        market=market,
        year=year,
        input_path=relative_source_path(input_path),
        output_path=relative_source_path(output_path),
    )

    if not input_path.exists():
        result.failures.append("input file missing")
        return result

    result.source_file_hash = sha256_file(input_path)
    raw = pd.read_parquet(input_path)
    result.raw_rows = len(raw)
    if raw.empty:
        result.failures.append("empty file")
        return result

    result.missing_required_raw_cols = [
        c for c in REQUIRED_OHLCV_COLUMNS if c not in raw.columns
    ]
    if result.missing_required_raw_cols:
        result.failures.append("missing required OHLCV columns")
        return result

    ts, timestamp_source = _timestamp_from_raw(raw)
    if ts is None or timestamp_source is None:
        result.failures.append("missing timestamp source")
        return result

    result.timestamp_source = timestamp_source
    result.raw_schema_variant = _schema_variant(raw, timestamp_source)

    result.missing_audit_cols = [c for c in AUDIT_RAW_COLUMNS if c not in raw.columns]
    if result.missing_audit_cols:
        result.warnings.append("missing optional Databento audit metadata columns")
        for col in result.missing_audit_cols:
            raw[col] = pd.NA

    raw = raw.copy()
    raw["source_row_number"] = np.arange(len(raw), dtype=np.int64)
    raw["ts"] = pd.DatetimeIndex(ts)
    raw = raw.reset_index(drop=True)
    result.null_ts = int(raw["ts"].isna().sum())
    if result.null_ts:
        result.failures.append("null or unparseable timestamp")
        return result

    result.symbol_nonnull_count = int(raw["symbol"].notna().sum())
    result.instrument_id_nonnull_count = int(raw["instrument_id"].notna().sum())
    result.instrument_id_nunique = int(raw["instrument_id"].dropna().nunique())
    result.metadata_available = result.instrument_id_nonnull_count > 0
    result.roll_detection_available = result.metadata_available
    if result.roll_detection_available:
        result.roll_detection_source = "instrument_id"
        result.roll_policy_status = "active"
    else:
        result.roll_detection_source = "unavailable"
        result.roll_policy_status = "unavailable_metadata"
        result.warnings.append("roll detection unavailable: missing populated instrument_id")

    result.duplicate_timestamps = int(raw["ts"].duplicated().sum())
    if result.duplicate_timestamps:
        result.failures.append("duplicate ts_event rows")
        return result

    raw["valid_ohlcv"] = _valid_ohlcv(raw)
    result.invalid_ohlcv_rows = int((~raw["valid_ohlcv"]).sum())
    if result.invalid_ohlcv_rows:
        result.failures.append("invalid OHLCV rows")
        return result

    volume = pd.to_numeric(raw["volume"], errors="coerce")
    result.negative_volume_rows = int((volume < 0).sum())
    if result.negative_volume_rows:
        result.failures.append("negative volume rows")
        return result
    non_integer_volume = int(((volume.dropna() % 1) != 0).sum())
    if non_integer_volume:
        result.warnings.append(f"non-integer-like volume rows={non_integer_volume}")

    raw = raw.sort_values("ts", kind="mergesort").reset_index(drop=True)
    raw_meta = _session_metadata(raw["ts"])
    df = pd.concat([raw, raw_meta], axis=1)
    df["market"] = market
    df["year"] = year
    df["raw_row_present"] = True
    df["is_synthetic"] = False
    df["source_path"] = relative_source_path(input_path)
    df["source_file_hash"] = result.source_file_hash
    df["raw_schema_variant"] = result.raw_schema_variant
    df["timestamp_source"] = result.timestamp_source
    df["metadata_available"] = result.metadata_available
    df["roll_detection_available"] = result.roll_detection_available
    df["roll_detection_source"] = result.roll_detection_source
    df["roll_policy_status"] = result.roll_policy_status

    base_cols = [
        "ts",
        "market",
        "year",
        "symbol",
        "instrument_id",
        "publisher_id",
        "rtype",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "raw_row_present",
        "is_synthetic",
        "valid_ohlcv",
        "inside_session",
        "session_id",
        "session_date",
        "session_template",
        "minutes_since_session_open",
        "minutes_until_session_close",
        "session_progress",
        "minute_of_day",
        "day_of_week",
        "source_path",
        "source_file_hash",
        "source_row_number",
        "raw_schema_variant",
        "timestamp_source",
        "metadata_available",
        "roll_detection_available",
        "roll_detection_source",
        "roll_policy_status",
    ]
    df = df[base_cols]

    synthetic = _build_synthetic_rows(
        df,
        market,
        year,
        relative_source_path(input_path),
        result.source_file_hash,
        max_synthetic_gap_minutes,
    )
    if not synthetic.empty:
        result.warnings.append("synthetic rows inserted")
        df = pd.concat([df, synthetic[base_cols]], ignore_index=True)

    df = _add_roll_fields(df, roll_window_bars)
    df = _add_session_edge_flags(df)
    df["causal_valid"] = (
        df["raw_row_present"]
        & ~df["is_synthetic"]
        & df["valid_ohlcv"]
        & df["inside_session"]
        & ~df["roll_window_flag"]
    ).astype(bool)

    result.output_rows = len(df)
    result.synthetic_rows = int(df["is_synthetic"].sum())
    result.outside_session_rows = int((~df["inside_session"]).sum())
    result.roll_boundary_rows = int(df["roll_boundary_flag"].sum())
    result.roll_window_rows = int(df["roll_window_flag"].sum())

    if result.synthetic_rows:
        missing_minutes = result.synthetic_rows
        result.warnings.append(f"missing session minutes filled={missing_minutes}")
    if result.outside_session_rows:
        result.warnings.append(f"outside-session raw rows={result.outside_session_rows}")
    if result.roll_boundary_rows:
        result.warnings.append(f"roll boundary rows={result.roll_boundary_rows}")
    if result.roll_window_rows:
        result.warnings.append(f"roll exclusion rows={result.roll_window_rows}")

    output = _coerce_output_types(df.sort_values("ts", kind="mergesort").reset_index(drop=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return result


def write_reports(results: Iterable[ValidationResult], reports_root: Path, profile: str) -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    csv_rows = [result.to_csv_row() for result in results]

    validation_json = {
        "profile": profile,
        "stage": "causal_base",
        "status": "FAIL" if any(r["status"] == "FAIL" for r in rows) else "WARN"
        if any(r["status"] == "WARN" for r in rows)
        else "PASS",
        "files": rows,
        "summary": {
            "file_count": len(rows),
            "pass_count": sum(r["status"] == "PASS" for r in rows),
            "warn_count": sum(r["status"] == "WARN" for r in rows),
            "fail_count": sum(r["status"] == "FAIL" for r in rows),
            "raw_rows": int(sum(r["raw_rows"] for r in rows)),
            "output_rows": int(sum(r["output_rows"] for r in rows)),
            "synthetic_rows": int(sum(r["synthetic_rows"] for r in rows)),
            "roll_boundary_rows": int(sum(r["roll_boundary_rows"] for r in rows)),
            "roll_window_rows": int(sum(r["roll_window_rows"] for r in rows)),
            "roll_boundary_count": int(sum(r["roll_boundary_rows"] for r in rows)),
            "roll_window_count": int(sum(r["roll_window_rows"] for r in rows)),
        },
    }

    manifest = {
        "profile": profile,
        "stage": "causal_base",
        "status": validation_json["status"],
        "outputs": [
            {
                "market": row["market"],
                "year": row["year"],
                "input_path": row["input_path"],
                "output_path": row["output_path"],
                "source_file_hash": row["source_file_hash"],
                "raw_rows": row["raw_rows"],
                "output_rows": row["output_rows"],
                "synthetic_rows": row["synthetic_rows"],
                "raw_schema_variant": row["raw_schema_variant"],
                "timestamp_source": row["timestamp_source"],
                "metadata_available": row["metadata_available"],
                "roll_detection_available": row["roll_detection_available"],
                "roll_detection_source": row["roll_detection_source"],
                "roll_policy_status": row["roll_policy_status"],
                "symbol_nonnull_count": row["symbol_nonnull_count"],
                "instrument_id_nonnull_count": row["instrument_id_nonnull_count"],
                "instrument_id_nunique": row["instrument_id_nunique"],
                "roll_boundary_count": row["roll_boundary_count"],
                "roll_window_count": row["roll_window_count"],
                "warnings": row["warnings"],
                "status": row["status"],
            }
            for row in rows
        ],
        "summary": validation_json["summary"],
    }

    (reports_root / "causal_base_validation.json").write_text(
        json.dumps(validation_json, indent=2), encoding="utf-8"
    )
    (reports_root / "causal_base_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pd.DataFrame(csv_rows).to_csv(reports_root / "causal_base_validation.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=(
            "Use all_raw to process every top-level raw market/year file, or "
            "tier_1_CL_ES_ZN for the small legacy smoke-test subset."
        ),
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data/causally_gated_normalized")
    parser.add_argument("--reports-root", default="reports/causal_base")
    parser.add_argument("--roll-window-bars", type=int, default=DEFAULT_ROLL_WINDOW_BARS)
    parser.add_argument(
        "--max-synthetic-gap-minutes",
        type=int,
        default=DEFAULT_MAX_SYNTHETIC_GAP_MINUTES,
        help="Fill only missing in-session gaps up to this size.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    inputs = resolve_profile_inputs(args.profile, raw_root)

    results: list[ValidationResult] = []
    for market, year, input_path in inputs:
        output_path = output_root / market / f"{year}.parquet"
        result = process_file(
            input_path,
            output_path,
            profile=args.profile,
            roll_window_bars=args.roll_window_bars,
            max_synthetic_gap_minutes=args.max_synthetic_gap_minutes,
        )
        results.append(result)
        print(
            f"{result.status} {market} {year}: raw={result.raw_rows} "
            f"out={result.output_rows} synthetic={result.synthetic_rows} "
            f"warnings={len(result.warnings)} failures={len(result.failures)}"
        )

    write_reports(results, reports_root, args.profile)
    return 1 if any(result.failures for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
