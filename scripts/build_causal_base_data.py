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
import yaml


DEFAULT_PROFILE = "all_raw"
DISCOVERY_PROFILES = {"all_raw", "all_raw_data"}
DEFAULT_PROFILE_CONFIG = Path("configs/alpha_tiered.yaml")
DEFAULT_SESSION_CONFIG = Path("configs/market_sessions.yaml")

# Discovery profiles process every top-level data/raw/{market}/{year}.parquet file.
# Static profiles are optional limited subsets for faster smoke tests.
STATIC_PROFILE_MARKETS = {
    "tier_1_CL_ES_ZN": ["CL", "ES", "ZN"],
    "tier_1_core": ["CL", "ES", "ZN"],
}

STATIC_PROFILE_YEARS = {
    "tier_1_CL_ES_ZN": [2023, 2024, 2025],
    "tier_1_core": [2023, 2024, 2025],
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
    "synthetic_gap_id",
    "synthetic_gap_size_minutes",
    "synthetic_gap_reason",
    "valid_ohlcv",
    "data_quality_status",
    "data_quality_degraded",
    "session_data_quality_degraded",
    "trainable_data_quality",
    "inside_session",
    "causal_valid",
    "session_id",
    "session_date",
    "session_segment_id",
    "boundary_session_flag",
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
DEFAULT_MAX_SYNTHETIC_ROWS_PCT = 2.0
DEFAULT_MAX_DEGRADED_ROWS_PCT = 1.0
DEFAULT_MAX_ROLL_WINDOW_ROWS_PCT = 1.0
DEFAULT_REQUIRE_ROLL_METADATA_PROFILES = {
    "tier_1_core",
    "tier_1_CL_ES_ZN",
    "tier_1_core_recent",
    "tier_1_core_long",
    "tier_2_liquid",
    "tier_2_liquid_recent",
    "tier_2_liquid_long",
    "tier_3_full",
    "tier_3_full_long",
}


@dataclass(frozen=True)
class CausalBaseConfig:
    max_synthetic_rows_pct: float = DEFAULT_MAX_SYNTHETIC_ROWS_PCT
    max_synthetic_gap_minutes: int = DEFAULT_MAX_SYNTHETIC_GAP_MINUTES
    max_degraded_rows_pct: float = DEFAULT_MAX_DEGRADED_ROWS_PCT
    max_roll_window_rows_pct: float = DEFAULT_MAX_ROLL_WINDOW_ROWS_PCT
    require_roll_metadata_for_profiles: tuple[str, ...] = tuple(
        sorted(DEFAULT_REQUIRE_ROLL_METADATA_PROFILES)
    )


@dataclass(frozen=True)
class SessionCalendar:
    session_template: str = SESSION_TEMPLATE
    timezone: str = EXCHANGE_TZ
    regular_open: str = "17:00"
    regular_close: str = "16:00"
    holidays: frozenset[str] = frozenset()
    closed_dates: frozenset[str] = frozenset()
    early_closes: dict[str, str] = field(default_factory=dict)
    source: str = "hardcoded_regular_session"
    config_available: bool = False

    @property
    def status(self) -> str:
        if self.config_available and (
            bool(self.holidays) or bool(self.closed_dates) or bool(self.early_closes)
        ):
            return "config_backed"
        if self.config_available:
            return "config_backed_regular_session"
        return "hardcoded_regular_session"

    @property
    def holiday_calendar_available(self) -> bool:
        return bool(self.holidays) or bool(self.closed_dates)

    @property
    def early_close_calendar_available(self) -> bool:
        return bool(self.early_closes)


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
    synthetic_gap_count: int = 0
    max_synthetic_gap_minutes: int = 0
    synthetic_rows_pct: float = 0.0
    synthetic_gap_threshold_breached: bool = False
    outside_session_rows: int = 0
    roll_boundary_rows: int = 0
    roll_window_rows: int = 0
    roll_window_rows_pct: float = 0.0
    roll_window_threshold_breached: bool = False
    boundary_session_rows: int = 0
    causal_valid_rows: int = 0
    causal_invalid_rows: int = 0
    raw_schema_variant: str | None = None
    timestamp_source: str | None = None
    metadata_available: bool = False
    roll_detection_available: bool = False
    roll_detection_source: str = "unavailable"
    roll_policy_status: str = "unavailable_metadata"
    session_calendar_status: str = "hardcoded_regular_session"
    holiday_calendar_available: bool = False
    early_close_calendar_available: bool = False
    symbol_nonnull_count: int = 0
    instrument_id_nonnull_count: int = 0
    instrument_id_nunique: int = 0
    missing_required_raw_cols: list[str] = field(default_factory=list)
    missing_audit_cols: list[str] = field(default_factory=list)
    duplicate_timestamps: int = 0
    null_ts: int = 0
    invalid_ohlcv_rows: int = 0
    negative_volume_rows: int = 0
    degraded_bar_rows: int = 0
    degraded_session_rows: int = 0
    degraded_rows_pct: float = 0.0
    degraded_threshold_breached: bool = False
    max_synthetic_rows_pct_threshold: float = DEFAULT_MAX_SYNTHETIC_ROWS_PCT
    max_synthetic_gap_minutes_threshold: int = DEFAULT_MAX_SYNTHETIC_GAP_MINUTES
    max_degraded_rows_pct_threshold: float = DEFAULT_MAX_DEGRADED_ROWS_PCT
    max_roll_window_rows_pct_threshold: float = DEFAULT_MAX_ROLL_WINDOW_ROWS_PCT
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


def _read_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def load_causal_base_config(profile_config_path: Path = DEFAULT_PROFILE_CONFIG) -> CausalBaseConfig:
    payload = _read_yaml(profile_config_path)
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    causal_base = payload.get("causal_base", {})
    if not isinstance(causal_base, dict):
        causal_base = {}

    def get_float(name: str, default: float) -> float:
        value = causal_base.get(name, defaults.get(name, default))
        return float(value)

    def get_int(name: str, default: int) -> int:
        value = causal_base.get(name, defaults.get(name, default))
        return int(value)

    required = causal_base.get(
        "require_roll_metadata_for_profiles",
        defaults.get("require_roll_metadata_for_profiles", sorted(DEFAULT_REQUIRE_ROLL_METADATA_PROFILES)),
    )
    if not isinstance(required, list):
        required = sorted(DEFAULT_REQUIRE_ROLL_METADATA_PROFILES)

    return CausalBaseConfig(
        max_synthetic_rows_pct=get_float(
            "max_synthetic_rows_pct", DEFAULT_MAX_SYNTHETIC_ROWS_PCT
        ),
        max_synthetic_gap_minutes=get_int(
            "max_synthetic_gap_minutes", DEFAULT_MAX_SYNTHETIC_GAP_MINUTES
        ),
        max_degraded_rows_pct=get_float(
            "max_degraded_rows_pct", DEFAULT_MAX_DEGRADED_ROWS_PCT
        ),
        max_roll_window_rows_pct=get_float(
            "max_roll_window_rows_pct", DEFAULT_MAX_ROLL_WINDOW_ROWS_PCT
        ),
        require_roll_metadata_for_profiles=tuple(str(item) for item in required),
    )


def load_profile_map(profile_config_path: Path = DEFAULT_PROFILE_CONFIG) -> tuple[
    dict[str, list[str]], dict[str, list[int]], dict[str, str], set[str]
]:
    markets = {key: value[:] for key, value in STATIC_PROFILE_MARKETS.items()}
    years = {key: value[:] for key, value in STATIC_PROFILE_YEARS.items()}
    aliases: dict[str, str] = {}
    discovery = set(DISCOVERY_PROFILES)

    payload = _read_yaml(profile_config_path)
    raw_aliases = payload.get("aliases", {})
    if isinstance(raw_aliases, dict):
        aliases.update({str(k): str(v) for k, v in raw_aliases.items()})
    profiles = payload.get("profiles", {})
    if isinstance(profiles, dict):
        default_years = payload.get("defaults", {}).get("years", []) if isinstance(payload.get("defaults", {}), dict) else []
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            profile_name = str(name)
            if profile.get("discovery"):
                discovery.add(profile_name)
                continue
            profile_markets = profile.get("markets", [])
            profile_years = profile.get("years", default_years)
            if isinstance(profile_markets, list) and isinstance(profile_years, list):
                markets[profile_name] = [str(item) for item in profile_markets]
                years[profile_name] = [int(item) for item in profile_years]

    return markets, years, aliases, discovery


def resolve_profile_name(profile: str, aliases: dict[str, str]) -> str:
    seen: set[str] = set()
    resolved = profile
    while resolved in aliases and resolved not in seen:
        seen.add(resolved)
        resolved = aliases[resolved]
    return resolved


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour_text, minute_text = value.split(":", 1)
    return int(hour_text), int(minute_text)


def load_session_calendar(
    market: str,
    config_path: Path = DEFAULT_SESSION_CONFIG,
    *,
    allow_hardcoded_calendar: bool = False,
) -> SessionCalendar:
    if not config_path.exists():
        if not allow_hardcoded_calendar:
            raise FileNotFoundError(
                f"Session calendar config missing: {config_path}. "
                "Use --allow-hardcoded-calendar only for tests."
            )
        return SessionCalendar(source="hardcoded_regular_session", config_available=False)

    payload = _read_yaml(config_path)
    markets = payload.get("markets", {})
    templates = payload.get("session_templates", {})
    if not isinstance(markets, dict) or not isinstance(templates, dict):
        raise ValueError("market_sessions.yaml requires markets and session_templates mappings")

    market_cfg = markets.get(market, markets.get("default", {}))
    if not isinstance(market_cfg, dict):
        market_cfg = {}
    template_name = str(market_cfg.get("session_template", SESSION_TEMPLATE))
    template = templates.get(template_name)
    if not isinstance(template, dict):
        raise ValueError(f"Missing session template {template_name!r} for {market}")

    holidays = template.get("holidays", [])
    closed_dates = template.get("closed_dates", [])
    early_closes = template.get("early_closes", {})
    if not isinstance(holidays, list):
        holidays = []
    if not isinstance(closed_dates, list):
        closed_dates = []
    if not isinstance(early_closes, dict):
        early_closes = {}

    return SessionCalendar(
        session_template=template_name,
        timezone=str(template.get("timezone", EXCHANGE_TZ)),
        regular_open=str(template.get("regular_open", "17:00")),
        regular_close=str(template.get("regular_close", "16:00")),
        holidays=frozenset(str(item) for item in holidays),
        closed_dates=frozenset(str(item) for item in closed_dates),
        early_closes={str(k): str(v) for k, v in early_closes.items()},
        source=relative_source_path(config_path),
        config_available=True,
    )


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


def resolve_profile_inputs(
    profile: str,
    raw_root: Path,
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
) -> list[tuple[str, int, Path]]:
    profile_markets, profile_years, aliases, discovery_profiles = load_profile_map(
        profile_config_path
    )
    resolved_profile = resolve_profile_name(profile, aliases)
    if resolved_profile in discovery_profiles:
        return discover_raw_inputs(raw_root)

    if resolved_profile not in profile_markets:
        known = ", ".join(sorted([*profile_markets, *discovery_profiles]))
        raise SystemExit(f"Unknown profile {profile!r}. Known profiles: {known}")

    inputs: list[tuple[str, int, Path]] = []
    for market in profile_markets[resolved_profile]:
        for year in profile_years[resolved_profile]:
            inputs.append((market, year, raw_root / market / f"{year}.parquet"))
    return inputs


def infer_market_year(path: Path) -> tuple[str, int]:
    try:
        return path.parent.name, int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Cannot infer market/year from {path}") from exc


def _session_metadata(ts: pd.Series, calendar: SessionCalendar | None = None) -> pd.DataFrame:
    calendar = calendar or SessionCalendar()
    local = ts.dt.tz_convert(calendar.timezone)
    dow = local.dt.dayofweek
    minutes = local.dt.hour * 60 + local.dt.minute
    open_hour, open_minute = _parse_hhmm(calendar.regular_open)
    close_hour, close_minute = _parse_hhmm(calendar.regular_close)
    open_minutes = open_hour * 60 + open_minute
    close_minutes = close_hour * 60 + close_minute

    after_open = minutes >= open_minutes
    before_close = minutes < close_minutes

    local_midnight = local.dt.normalize().dt.tz_localize(None)
    session_date = local_midnight.where(before_close, local_midnight + pd.Timedelta(days=1))
    session_date = pd.to_datetime(session_date.dt.date)

    session_date_str = session_date.dt.strftime("%Y-%m-%d")
    close_times = session_date_str.map(calendar.early_closes).fillna(calendar.regular_close)
    close_parts = close_times.str.split(":", n=1, expand=True).astype(int)

    open_naive = (
        session_date
        - pd.Timedelta(days=1)
        + pd.to_timedelta(open_hour, unit="h")
        + pd.to_timedelta(open_minute, unit="m")
    )
    close_naive = (
        session_date
        + pd.to_timedelta(close_parts[0], unit="h")
        + pd.to_timedelta(close_parts[1], unit="m")
    )
    open_local = open_naive.dt.tz_localize(calendar.timezone)
    close_local = close_naive.dt.tz_localize(calendar.timezone)

    trade_date_open = session_date.dt.dayofweek.isin([0, 1, 2, 3, 4])
    closed = session_date_str.isin(calendar.holidays | calendar.closed_dates)
    inside = (local >= open_local) & (local < close_local) & trade_date_open & ~closed

    since_open = (local - open_local).dt.total_seconds() / 60.0
    until_close = (close_local - local).dt.total_seconds() / 60.0
    denom = since_open + until_close
    progress = np.where(denom > 0, since_open / denom, np.nan)

    session_id = session_date_str.where(inside, pd.NA)

    metadata = pd.DataFrame(
        {
            "inside_session": inside.astype(bool),
            "session_date": session_date_str.where(inside, pd.NA),
            "session_id": session_id.where(session_id.isna(), "session_" + session_id),
            "session_template": np.where(inside, calendar.session_template, pd.NA),
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


def _prepare_raw_frame(
    input_path: Path,
    *,
    market: str,
    year: int,
    result: ValidationResult | None,
    required: bool,
) -> pd.DataFrame | None:
    if not input_path.exists():
        if required and result is not None:
            result.failures.append("input file missing")
        return None

    source_hash = sha256_file(input_path)
    raw = pd.read_parquet(input_path)
    if required and result is not None:
        result.source_file_hash = source_hash
        result.raw_rows = len(raw)
    if raw.empty:
        if required and result is not None:
            result.failures.append("empty file")
        return None

    missing_required = [c for c in REQUIRED_OHLCV_COLUMNS if c not in raw.columns]
    if missing_required:
        if required and result is not None:
            result.missing_required_raw_cols = missing_required
            result.failures.append("missing required OHLCV columns")
        return None

    ts, timestamp_source = _timestamp_from_raw(raw)
    if ts is None or timestamp_source is None:
        if required and result is not None:
            result.failures.append("missing timestamp source")
        return None

    raw_schema_variant = _schema_variant(raw, timestamp_source)
    missing_audit = [c for c in AUDIT_RAW_COLUMNS if c not in raw.columns]
    raw = raw.copy()
    for col in missing_audit:
        raw[col] = pd.NA

    raw["source_row_number"] = np.arange(len(raw), dtype=np.int64)
    raw["ts"] = pd.DatetimeIndex(ts)
    raw = raw.reset_index(drop=True)
    null_ts = int(raw["ts"].isna().sum())
    if null_ts:
        if required and result is not None:
            result.null_ts = null_ts
            result.failures.append("null or unparseable timestamp")
        return None

    duplicate_timestamps = int(raw["ts"].duplicated().sum())
    if duplicate_timestamps:
        if required and result is not None:
            result.duplicate_timestamps = duplicate_timestamps
            result.failures.append("duplicate ts_event rows")
        return None

    raw["valid_ohlcv"] = _valid_ohlcv(raw)
    invalid_ohlcv_rows = int((~raw["valid_ohlcv"]).sum())
    if invalid_ohlcv_rows:
        if required and result is not None:
            result.invalid_ohlcv_rows = invalid_ohlcv_rows
            result.failures.append("invalid OHLCV rows")
        return None

    volume = pd.to_numeric(raw["volume"], errors="coerce")
    negative_volume_rows = int((volume < 0).sum())
    if negative_volume_rows:
        if required and result is not None:
            result.negative_volume_rows = negative_volume_rows
            result.failures.append("negative volume rows")
        return None

    if "data_quality_status" not in raw.columns:
        raw["data_quality_status"] = "unknown"
        if required and result is not None:
            result.warnings.append("missing data_quality_status; assuming trainable data quality")
    else:
        raw["data_quality_status"] = raw["data_quality_status"].fillna("unknown").astype(str)

    if "data_quality_degraded" not in raw.columns:
        raw["data_quality_degraded"] = False
        if required and result is not None:
            result.warnings.append("missing data_quality_degraded; assuming no degraded bars")
    else:
        raw["data_quality_degraded"] = raw["data_quality_degraded"].fillna(False).astype(bool)

    if required and result is not None:
        result.timestamp_source = timestamp_source
        result.raw_schema_variant = raw_schema_variant
        result.missing_audit_cols = missing_audit
        if missing_audit:
            result.warnings.append("missing optional Databento audit metadata columns")
        result.symbol_nonnull_count = int(raw["symbol"].notna().sum())
        result.instrument_id_nonnull_count = int(raw["instrument_id"].notna().sum())
        result.instrument_id_nunique = int(raw["instrument_id"].dropna().nunique())
        result.degraded_bar_rows = int(raw["data_quality_degraded"].sum())
        non_integer_volume = int(((volume.dropna() % 1) != 0).sum())
        if non_integer_volume:
            result.warnings.append(f"non-integer-like volume rows={non_integer_volume}")

    raw = raw.sort_values("ts", kind="mergesort").reset_index(drop=True)
    raw["market"] = market
    raw["year"] = year
    raw["raw_row_present"] = True
    raw["is_synthetic"] = False
    raw["synthetic_gap_id"] = pd.NA
    raw["synthetic_gap_size_minutes"] = pd.NA
    raw["synthetic_gap_reason"] = pd.NA
    raw["source_path"] = relative_source_path(input_path)
    raw["source_file_hash"] = source_hash
    raw["raw_schema_variant"] = raw_schema_variant
    raw["timestamp_source"] = timestamp_source
    return raw


def _build_synthetic_rows(
    df: pd.DataFrame,
    max_gap_minutes: int,
    calendar: SessionCalendar,
) -> pd.DataFrame:
    synthetic: list[dict[str, object]] = []
    inside = df[df["inside_session"]].sort_values("ts")
    gap_id = 0

    for _, group in inside.groupby("session_id", sort=False):
        group = group.sort_values("ts")
        if len(group) < 2:
            continue
        prev_rows = group.iloc[:-1]
        next_rows = group.iloc[1:]
        gaps = (next_rows["ts"].to_numpy() - prev_rows["ts"].to_numpy()).astype(
            "timedelta64[m]"
        ).astype(int)

        for prev, next_row, gap in zip(
            prev_rows.itertuples(index=False),
            next_rows.itertuples(index=False),
            gaps,
        ):
            if gap <= 1 or gap > max_gap_minutes:
                continue
            prev_instrument = getattr(prev, "instrument_id")
            next_instrument = getattr(next_row, "instrument_id")
            if (
                pd.notna(prev_instrument)
                and pd.notna(next_instrument)
                and prev_instrument != next_instrument
            ):
                continue
            prev_close = getattr(prev, "close")
            if pd.isna(prev_close):
                continue
            gap_id += 1
            for offset in range(1, int(gap)):
                ts = getattr(prev, "ts") + pd.Timedelta(minutes=offset)
                synthetic.append(
                    {
                        "ts": ts,
                        "market": getattr(prev, "market"),
                        "year": int(ts.year),
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
                        "synthetic_gap_id": gap_id,
                        "synthetic_gap_size_minutes": int(gap),
                        "synthetic_gap_reason": "missing_in_session_minute",
                        "valid_ohlcv": True,
                        "data_quality_status": getattr(prev, "data_quality_status"),
                        "data_quality_degraded": getattr(prev, "data_quality_degraded"),
                        "source_path": getattr(prev, "source_path"),
                        "source_file_hash": getattr(prev, "source_file_hash"),
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
    synth_meta = _session_metadata(synth_df["ts"], calendar)
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


def _add_boundary_session_flag(
    df: pd.DataFrame,
    *,
    target_year: int,
    has_previous_context: bool,
    has_next_context: bool,
) -> pd.DataFrame:
    df["boundary_session_flag"] = False
    inside = (
        df["inside_session"]
        & df["session_segment_id"].notna()
        & df["year"].eq(target_year)
    )
    if not inside.any():
        return df

    for _, group in df.loc[inside].groupby(["market", "year"], sort=False):
        ordered_segments = group.sort_values("ts", kind="mergesort")["session_segment_id"]
        if ordered_segments.empty:
            continue
        boundary_segments: set[object] = set()
        if not has_previous_context:
            boundary_segments.add(ordered_segments.iloc[0])
        if not has_next_context:
            boundary_segments.add(ordered_segments.iloc[-1])
        df.loc[df["session_segment_id"].isin(boundary_segments), "boundary_session_flag"] = True
    return df


def _coerce_output_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    bool_cols = [
        "raw_row_present",
        "is_synthetic",
        "valid_ohlcv",
        "data_quality_degraded",
        "session_data_quality_degraded",
        "trainable_data_quality",
        "inside_session",
        "causal_valid",
        "boundary_session_flag",
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
        "synthetic_gap_id",
        "synthetic_gap_size_minutes",
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
    df["data_quality_status"] = df["data_quality_status"].fillna("unknown").astype(str)
    df["synthetic_gap_reason"] = df["synthetic_gap_reason"].astype("string")

    return df[OUTPUT_COLUMNS]


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    profile: str,
    roll_window_bars: int = DEFAULT_ROLL_WINDOW_BARS,
    max_synthetic_gap_minutes: int = DEFAULT_MAX_SYNTHETIC_GAP_MINUTES,
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    session_config_path: Path = DEFAULT_SESSION_CONFIG,
    allow_hardcoded_calendar: bool = False,
) -> ValidationResult:
    market, year = infer_market_year(input_path)
    config = load_causal_base_config(profile_config_path)
    _, _, aliases, _ = load_profile_map(profile_config_path)
    resolved_profile = resolve_profile_name(profile, aliases)
    effective_max_synthetic_gap_minutes = (
        config.max_synthetic_gap_minutes
        if max_synthetic_gap_minutes == DEFAULT_MAX_SYNTHETIC_GAP_MINUTES
        else max_synthetic_gap_minutes
    )
    calendar = load_session_calendar(
        market,
        session_config_path,
        allow_hardcoded_calendar=allow_hardcoded_calendar,
    )
    result = ValidationResult(
        profile=profile,
        market=market,
        year=year,
        input_path=relative_source_path(input_path),
        output_path=relative_source_path(output_path),
        session_calendar_status=calendar.status,
        holiday_calendar_available=calendar.holiday_calendar_available,
        early_close_calendar_available=calendar.early_close_calendar_available,
        max_synthetic_rows_pct_threshold=config.max_synthetic_rows_pct,
        max_synthetic_gap_minutes_threshold=effective_max_synthetic_gap_minutes,
        max_degraded_rows_pct_threshold=config.max_degraded_rows_pct,
        max_roll_window_rows_pct_threshold=config.max_roll_window_rows_pct,
    )

    if not calendar.config_available:
        result.warnings.append("hardcoded session calendar used")

    current_raw = _prepare_raw_frame(
        input_path,
        market=market,
        year=year,
        result=result,
        required=True,
    )
    if result.failures or current_raw is None:
        return result

    result.metadata_available = result.instrument_id_nonnull_count > 0
    result.roll_detection_available = result.metadata_available
    if result.roll_detection_available:
        result.roll_detection_source = "instrument_id"
        result.roll_policy_status = "active"
    else:
        result.roll_detection_source = "unavailable"
        result.roll_policy_status = "unavailable_metadata"
        if resolved_profile in config.require_roll_metadata_for_profiles:
            result.failures.append("required roll metadata unavailable")
            return result
        result.warnings.append("roll detection unavailable: missing populated instrument_id")

    frames = []
    prev_raw = _prepare_raw_frame(
        input_path.parent / f"{year - 1}.parquet",
        market=market,
        year=year - 1,
        result=None,
        required=False,
    )
    if prev_raw is not None:
        frames.append(prev_raw)
    frames.append(current_raw)
    next_raw = _prepare_raw_frame(
        input_path.parent / f"{year + 1}.parquet",
        market=market,
        year=year + 1,
        result=None,
        required=False,
    )
    if next_raw is not None:
        frames.append(next_raw)

    raw_all = pd.concat(frames, ignore_index=True).sort_values("ts", kind="mergesort").reset_index(drop=True)
    raw_meta = _session_metadata(raw_all["ts"], calendar)
    df = pd.concat([raw_all, raw_meta], axis=1)
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
        "synthetic_gap_id",
        "synthetic_gap_size_minutes",
        "synthetic_gap_reason",
        "valid_ohlcv",
        "data_quality_status",
        "data_quality_degraded",
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
        effective_max_synthetic_gap_minutes,
        calendar,
    )
    if not synthetic.empty:
        df = pd.concat([df, synthetic[base_cols]], ignore_index=True)

    df = _add_roll_fields(df, roll_window_bars)
    df = _add_session_edge_flags(df)
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    year_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    has_previous_context = bool(
        (df["ts"].lt(year_start) & df["inside_session"]).any()
    )
    has_next_context = bool(
        (df["ts"].ge(year_end) & df["inside_session"]).any()
    )
    df = df[df["ts"].ge(year_start) & df["ts"].lt(year_end)].copy()
    df = _add_boundary_session_flag(
        df,
        target_year=year,
        has_previous_context=has_previous_context,
        has_next_context=has_next_context,
    )
    df["data_quality_degraded"] = df["data_quality_degraded"].fillna(False).astype(bool)
    df["session_data_quality_degraded"] = False
    session_mask = df["session_id"].notna()
    if session_mask.any():
        df.loc[session_mask, "session_data_quality_degraded"] = (
            df.loc[session_mask]
            .groupby("session_id", sort=False)["data_quality_degraded"]
            .transform("any")
            .astype(bool)
        )
    df["trainable_data_quality"] = ~df["session_data_quality_degraded"].astype(bool)
    df["causal_valid"] = (
        df["raw_row_present"]
        & ~df["is_synthetic"]
        & df["valid_ohlcv"]
        & df["inside_session"]
        & df["trainable_data_quality"]
        & ~df["roll_window_flag"]
        & ~df["boundary_session_flag"]
    ).astype(bool)

    result.output_rows = len(df)
    result.synthetic_rows = int(df["is_synthetic"].sum())
    result.synthetic_gap_count = int(df["synthetic_gap_id"].dropna().nunique())
    synthetic_gap_sizes = pd.to_numeric(df["synthetic_gap_size_minutes"], errors="coerce")
    result.max_synthetic_gap_minutes = (
        int(synthetic_gap_sizes.max()) if synthetic_gap_sizes.notna().any() else 0
    )
    result.synthetic_rows_pct = (
        round(100.0 * result.synthetic_rows / result.output_rows, 6)
        if result.output_rows
        else 0.0
    )
    result.outside_session_rows = int((~df["inside_session"]).sum())
    result.roll_boundary_rows = int(df["roll_boundary_flag"].sum())
    result.roll_window_rows = int(df["roll_window_flag"].sum())
    result.roll_window_rows_pct = (
        round(100.0 * result.roll_window_rows / result.output_rows, 6)
        if result.output_rows
        else 0.0
    )
    result.boundary_session_rows = int(df["boundary_session_flag"].sum())
    result.causal_valid_rows = int(df["causal_valid"].sum())
    result.causal_invalid_rows = result.output_rows - result.causal_valid_rows
    result.degraded_session_rows = int(
        df.loc[df["is_session_open"], "session_data_quality_degraded"].sum()
    )
    result.degraded_rows_pct = (
        round(100.0 * result.degraded_bar_rows / result.raw_rows, 6)
        if result.raw_rows
        else 0.0
    )
    result.synthetic_gap_threshold_breached = (
        result.synthetic_rows_pct > config.max_synthetic_rows_pct
        or result.max_synthetic_gap_minutes > effective_max_synthetic_gap_minutes
    )
    result.roll_window_threshold_breached = (
        result.roll_window_rows_pct > config.max_roll_window_rows_pct
    )
    result.degraded_threshold_breached = (
        result.degraded_rows_pct > config.max_degraded_rows_pct
    )

    if result.synthetic_gap_threshold_breached:
        result.warnings.append(
            "synthetic threshold breached: "
            f"rows_pct={result.synthetic_rows_pct} "
            f"max_gap_minutes={result.max_synthetic_gap_minutes}"
        )
    if result.roll_window_threshold_breached:
        result.warnings.append(
            "roll exclusion threshold breached: "
            f"rows_pct={result.roll_window_rows_pct} rows={result.roll_window_rows}"
        )
    if result.degraded_threshold_breached:
        result.warnings.append(
            "degraded threshold breached: "
            f"rows_pct={result.degraded_rows_pct} bars={result.degraded_bar_rows} "
            f"sessions={result.degraded_session_rows}"
        )

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
            "synthetic_gap_count": int(sum(r["synthetic_gap_count"] for r in rows)),
            "max_synthetic_gap_minutes": int(
                max([r["max_synthetic_gap_minutes"] for r in rows] or [0])
            ),
            "synthetic_gap_threshold_breached_files": int(
                sum(bool(r["synthetic_gap_threshold_breached"]) for r in rows)
            ),
            "roll_boundary_rows": int(sum(r["roll_boundary_rows"] for r in rows)),
            "roll_window_rows": int(sum(r["roll_window_rows"] for r in rows)),
            "roll_window_threshold_breached_files": int(
                sum(bool(r["roll_window_threshold_breached"]) for r in rows)
            ),
            "roll_boundary_count": int(sum(r["roll_boundary_rows"] for r in rows)),
            "roll_window_count": int(sum(r["roll_window_rows"] for r in rows)),
            "boundary_session_rows": int(sum(r["boundary_session_rows"] for r in rows)),
            "causal_valid_rows": int(sum(r["causal_valid_rows"] for r in rows)),
            "causal_invalid_rows": int(sum(r["causal_invalid_rows"] for r in rows)),
            "degraded_bar_rows": int(sum(r["degraded_bar_rows"] for r in rows)),
            "degraded_session_rows": int(sum(r["degraded_session_rows"] for r in rows)),
            "degraded_threshold_breached_files": int(
                sum(bool(r["degraded_threshold_breached"]) for r in rows)
            ),
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
                "synthetic_gap_count": row["synthetic_gap_count"],
                "max_synthetic_gap_minutes": row["max_synthetic_gap_minutes"],
                "synthetic_rows_pct": row["synthetic_rows_pct"],
                "synthetic_gap_threshold_breached": row["synthetic_gap_threshold_breached"],
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
                "roll_window_rows_pct": row["roll_window_rows_pct"],
                "roll_window_threshold_breached": row["roll_window_threshold_breached"],
                "boundary_session_rows": row["boundary_session_rows"],
                "causal_valid_rows": row["causal_valid_rows"],
                "causal_invalid_rows": row["causal_invalid_rows"],
                "degraded_bar_rows": row["degraded_bar_rows"],
                "degraded_session_rows": row["degraded_session_rows"],
                "degraded_rows_pct": row["degraded_rows_pct"],
                "degraded_threshold_breached": row["degraded_threshold_breached"],
                "max_synthetic_rows_pct_threshold": row["max_synthetic_rows_pct_threshold"],
                "max_synthetic_gap_minutes_threshold": row[
                    "max_synthetic_gap_minutes_threshold"
                ],
                "max_degraded_rows_pct_threshold": row["max_degraded_rows_pct_threshold"],
                "max_roll_window_rows_pct_threshold": row[
                    "max_roll_window_rows_pct_threshold"
                ],
                "session_calendar_status": row["session_calendar_status"],
                "holiday_calendar_available": row["holiday_calendar_available"],
                "early_close_calendar_available": row["early_close_calendar_available"],
                "warning_count": row["warning_count"],
                "warnings": row["warnings"],
                "failure_count": row["failure_count"],
                "failures": row["failures"],
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
    parser.add_argument("--profile-config", default=str(DEFAULT_PROFILE_CONFIG))
    parser.add_argument("--session-config", default=str(DEFAULT_SESSION_CONFIG))
    parser.add_argument("--allow-hardcoded-calendar", action="store_true")
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
    profile_config_path = Path(args.profile_config)
    session_config_path = Path(args.session_config)
    config = load_causal_base_config(profile_config_path)
    inputs = resolve_profile_inputs(args.profile, raw_root, profile_config_path)

    results: list[ValidationResult] = []
    for market, year, input_path in inputs:
        output_path = output_root / market / f"{year}.parquet"
        result = process_file(
            input_path,
            output_path,
            profile=args.profile,
            roll_window_bars=args.roll_window_bars,
            max_synthetic_gap_minutes=config.max_synthetic_gap_minutes,
            profile_config_path=profile_config_path,
            session_config_path=session_config_path,
            allow_hardcoded_calendar=args.allow_hardcoded_calendar,
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
