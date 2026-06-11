#!/usr/bin/env python3
"""Build Phase 3 target/label parquet files from causal 1-minute bars."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


DEFAULT_PROFILE = "all_causal"
DISCOVERY_PROFILES = {"all_causal", "all_causal_data", "all_raw", "all_raw_data"}
DEFAULT_PROFILE_CONFIG = Path("configs/alpha_tiered.yaml")
STATIC_PROFILE_MARKETS = {
    "tier_1_core": ["CL", "ES", "ZN"],
}
STATIC_PROFILE_YEARS = {
    "tier_1_core": [2023, 2024, 2025],
}

ENTRY_OFFSET_BARS = 1
EXIT_OFFSET_BARS = 16
REGIME_OFFSET_BARS = 31
ATR_LOOKBACK_BARS = 60
LABEL_SEMANTICS_ID = "phase3_labels_v1_next_1m_open_to_15m_open"

REQUIRED_INPUT_COLUMNS = [
    "ts",
    "market",
    "year",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "causal_valid",
    "session_segment_id",
    "is_synthetic",
    "valid_ohlcv",
    "boundary_session_flag",
    "roll_window_flag",
]

GENERIC_LABEL_COLUMNS = [
    "target_entry_ts",
    "target_exit_ts",
    "target_entry_price",
    "target_exit_price",
    "target_horizon_bars",
    "target_ret_15m",
    "target_ret_ticks_15m",
    "target_gross_dollars_15m",
    "target_estimated_cost_ticks",
    "target_estimated_cost_dollars",
    "target_net_ticks_after_est_cost",
    "target_net_dollars_after_est_cost",
    "target_sign_15m",
    "target_sign_with_deadzone",
    "target_tradeable_after_cost",
    "target_valid",
    "target_invalid_reason",
]

REGIME_LABEL_COLUMNS = [
    "mae_ticks_15m",
    "mfe_ticks_15m",
    "fade_long_success_15m",
    "fade_short_success_15m",
    "trend_danger_up_30m",
    "trend_danger_down_30m",
    "revert_to_vwap_30m",
    "revert_to_session_mid_30m",
]

LABEL_PROVENANCE_COLUMNS = [
    "label_semantics",
    "cost_source",
    "cost_provisional",
]

LABEL_COLUMNS = GENERIC_LABEL_COLUMNS + REGIME_LABEL_COLUMNS + LABEL_PROVENANCE_COLUMNS

DEFAULT_MARKET_CONFIGS = {
    "CL": {
        "tick_size": 0.01,
        "tick_value": 10.0,
        "point_value": 1000.0,
        "min_profit_ticks": 2.0,
        "min_stop_ticks": 4.0,
        "estimated_cost_ticks": 2.0,
    },
    "ES": {
        "tick_size": 0.25,
        "tick_value": 12.5,
        "point_value": 50.0,
        "min_profit_ticks": 2.0,
        "min_stop_ticks": 4.0,
        "estimated_cost_ticks": 2.0,
    },
    "ZN": {
        "tick_size": 1.0 / 64.0,
        "tick_value": 15.625,
        "point_value": 1000.0,
        "min_profit_ticks": 2.0,
        "min_stop_ticks": 4.0,
        "estimated_cost_ticks": 2.0,
    },
}

UNKNOWN_MARKET_DEFAULT = {
    "tick_size": 0.01,
    "tick_value": 10.0,
    "point_value": 1000.0,
    "min_profit_ticks": 2.0,
    "min_stop_ticks": 4.0,
    "estimated_cost_ticks": 2.0,
}


@dataclass
class MarketConfig:
    market: str
    tick_size: float
    tick_value: float
    point_value: float
    min_profit_ticks: float
    min_stop_ticks: float
    estimated_cost_ticks: float
    estimated_cost_dollars: float
    source: str
    cost_source: str
    provisional: bool
    defaults_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "market": self.market,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "point_value": self.point_value,
            "min_profit_ticks": self.min_profit_ticks,
            "min_stop_ticks": self.min_stop_ticks,
            "estimated_cost_ticks": self.estimated_cost_ticks,
            "estimated_cost_dollars": self.estimated_cost_dollars,
            "source": self.source,
            "cost_source": self.cost_source,
            "provisional": self.provisional,
            "defaults_used": self.defaults_used,
        }


@dataclass
class LabelResult:
    profile: str
    market: str
    year: int
    input_path: str
    output_path: str
    input_rows: int = 0
    output_rows: int = 0
    target_valid_rows: int = 0
    target_invalid_rows: int = 0
    invalid_reason_counts: dict[str, int] = field(default_factory=dict)
    roll_detection_available: bool = False
    roll_detection_available_rows: int = 0
    roll_detection_unavailable_rows: int = 0
    roll_protection_unavailable: bool = False
    config: dict[str, object] = field(default_factory=dict)
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
        return data


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_optional_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def config_hash(paths: Iterable[Path]) -> str:
    payload = {
        relative_path(path): hash_optional_file(path)
        for path in paths
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unknown"
    return commit


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def discover_inputs(input_root: Path) -> list[tuple[str, int, Path]]:
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    inputs: list[tuple[str, int, Path]] = []
    for market_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for parquet_path in sorted(market_dir.glob("*.parquet")):
            if parquet_path.stem.isdigit():
                inputs.append((market_dir.name, int(parquet_path.stem), parquet_path))

    if not inputs:
        raise SystemExit(f"No causal year parquet files found under {input_root}")
    return inputs


def _read_yaml(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        return {}
    return payload


def load_profile_map(
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
) -> tuple[dict[str, list[str]], dict[str, list[int]], dict[str, str], set[str]]:
    markets = {key: value[:] for key, value in STATIC_PROFILE_MARKETS.items()}
    years = {key: value[:] for key, value in STATIC_PROFILE_YEARS.items()}
    aliases: dict[str, str] = {}
    discovery = set(DISCOVERY_PROFILES)

    payload = _read_yaml(profile_config_path)
    profiles = payload.get("profiles", {})
    if isinstance(profiles, Mapping):
        for profile_name, profile in profiles.items():
            if not isinstance(profile_name, str) or not isinstance(profile, Mapping):
                continue
            if bool(profile.get("discovery", False)):
                discovery.add(profile_name)
                continue
            profile_markets = profile.get("markets", [])
            profile_years = profile.get("years", [])
            if isinstance(profile_markets, list) and isinstance(profile_years, list):
                markets[profile_name] = [str(item) for item in profile_markets]
                years[profile_name] = [int(item) for item in profile_years]

    raw_aliases = payload.get("aliases", {})
    if isinstance(raw_aliases, Mapping):
        aliases = {str(key): str(value) for key, value in raw_aliases.items()}

    return markets, years, aliases, discovery


def resolve_profile_name(profile: str, aliases: Mapping[str, str]) -> str:
    seen: set[str] = set()
    resolved = profile
    while resolved in aliases:
        if resolved in seen:
            raise SystemExit(f"Profile alias cycle detected at {resolved!r}")
        seen.add(resolved)
        resolved = aliases[resolved]
    return resolved


def resolve_profile_inputs(
    profile: str,
    input_root: Path,
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
) -> list[tuple[str, int, Path]]:
    profile_markets, profile_years, aliases, discovery_profiles = load_profile_map(
        profile_config_path
    )
    resolved_profile = resolve_profile_name(profile, aliases)

    if resolved_profile in discovery_profiles:
        return discover_inputs(input_root)

    if resolved_profile not in profile_markets:
        known = ", ".join(sorted([*profile_markets, *discovery_profiles]))
        raise SystemExit(f"Unknown profile {profile!r}. Known profiles: {known}")

    return [
        (market, year, input_root / market / f"{year}.parquet")
        for market in profile_markets[resolved_profile]
        for year in profile_years[resolved_profile]
    ]


def infer_market_year(path: Path) -> tuple[str, int]:
    try:
        return path.parent.name, int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Cannot infer market/year from {path}") from exc


def _market_config_blob(raw: Mapping[str, object], market: str) -> Mapping[str, object]:
    for key in ("markets", "market_configs", "contracts"):
        nested = raw.get(key)
        if isinstance(nested, Mapping) and isinstance(nested.get(market), Mapping):
            return nested[market]  # type: ignore[return-value]
    if isinstance(raw.get(market), Mapping):
        return raw[market]  # type: ignore[return-value]
    return {}


def _float_field(
    data: Mapping[str, object],
    defaults: Mapping[str, float],
    field_name: str,
    defaults_used: list[str],
) -> float:
    value = data.get(field_name)
    if value is None:
        defaults_used.append(field_name)
        return float(defaults[field_name])
    return float(value)


def _cost_ticks(
    data: Mapping[str, object],
    tick_value: float,
    default_cost_ticks: float,
    defaults_used: list[str],
) -> float:
    for field_name in (
        "estimated_cost_ticks",
        "target_estimated_cost_ticks",
        "round_trip_cost_ticks",
        "round_turn_cost_ticks",
        "cost_ticks",
        "total_cost_ticks",
    ):
        if data.get(field_name) is not None:
            return float(data[field_name])

    for field_name in (
        "estimated_cost_dollars",
        "target_estimated_cost_dollars",
        "round_trip_cost_dollars",
        "round_turn_cost_dollars",
        "cost_dollars",
        "total_cost_dollars",
    ):
        if data.get(field_name) is not None:
            return float(data[field_name]) / tick_value

    slippage = float(data.get("slippage_ticks_per_side", 0.0) or 0.0)
    commission = float(
        data.get("commission_per_side_dollars", data.get("commission_per_contract_dollars", 0.0))
        or 0.0
    )
    fees = float(data.get("fees_per_side_dollars", 0.0) or 0.0)
    if slippage or commission or fees:
        return (2.0 * slippage) + (2.0 * (commission + fees) / tick_value)

    defaults_used.append("estimated_cost_ticks")
    return float(default_cost_ticks)


def load_market_config(market: str, costs_config: Path) -> MarketConfig:
    base_defaults = DEFAULT_MARKET_CONFIGS.get(market, UNKNOWN_MARKET_DEFAULT)
    data: Mapping[str, object] = {}
    source = "embedded_defaults"
    defaults_used: list[str] = []

    if costs_config.exists():
        raw = yaml.safe_load(costs_config.read_text(encoding="utf-8")) or {}
        if isinstance(raw, Mapping):
            data = _market_config_blob(raw, market)
            source = relative_path(costs_config)
            if not data:
                defaults_used.append("market_cost_missing")
        else:
            defaults_used.append("invalid_costs_config_shape")
    else:
        defaults_used.append("costs_config_missing")

    tick_size = _float_field(data, base_defaults, "tick_size", defaults_used)
    tick_value = _float_field(data, base_defaults, "tick_value", defaults_used)
    point_value = _float_field(data, base_defaults, "point_value", defaults_used)
    min_profit_ticks = _float_field(data, base_defaults, "min_profit_ticks", defaults_used)
    min_stop_ticks = _float_field(data, base_defaults, "min_stop_ticks", defaults_used)
    estimated_cost_ticks = _cost_ticks(
        data,
        tick_value,
        float(base_defaults["estimated_cost_ticks"]),
        defaults_used,
    )

    return MarketConfig(
        market=market,
        tick_size=tick_size,
        tick_value=tick_value,
        point_value=point_value,
        min_profit_ticks=min_profit_ticks,
        min_stop_ticks=min_stop_ticks,
        estimated_cost_ticks=estimated_cost_ticks,
        estimated_cost_dollars=estimated_cost_ticks * tick_value,
        source=source,
        cost_source=str(data.get("cost_source", source)),
        provisional=bool(data.get("provisional", bool(defaults_used))),
        defaults_used=sorted(set(defaults_used)),
    )


def _as_bool(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[column].fillna(default).astype(bool)


def _future_path_checks(df: pd.DataFrame, horizon_offset: int) -> dict[str, pd.Series]:
    idx = df.index
    current_segment = df["session_segment_id"].astype("string")
    synthetic = pd.Series(False, index=idx)
    invalid_ohlcv = pd.Series(False, index=idx)
    boundary = pd.Series(False, index=idx)
    roll = pd.Series(False, index=idx)
    segment_cross = pd.Series(False, index=idx)

    roll_boundary = _as_bool(df, "roll_boundary_flag")
    for offset in range(0, horizon_offset + 1):
        synthetic |= _as_bool(df, "is_synthetic").shift(-offset, fill_value=False)
        invalid_ohlcv |= ~_as_bool(df, "valid_ohlcv", default=True).shift(
            -offset, fill_value=True
        )
        boundary |= _as_bool(df, "boundary_session_flag").shift(-offset, fill_value=False)
        roll |= _as_bool(df, "roll_window_flag").shift(-offset, fill_value=False)
        roll |= roll_boundary.shift(-offset, fill_value=False)
        if offset == 0:
            continue
        shifted_segment = df["session_segment_id"].astype("string").shift(-offset)
        segment_cross |= shifted_segment.ne(current_segment).fillna(True)

    return {
        "synthetic": synthetic.astype(bool),
        "invalid_ohlcv": invalid_ohlcv.astype(bool),
        "boundary": boundary.astype(bool),
        "roll": roll.astype(bool),
        "segment_cross": segment_cross.astype(bool),
    }


def _true_range_ticks(df: pd.DataFrame, tick_size: float) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(ATR_LOOKBACK_BARS, min_periods=1).mean() / tick_size


def _future_extreme(df: pd.DataFrame, column: str, horizon_offset: int, op: str) -> pd.Series:
    shifted = [
        pd.to_numeric(df[column], errors="coerce").shift(-offset)
        for offset in range(1, horizon_offset + 1)
    ]
    frame = pd.concat(shifted, axis=1)
    if op == "max":
        return frame.max(axis=1)
    if op == "min":
        return frame.min(axis=1)
    raise ValueError(f"Unknown future extreme op: {op}")


def _first_hit(
    df: pd.DataFrame,
    entry_price: pd.Series,
    threshold_ticks: pd.Series,
    tick_size: float,
    *,
    side: str,
    kind: str,
) -> np.ndarray:
    first = np.full(len(df), np.inf)
    for offset in range(1, EXIT_OFFSET_BARS + 1):
        high = pd.to_numeric(df["high"], errors="coerce").shift(-offset)
        low = pd.to_numeric(df["low"], errors="coerce").shift(-offset)
        if side == "long" and kind == "profit":
            hit = high >= entry_price + threshold_ticks * tick_size
        elif side == "long" and kind == "adverse":
            hit = low <= entry_price - threshold_ticks * tick_size
        elif side == "short" and kind == "profit":
            hit = low <= entry_price - threshold_ticks * tick_size
        elif side == "short" and kind == "adverse":
            hit = high >= entry_price + threshold_ticks * tick_size
        else:
            raise ValueError(f"Unsupported hit type: {side} {kind}")
        mask = np.isinf(first) & hit.fillna(False).to_numpy()
        first[mask] = float(offset)
    return first


def _past_session_levels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    segment = df["session_segment_id"].astype("string")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0

    pv = (typical * volume).groupby(segment, dropna=False).cumsum()
    cum_volume = volume.groupby(segment, dropna=False).cumsum()
    vwap = pv / cum_volume.replace(0.0, np.nan)
    session_mid = (
        high.groupby(segment, dropna=False).cummax()
        + low.groupby(segment, dropna=False).cummin()
    ) / 2.0
    return vwap, session_mid


def _price_valid(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & (numeric > 0)


def add_labels(df: pd.DataFrame, config: MarketConfig) -> pd.DataFrame:
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True).copy()
    tick_size = config.tick_size
    tick_value = config.tick_value

    entry_price = pd.to_numeric(df["open"], errors="coerce").shift(-ENTRY_OFFSET_BARS)
    exit_price = pd.to_numeric(df["open"], errors="coerce").shift(-EXIT_OFFSET_BARS)
    entry_ts = pd.to_datetime(df["ts"], utc=True, errors="coerce").shift(-ENTRY_OFFSET_BARS)
    exit_ts = pd.to_datetime(df["ts"], utc=True, errors="coerce").shift(-EXIT_OFFSET_BARS)

    atr_ref_ticks = _true_range_ticks(df, tick_size)
    profit_threshold_ticks = np.maximum(config.min_profit_ticks, 0.50 * atr_ref_ticks)
    adverse_threshold_ticks = np.maximum(config.min_stop_ticks, 1.00 * atr_ref_ticks)
    trend_adverse_threshold_ticks = np.maximum(config.min_stop_ticks, 1.50 * atr_ref_ticks)

    checks_15m = _future_path_checks(df, EXIT_OFFSET_BARS)
    checks_30m = _future_path_checks(df, REGIME_OFFSET_BARS)

    causal_valid = _as_bool(df, "causal_valid")
    entry_missing = entry_ts.isna()
    exit_missing = exit_ts.isna()
    entry_exit_invalid = ~_price_valid(entry_price) | ~_price_valid(exit_price)

    target_valid = (
        causal_valid
        & ~entry_missing
        & ~exit_missing
        & ~checks_15m["segment_cross"]
        & ~checks_15m["synthetic"]
        & ~checks_15m["invalid_ohlcv"]
        & ~checks_15m["boundary"]
        & ~checks_15m["roll"]
        & ~entry_exit_invalid
    )

    invalid_reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason_masks = [
        ("current_causal_valid_false", ~causal_valid),
        ("entry_missing", entry_missing),
        ("exit_missing", exit_missing),
        ("session_segment_cross", checks_15m["segment_cross"]),
        ("synthetic_path", checks_15m["synthetic"]),
        ("invalid_ohlcv_path", checks_15m["invalid_ohlcv"]),
        ("boundary_session_path", checks_15m["boundary"]),
        ("roll_path", checks_15m["roll"]),
        ("entry_exit_price_invalid", entry_exit_invalid),
    ]
    for reason, mask in reason_masks:
        invalid_reason = invalid_reason.mask(invalid_reason.isna() & ~target_valid & mask, reason)

    gross_ticks = (exit_price - entry_price) / tick_size
    gross_dollars = gross_ticks * tick_value
    net_magnitude = (gross_ticks.abs() - config.estimated_cost_ticks).clip(lower=0.0)
    net_ticks = np.sign(gross_ticks) * net_magnitude
    net_dollars = net_ticks * tick_value
    sign = np.sign(gross_ticks).fillna(0).astype("int64")
    deadzone_ticks = config.estimated_cost_ticks + config.min_profit_ticks
    sign_deadzone = sign.mask(gross_ticks.abs() <= deadzone_ticks, 0).astype("int64")

    future_high_15m = _future_extreme(df, "high", EXIT_OFFSET_BARS, "max")
    future_low_15m = _future_extreme(df, "low", EXIT_OFFSET_BARS, "min")
    mfe_ticks = (future_high_15m - entry_price) / tick_size
    mae_ticks = (future_low_15m - entry_price) / tick_size

    long_profit_first = _first_hit(
        df, entry_price, profit_threshold_ticks, tick_size, side="long", kind="profit"
    )
    long_adverse_first = _first_hit(
        df, entry_price, adverse_threshold_ticks, tick_size, side="long", kind="adverse"
    )
    short_profit_first = _first_hit(
        df, entry_price, profit_threshold_ticks, tick_size, side="short", kind="profit"
    )
    short_adverse_first = _first_hit(
        df, entry_price, adverse_threshold_ticks, tick_size, side="short", kind="adverse"
    )

    fade_long_success = (
        target_valid.to_numpy()
        & np.isfinite(long_profit_first)
        & (long_profit_first < long_adverse_first)
    )
    fade_short_success = (
        target_valid.to_numpy()
        & np.isfinite(short_profit_first)
        & (short_profit_first < short_adverse_first)
    )

    entry_30m_valid = ~entry_missing & _price_valid(entry_price)
    valid_30m = (
        causal_valid
        & entry_30m_valid
        & ~checks_30m["segment_cross"]
        & ~checks_30m["synthetic"]
        & ~checks_30m["invalid_ohlcv"]
        & ~checks_30m["boundary"]
        & ~checks_30m["roll"]
    )
    future_high_30m = _future_extreme(df, "high", REGIME_OFFSET_BARS, "max")
    future_low_30m = _future_extreme(df, "low", REGIME_OFFSET_BARS, "min")
    trend_danger_up = valid_30m & (
        ((future_high_30m - entry_price) / tick_size) >= trend_adverse_threshold_ticks
    )
    trend_danger_down = valid_30m & (
        ((entry_price - future_low_30m) / tick_size) >= trend_adverse_threshold_ticks
    )

    vwap, session_mid = _past_session_levels(df)
    revert_to_vwap = valid_30m & (
        ((entry_price >= vwap) & (future_low_30m <= vwap))
        | ((entry_price < vwap) & (future_high_30m >= vwap))
    )
    revert_to_session_mid = valid_30m & (
        ((entry_price >= session_mid) & (future_low_30m <= session_mid))
        | ((entry_price < session_mid) & (future_high_30m >= session_mid))
    )

    df["target_entry_ts"] = entry_ts.where(target_valid)
    df["target_exit_ts"] = exit_ts.where(target_valid)
    df["target_entry_price"] = entry_price.where(target_valid)
    df["target_exit_price"] = exit_price.where(target_valid)
    df["target_horizon_bars"] = pd.Series(EXIT_OFFSET_BARS - ENTRY_OFFSET_BARS, index=df.index).where(
        target_valid
    )
    df["target_ret_15m"] = (exit_price / entry_price - 1.0).where(target_valid)
    df["target_ret_ticks_15m"] = gross_ticks.where(target_valid)
    df["target_gross_dollars_15m"] = gross_dollars.where(target_valid)
    df["target_estimated_cost_ticks"] = pd.Series(
        config.estimated_cost_ticks, index=df.index
    ).where(target_valid)
    df["target_estimated_cost_dollars"] = pd.Series(
        config.estimated_cost_dollars, index=df.index
    ).where(target_valid)
    df["target_net_ticks_after_est_cost"] = net_ticks.where(target_valid)
    df["target_net_dollars_after_est_cost"] = net_dollars.where(target_valid)
    df["target_sign_15m"] = sign.where(target_valid, 0).astype("int64")
    df["target_sign_with_deadzone"] = sign_deadzone.where(target_valid, 0).astype("int64")
    df["target_tradeable_after_cost"] = (gross_ticks.abs() > config.estimated_cost_ticks).where(
        target_valid, False
    )
    df["target_valid"] = target_valid.astype(bool)
    df["target_invalid_reason"] = invalid_reason

    df["mae_ticks_15m"] = mae_ticks.where(target_valid)
    df["mfe_ticks_15m"] = mfe_ticks.where(target_valid)
    df["fade_long_success_15m"] = pd.Series(fade_long_success, index=df.index)
    df["fade_short_success_15m"] = pd.Series(fade_short_success, index=df.index)
    df["trend_danger_up_30m"] = trend_danger_up.fillna(False).astype(bool)
    df["trend_danger_down_30m"] = trend_danger_down.fillna(False).astype(bool)
    df["revert_to_vwap_30m"] = revert_to_vwap.fillna(False).astype(bool)
    df["revert_to_session_mid_30m"] = revert_to_session_mid.fillna(False).astype(bool)
    df["label_semantics"] = LABEL_SEMANTICS_ID
    df["cost_source"] = config.cost_source
    df["cost_provisional"] = bool(config.provisional)

    return df


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    profile: str,
    costs_config: Path,
) -> LabelResult:
    market, year = infer_market_year(input_path)
    config = load_market_config(market, costs_config)
    result = LabelResult(
        profile=profile,
        market=market,
        year=year,
        input_path=relative_path(input_path),
        output_path=relative_path(output_path),
        config=config.to_dict(),
    )

    if config.defaults_used:
        result.warnings.append("market config defaults used: " + ",".join(config.defaults_used))
    if (
        "costs_config_missing" in config.defaults_used
        or "market_cost_missing" in config.defaults_used
        or "estimated_cost_ticks" in config.defaults_used
    ):
        result.warnings.append("placeholder costs used")
    if config.provisional:
        result.warnings.append(f"provisional costs used: {config.cost_source}")

    if not input_path.exists():
        result.failures.append("input file missing")
        return result

    df = pd.read_parquet(input_path)
    result.input_rows = len(df)
    if df.empty:
        result.failures.append("empty file")
        return result

    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        result.failures.append("missing required input columns: " + ",".join(missing))
        return result

    roll_detection_available = _as_bool(df, "roll_detection_available", default=False)
    result.roll_detection_available_rows = int(roll_detection_available.sum())
    result.roll_detection_unavailable_rows = int((~roll_detection_available).sum())
    result.roll_detection_available = result.roll_detection_unavailable_rows == 0
    if result.roll_detection_unavailable_rows:
        result.roll_protection_unavailable = True
        result.warnings.append(
            "roll protection unavailable for "
            f"{result.roll_detection_unavailable_rows} rows: roll_detection_available false"
        )

    output = add_labels(df, config)
    result.output_rows = len(output)
    result.target_valid_rows = int(output["target_valid"].sum())
    result.target_invalid_rows = result.output_rows - result.target_valid_rows
    counts = output["target_invalid_reason"].dropna().value_counts().sort_index()
    result.invalid_reason_counts = {str(k): int(v) for k, v in counts.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return result


def _aggregate_invalid_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for row in rows:
        counts = row.get("invalid_reason_counts", {})
        if not isinstance(counts, Mapping):
            continue
        for reason, count in counts.items():
            aggregate[str(reason)] = aggregate.get(str(reason), 0) + int(count)
    return dict(sorted(aggregate.items()))


def write_reports(
    results: Iterable[LabelResult],
    reports_root: Path,
    profile: str,
    *,
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    costs_config_path: Path = Path("configs/costs.yaml"),
) -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    run_failures = [
        {
            "market": row["market"],
            "year": row["year"],
            "failures": row["failures"],
        }
        for row in rows
        if row["failures"]
    ]
    script_path = Path(__file__).resolve()
    provenance = {
        "generated_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "script_path": relative_path(script_path),
        "script_hash": sha256_file(script_path),
        "config_hash": config_hash([profile_config_path, costs_config_path]),
        "input_file_hashes": {
            str(row["input_path"]): hash_optional_file(Path(str(row["input_path"])))
            for row in rows
        },
        "output_file_hashes": {
            str(row["output_path"]): hash_optional_file(Path(str(row["output_path"])))
            for row in rows
        },
        "profile": profile,
        "markets": sorted({str(row["market"]) for row in rows}),
        "years": sorted({int(row["year"]) for row in rows}),
        "warning_count": int(sum(row["warning_count"] for row in rows)),
        "failure_count": int(sum(row["failure_count"] for row in rows)),
        "failures": run_failures,
    }
    status = (
        "FAIL"
        if any(row["status"] == "FAIL" for row in rows)
        else "WARN"
        if any(row["status"] == "WARN" for row in rows)
        else "PASS"
    )
    summary = {
        "file_count": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "warn_count": sum(row["status"] == "WARN" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "input_rows": int(sum(row["input_rows"] for row in rows)),
        "output_rows": int(sum(row["output_rows"] for row in rows)),
        "target_valid_rows": int(sum(row["target_valid_rows"] for row in rows)),
        "target_invalid_rows": int(sum(row["target_invalid_rows"] for row in rows)),
        "invalid_reason_counts": _aggregate_invalid_counts(rows),
        "roll_protection_unavailable_files": int(
            sum(bool(row["roll_protection_unavailable"]) for row in rows)
        ),
        "roll_detection_available_rows": int(
            sum(row["roll_detection_available_rows"] for row in rows)
        ),
        "roll_detection_unavailable_rows": int(
            sum(row["roll_detection_unavailable_rows"] for row in rows)
        ),
    }
    label_semantics = {
        "target_ret_ticks_15m": "signed directional price move; positive means price moved up, negative means price moved down",
        "target_net_ticks_after_est_cost": "signed directional move beyond estimated cost; costs reduce magnitude and never flip sign",
        "target_tradeable_after_cost": "absolute move exceeds estimated cost; not guaranteed profitability",
    }

    report = {
        **provenance,
        "stage": "labels",
        "status": status,
        "label_semantics": label_semantics,
        "files": rows,
        "summary": summary,
    }
    manifest = {
        **provenance,
        "stage": "labels",
        "status": status,
        "label_semantics": label_semantics,
        "outputs": [
            {
                "market": row["market"],
                "year": row["year"],
                "input_path": row["input_path"],
                "output_path": row["output_path"],
                "input_rows": row["input_rows"],
                "output_rows": row["output_rows"],
                "target_valid_rows": row["target_valid_rows"],
                "target_invalid_rows": row["target_invalid_rows"],
                "invalid_reason_counts": row["invalid_reason_counts"],
                "roll_detection_available": row["roll_detection_available"],
                "roll_detection_available_rows": row["roll_detection_available_rows"],
                "roll_detection_unavailable_rows": row["roll_detection_unavailable_rows"],
                "roll_protection_unavailable": row["roll_protection_unavailable"],
                "config": row["config"],
                "warnings": row["warnings"],
                "failures": row["failures"],
                "status": row["status"],
                "warning_count": row["warning_count"],
                "failure_count": row["failure_count"],
            }
            for row in rows
        ],
        "summary": summary,
    }

    (reports_root / "label_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (reports_root / "label_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--input-root", default="data/causally_gated_normalized")
    parser.add_argument("--output-root", default="data/labeled")
    parser.add_argument("--reports-root", default="reports/labels")
    parser.add_argument("--costs-config", default="configs/costs.yaml")
    parser.add_argument("--profile-config", default=str(DEFAULT_PROFILE_CONFIG))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    costs_config = Path(args.costs_config)
    profile_config = Path(args.profile_config)
    inputs = resolve_profile_inputs(args.profile, input_root, profile_config)

    results: list[LabelResult] = []
    for market, year, input_path in inputs:
        output_path = output_root / market / f"{year}.parquet"
        result = process_file(
            input_path,
            output_path,
            profile=args.profile,
            costs_config=costs_config,
        )
        results.append(result)
        print(
            f"{result.status} {market} {year}: rows={result.input_rows} "
            f"valid={result.target_valid_rows} invalid={result.target_invalid_rows} "
            f"warnings={len(result.warnings)} failures={len(result.failures)}"
        )

    write_reports(
        results,
        reports_root,
        args.profile,
        profile_config_path=profile_config,
        costs_config_path=costs_config,
    )
    return 1 if any(result.failures for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
