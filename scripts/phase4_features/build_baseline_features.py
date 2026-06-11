#!/usr/bin/env python3
"""Build Phase 4 baseline + L0 regime feature matrices from labeled 1-minute bars."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


DEFAULT_PROFILE = "tier_1_core"
DEFAULT_PROFILE_CONFIG = Path("configs/alpha_tiered.yaml")
DEFAULT_INPUT_ROOT = Path("data/labeled")
DEFAULT_OUTPUT_ROOT = Path("data/feature_matrices/baseline")
DEFAULT_REPORTS_ROOT = Path("reports/features_baseline")
DEFAULT_COSTS_CONFIG = Path("configs/costs.yaml")
DISCOVERY_PROFILES = {"all_labeled", "all_labeled_data", "all_raw", "all_raw_data"}
STATIC_PROFILE_MARKETS = {"tier_1_core": ["CL", "ES", "ZN"]}
STATIC_PROFILE_YEARS = {"tier_1_core": [2023, 2024, 2025]}
TIER1_MARKETS = ("CL", "ES", "ZN")
EPS = 1e-12
PHASE3_LABEL_SEMANTICS_ID = "phase3_labels_v1_next_1m_open_to_15m_open"

FEATURE_FAMILIES: dict[str, list[str]] = {
    "baseline_ohlcv": [
        "feature_ret_1",
        "feature_ret_5",
        "feature_ret_10",
        "feature_ret_20",
        "feature_log_ret_1",
        "feature_range_norm",
        "feature_true_range",
        "feature_ewma_vol_20",
        "feature_volume_z_20",
        "feature_close_position_in_range",
        "feature_body_to_range",
        "feature_upper_wick_ratio",
        "feature_lower_wick_ratio",
        "feature_minutes_since_session_open",
        "feature_minutes_until_session_close",
        "feature_session_progress",
        "feature_minute_of_day_sin",
        "feature_minute_of_day_cos",
        "feature_day_of_week",
    ],
    "fade_safety_trend_danger": [
        "feature_efficiency_ratio_15",
        "feature_efficiency_ratio_30",
        "feature_efficiency_ratio_60",
        "feature_directional_bar_ratio_15",
        "feature_directional_bar_ratio_30",
        "feature_consecutive_up_bars",
        "feature_consecutive_down_bars",
        "feature_trend_persistence_30",
        "feature_signed_trend_persistence_30",
    ],
    "breakout_rejection": [
        "feature_prior_high_20_dist",
        "feature_prior_low_20_dist",
        "feature_breakout_above_20",
        "feature_breakout_below_20",
        "feature_failed_breakout_above_20",
        "feature_failed_breakout_below_20",
        "feature_close_back_inside_range_20",
        "feature_upper_wick_rejection",
        "feature_lower_wick_rejection",
    ],
    "range_chop": [
        "feature_realized_range_30",
        "feature_realized_range_60",
        "feature_range_compression_30_vs_120",
        "feature_chop_ratio_30",
        "feature_inside_bar_count_20",
        "feature_overlap_ratio_20",
    ],
    "session_structure": [
        "feature_session_open_dist",
        "feature_session_high_dist",
        "feature_session_low_dist",
        "feature_session_mid_dist",
        "feature_session_vwap_dist",
        "feature_session_range_percentile",
        "feature_opening_range_30_ready",
        "feature_opening_range_30_high_dist",
        "feature_opening_range_30_low_dist",
        "feature_opening_range_30_breakout_up",
        "feature_opening_range_30_breakout_down",
    ],
    "volatility_volume": [
        "feature_realized_vol_15",
        "feature_realized_vol_60",
        "feature_vol_expansion_15_vs_60",
        "feature_large_bar_count_30",
        "feature_shock_bar_flag",
        "feature_bars_since_shock",
        "feature_volume_z_60",
        "feature_volume_surge_with_range",
        "feature_volume_surge_without_progress",
        "feature_range_per_volume",
        "feature_volume_climax_flag",
        "feature_bars_since_volume_climax",
    ],
    "higher_timeframe_prior_session": [
        "feature_5m_ret_3",
        "feature_15m_ret_4",
        "feature_60m_trend_slope",
        "feature_daily_open_dist",
        "feature_prior_session_high_dist",
        "feature_prior_session_low_dist",
        "feature_prior_session_close_dist",
        "feature_prior_session_range_pct",
        "feature_overnight_gap_ticks",
    ],
    "time_buckets": [
        "feature_time_bucket_globex_open",
        "feature_time_bucket_europe",
        "feature_time_bucket_us_open",
        "feature_time_bucket_midday",
        "feature_time_bucket_power_hour",
        "feature_first_30m_flag",
        "feature_last_30m_flag",
    ],
    "tier1_intermarket": [
        "feature_rel_ret_vs_ES_15",
        "feature_rel_ret_vs_ZN_15",
        "feature_rel_ret_vs_CL_15",
        "feature_corr_vs_ES_60",
        "feature_corr_vs_ZN_60",
        "feature_corr_vs_CL_60",
        "feature_es_zn_divergence_30",
        "feature_cl_es_divergence_30",
    ],
    "effort_result": [
        "feature_effort_result_30",
        "feature_absorption_proxy_30",
        "feature_exhaustion_proxy_30",
        "feature_volume_per_tick_progress_30",
        "feature_range_without_close_progress_30",
    ],
    "trend_day_open_drive": [
        "feature_open_drive_up",
        "feature_open_drive_down",
        "feature_open_drive_strength_30",
        "feature_session_one_wayness",
        "feature_vwap_side_persistence",
        "feature_bars_above_vwap_30",
        "feature_bars_below_vwap_30",
        "feature_pullback_shallowness_30",
    ],
    "auction_acceptance": [
        "feature_session_range_extension_up",
        "feature_session_range_extension_down",
        "feature_session_acceptance_above_mid",
        "feature_session_acceptance_below_mid",
        "feature_failed_retest_session_high",
        "feature_failed_retest_session_low",
    ],
    "shock_decay": [
        "feature_shock_direction",
        "feature_bars_since_up_shock",
        "feature_bars_since_down_shock",
        "feature_post_shock_retrace_pct",
        "feature_post_shock_continuation_pct",
        "feature_post_shock_range_decay",
    ],
    "tier1_cross_market_regime": [
        "feature_tier1_direction_agreement_15",
        "feature_tier1_return_dispersion_15",
        "feature_tier1_risk_on_score_30",
        "feature_es_zn_risk_regime_30",
        "feature_cl_es_macro_divergence_30",
    ],
}

FEATURE_COLS = [feature for features in FEATURE_FAMILIES.values() for feature in features]
FEATURE_TO_FAMILY = {
    feature: family for family, features in FEATURE_FAMILIES.items() for feature in features
}

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

FORBIDDEN_FEATURE_COLUMNS = {
    "ts",
    "market",
    "year",
    "session_id",
    "session_date",
    "session_segment_id",
    "raw_row_present",
    "is_synthetic",
    "causal_valid",
    "valid_ohlcv",
    "inside_session",
    "boundary_session_flag",
    "feature_input_valid",
    "feature_row_valid",
    "training_row_valid",
    "target_valid",
    "target_invalid_reason",
    *REGIME_LABEL_COLUMNS,
    "rtype",
    "publisher_id",
    "instrument_id",
    "symbol",
    "source_path",
    "source_file_hash",
    "source_row_number",
    "raw_schema_variant",
    "timestamp_source",
    "metadata_available",
    "roll_detection_available",
    "roll_detection_source",
    "roll_policy_status",
    "synthetic_gap_id",
    "synthetic_gap_size_minutes",
    "synthetic_gap_reason",
    "data_quality_status",
    "data_quality_degraded",
    "session_data_quality_degraded",
    "trainable_data_quality",
}

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
    "valid_ohlcv",
    "is_synthetic",
    "roll_window_flag",
    "boundary_session_flag",
    "session_segment_id",
    "target_valid",
]

REQUIRED_LABEL_CONTRACT_COLUMNS = [
    "label_semantics",
    "cost_source",
    "cost_provisional",
    "roll_detection_available",
    "target_ret_15m",
    "target_ret_ticks_15m",
    "target_net_ticks_after_est_cost",
    "target_tradeable_after_cost",
    "target_valid",
]

LABEL_NON_NULL_COLUMNS = [
    "label_semantics",
    "cost_source",
    "cost_provisional",
    "roll_detection_available",
    "target_valid",
]

TARGET_VALUE_COLUMNS = [
    "target_ret_15m",
    "target_ret_ticks_15m",
    "target_net_ticks_after_est_cost",
    "target_tradeable_after_cost",
]


@dataclass
class FeatureResult:
    profile: str
    market: str
    year: int
    input_path: str
    output_path: str
    input_rows: int = 0
    output_rows: int = 0
    feature_input_valid_rows: int = 0
    training_row_valid_rows: int = 0
    target_valid_rows: int = 0
    feature_count: int = 0
    nan_counts: dict[str, int] = field(default_factory=dict)
    nan_pct: dict[str, float] = field(default_factory=dict)
    intermarket_missing_pct: dict[str, float] = field(default_factory=dict)
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
        payload = self.__dict__.copy()
        payload["status"] = self.status
        payload["warning_count"] = len(self.warnings)
        payload["failure_count"] = len(self.failures)
        return payload


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hash_or_missing(path: Path) -> str:
    return file_sha256(path) if path.exists() else "missing"


def file_hash_map(paths: Iterable[Path]) -> dict[str, str]:
    return {relative_path(path): file_hash_or_missing(path) for path in paths}


def config_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(relative_path(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash_or_missing(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _read_yaml(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, Mapping) else {}


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


def discover_inputs(input_root: Path) -> list[tuple[str, int, Path]]:
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    inputs: list[tuple[str, int, Path]] = []
    for market_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for parquet_path in sorted(market_dir.glob("*.parquet")):
            if parquet_path.stem.isdigit():
                inputs.append((market_dir.name, int(parquet_path.stem), parquet_path))
    if not inputs:
        raise SystemExit(f"No labeled year parquet files found under {input_root}")
    return inputs


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


def load_tick_sizes(costs_config: Path) -> dict[str, float]:
    raw = _read_yaml(costs_config)
    markets = raw.get("markets", {})
    tick_sizes: dict[str, float] = {}
    if isinstance(markets, Mapping):
        for market, config in markets.items():
            if isinstance(config, Mapping) and config.get("tick_size") is not None:
                tick_sizes[str(market)] = float(config["tick_size"])
    return tick_sizes


def resolve_market_tick_size(costs_config: Path, market: str) -> tuple[float | None, str | None]:
    if not costs_config.exists():
        return None, f"missing costs config: {relative_path(costs_config)}"
    raw = _read_yaml(costs_config)
    markets = raw.get("markets", {})
    if not isinstance(markets, Mapping):
        return None, "invalid costs config: markets mapping missing"
    if market not in markets:
        return None, f"missing tick_size for market: {market}"
    config = markets[market]
    if not isinstance(config, Mapping) or config.get("tick_size") is None:
        return None, f"missing tick_size for market: {market}"
    try:
        tick_size = float(config["tick_size"])
    except (TypeError, ValueError):
        return None, f"invalid tick_size for market: {market}"
    if not math.isfinite(tick_size) or tick_size <= 0.0:
        return None, f"invalid tick_size for market: {market}"
    return tick_size, None


def validate_label_contract(df: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    missing = [col for col in REQUIRED_LABEL_CONTRACT_COLUMNS if col not in df.columns]
    if missing:
        failures.append("missing required Phase 3 label columns: " + ",".join(missing))
        return failures

    null_cols = [col for col in LABEL_NON_NULL_COLUMNS if df[col].isna().any()]
    if null_cols:
        failures.append("null required Phase 3 label columns: " + ",".join(null_cols))

    semantics = df["label_semantics"].astype("string").fillna("")
    if semantics.ne(PHASE3_LABEL_SEMANTICS_ID).any():
        failures.append(
            "noncanonical label_semantics: expected " + PHASE3_LABEL_SEMANTICS_ID
        )

    cost_source_blank = df["cost_source"].astype("string").fillna("").str.strip().eq("")
    if cost_source_blank.any():
        failures.append("blank cost_source in Phase 3 labels")

    if bool_col(df, "cost_provisional").any():
        failures.append("provisional Phase 3 costs are not allowed for features")

    if not bool_col(df, "roll_detection_available").all():
        failures.append("roll_detection_available must be true for every labeled row")

    target_valid = bool_col(df, "target_valid")
    target_null_cols = [
        col for col in TARGET_VALUE_COLUMNS if df.loc[target_valid, col].isna().any()
    ]
    if target_null_cols:
        failures.append(
            "null target values on target_valid rows: " + ",".join(target_null_cols)
        )
    return failures


def bool_col(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[column].fillna(default).astype(bool)


def num_col(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def safe_div(numerator: pd.Series, denominator: pd.Series | float, eps: float = EPS) -> pd.Series:
    denom = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index)
    return numerator / denom.where(denom.abs() > eps)


def rolling_sum(series: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    return series.groupby(segment, sort=False).transform(
        lambda x: x.rolling(window, min_periods=window).sum()
    )


def rolling_mean(series: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    return series.groupby(segment, sort=False).transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )


def rolling_std(series: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    return series.groupby(segment, sort=False).transform(
        lambda x: x.rolling(window, min_periods=window).std(ddof=0)
    )


def rolling_max(series: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    return series.groupby(segment, sort=False).transform(
        lambda x: x.rolling(window, min_periods=window).max()
    )


def rolling_min(series: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    return series.groupby(segment, sort=False).transform(
        lambda x: x.rolling(window, min_periods=window).min()
    )


def rolling_corr(left: pd.Series, right: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    corr = pd.Series(np.nan, index=left.index, dtype=float)
    for _, idx in left.groupby(segment, sort=False).groups.items():
        corr.loc[idx] = left.loc[idx].rolling(window, min_periods=window).corr(right.loc[idx])
    return corr


def lag(series: pd.Series, segment: pd.Series, periods: int) -> pd.Series:
    return series.groupby(segment, sort=False).shift(periods)


def ret_over(close: pd.Series, valid: pd.Series, segment: pd.Series, periods: int) -> pd.Series:
    safe_close = close.where(valid)
    prev = lag(safe_close, segment, periods)
    window_valid = valid_window_mask(valid, segment, periods + 1)
    return ((safe_close / prev) - 1.0).where(window_valid & prev.notna())


def valid_window_mask(valid: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    return rolling_sum(valid.astype(float), segment, window).eq(float(window))


def rolling_sum_full_valid(
    series: pd.Series,
    valid: pd.Series,
    segment: pd.Series,
    window: int,
) -> pd.Series:
    return rolling_sum(series.where(valid), segment, window).where(
        valid_window_mask(valid, segment, window)
    )


def rolling_mean_full_valid(
    series: pd.Series,
    valid: pd.Series,
    segment: pd.Series,
    window: int,
) -> pd.Series:
    return rolling_mean(series.where(valid), segment, window).where(
        valid_window_mask(valid, segment, window)
    )


def rolling_true_count_full_valid(
    condition: pd.Series,
    valid: pd.Series,
    segment: pd.Series,
    window: int,
) -> pd.Series:
    return rolling_sum_full_valid(condition.where(valid).astype(float), valid, segment, window)


def bars_since(flag: pd.Series, valid: pd.Series, segment: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=flag.index, dtype=float)
    for _, idx in flag.groupby(segment, sort=False).groups.items():
        count = math.nan
        values: list[float] = []
        for is_flag, is_valid in zip(flag.loc[idx].astype(bool), valid.loc[idx].astype(bool)):
            if not is_valid:
                values.append(math.nan)
                continue
            if is_flag:
                count = 0.0
            elif math.isnan(count):
                values.append(math.nan)
                continue
            else:
                count += 1.0
            values.append(count)
        output.loc[idx] = values
    return output


def consecutive_bars(condition: pd.Series, valid: pd.Series, segment: pd.Series) -> pd.Series:
    output = pd.Series(0.0, index=condition.index)
    for _, idx in condition.groupby(segment, sort=False).groups.items():
        count = 0.0
        values: list[float] = []
        for cond, is_valid in zip(condition.loc[idx].astype(bool), valid.loc[idx].astype(bool)):
            count = count + 1.0 if cond and is_valid else 0.0
            values.append(count)
        output.loc[idx] = values
    return output


def first_30_session_stats(
    df: pd.DataFrame,
    valid: pd.Series,
    tick_size: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ready = pd.Series(False, index=df.index)
    high = pd.Series(np.nan, index=df.index, dtype=float)
    low = pd.Series(np.nan, index=df.index, dtype=float)
    drive_up = pd.Series(False, index=df.index)
    drive_down = pd.Series(False, index=df.index)
    for _, idx in df.groupby("session_segment_id", sort=False).groups.items():
        group = df.loc[idx]
        if len(group) < 30:
            continue
        first_idx = list(idx)[:30]
        first_valid = bool(valid.loc[first_idx].all())
        if not first_valid:
            continue
        or_high = float(num_col(group.loc[first_idx], "high").max())
        or_low = float(num_col(group.loc[first_idx], "low").min())
        session_open = float(num_col(group.iloc[[0]], "open").iloc[0])
        close_30 = float(num_col(group.loc[[first_idx[-1]]], "close").iloc[0])
        session_range = max(or_high - or_low, tick_size)
        up = close_30 > session_open and abs(close_30 - session_open) / session_range >= 0.65
        down = close_30 < session_open and abs(close_30 - session_open) / session_range >= 0.65
        after_ready = list(idx)[29:]
        ready.loc[after_ready] = True
        high.loc[after_ready] = or_high
        low.loc[after_ready] = or_low
        drive_up.loc[after_ready] = up
        drive_down.loc[after_ready] = down
    return ready, high, low, drive_up, drive_down


def prior_session_maps(df: pd.DataFrame) -> dict[str, pd.Series]:
    stats = (
        df.groupby("session_segment_id", sort=False)
        .agg(
            first_ts=("ts", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .sort_values("first_ts")
    )
    stats["range"] = stats["high"] - stats["low"]
    prior = stats[["high", "low", "close", "range"]].shift(1)
    return {
        name: df["session_segment_id"].map(prior[name]).astype(float)
        for name in ["high", "low", "close", "range"]
    }


def compute_feature_input_valid(df: pd.DataFrame) -> pd.Series:
    required_not_null = df[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    return (
        bool_col(df, "causal_valid")
        & bool_col(df, "valid_ohlcv", default=True)
        & ~bool_col(df, "is_synthetic")
        & ~bool_col(df, "roll_window_flag")
        & ~bool_col(df, "boundary_session_flag")
        & required_not_null
    )


def add_base_market_features(df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts", kind="mergesort").reset_index(drop=True)
    segment = out["session_segment_id"].astype("string")
    valid = compute_feature_input_valid(out)
    out["feature_input_valid"] = valid
    out["feature_row_valid"] = valid
    out["training_row_valid"] = valid & bool_col(out, "target_valid")

    open_ = num_col(out, "open")
    high = num_col(out, "high")
    low = num_col(out, "low")
    close = num_col(out, "close")
    volume = num_col(out, "volume")
    safe_open = open_.where(valid)
    safe_high = high.where(valid)
    safe_low = low.where(valid)
    safe_close = close.where(valid)
    safe_volume = volume.where(valid)
    prev_close = lag(safe_close, segment, 1)
    true_range = pd.concat(
        [(safe_high - safe_low).abs(), (safe_high - prev_close).abs(), (safe_low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    for periods in (1, 5, 10, 20):
        out[f"feature_ret_{periods}"] = ret_over(close, valid, segment, periods)
    out["feature_log_ret_1"] = np.log1p(out["feature_ret_1"])
    out["feature_range_norm"] = safe_div(safe_high - safe_low, safe_close.abs())
    out["feature_true_range"] = true_range
    out["feature_ewma_vol_20"] = out["feature_ret_1"].groupby(segment, sort=False).transform(
        lambda x: x.ewm(span=20, min_periods=20, adjust=False).std()
    )
    vol_mean_20 = rolling_mean(safe_volume, segment, 20)
    vol_std_20 = rolling_std(safe_volume, segment, 20)
    out["feature_volume_z_20"] = safe_div(safe_volume - vol_mean_20, vol_std_20)
    bar_range = safe_high - safe_low
    out["feature_close_position_in_range"] = safe_div(safe_close - safe_low, bar_range)
    out["feature_body_to_range"] = safe_div((safe_close - safe_open).abs(), bar_range)
    out["feature_upper_wick_ratio"] = safe_div(safe_high - pd.concat([safe_open, safe_close], axis=1).max(axis=1), bar_range)
    out["feature_lower_wick_ratio"] = safe_div(pd.concat([safe_open, safe_close], axis=1).min(axis=1) - safe_low, bar_range)
    out["feature_minutes_since_session_open"] = num_col(out, "minutes_since_session_open").where(valid)
    out["feature_minutes_until_session_close"] = num_col(out, "minutes_until_session_close").where(valid)
    out["feature_session_progress"] = num_col(out, "session_progress").where(valid)
    minute = num_col(out, "minute_of_day")
    out["feature_minute_of_day_sin"] = np.sin(2.0 * np.pi * minute / 1440.0).where(valid)
    out["feature_minute_of_day_cos"] = np.cos(2.0 * np.pi * minute / 1440.0).where(valid)
    out["feature_day_of_week"] = num_col(out, "day_of_week").where(valid)

    one_bar_abs_move = (safe_close - lag(safe_close, segment, 1)).abs()
    for window in (15, 30, 60):
        move = (safe_close - lag(safe_close, segment, window)).abs()
        path = rolling_sum(one_bar_abs_move.where(valid), segment, window)
        out[f"feature_efficiency_ratio_{window}"] = safe_div(move, path)
    direction = np.sign(safe_close - prev_close).where(valid)
    for window in (15, 30):
        previous_direction = lag(direction, segment, 1)
        same_direction = direction.eq(previous_direction).where(previous_direction.notna())
        out[f"feature_directional_bar_ratio_{window}"] = rolling_mean_full_valid(
            same_direction.astype(float),
            valid,
            segment,
            window,
        )
    out["feature_consecutive_up_bars"] = consecutive_bars(safe_close.gt(prev_close), valid, segment)
    out["feature_consecutive_down_bars"] = consecutive_bars(safe_close.lt(prev_close), valid, segment)
    out["feature_trend_persistence_30"] = out["feature_efficiency_ratio_30"]
    out["feature_signed_trend_persistence_30"] = out["feature_efficiency_ratio_30"] * np.sign(
        safe_close - lag(safe_close, segment, 30)
    )

    prior_high_20 = rolling_max(safe_high, segment, 20).groupby(segment, sort=False).shift(1)
    prior_low_20 = rolling_min(safe_low, segment, 20).groupby(segment, sort=False).shift(1)
    prior_ready_20 = valid_window_mask(valid, segment, 20).groupby(segment, sort=False).shift(1).fillna(False)
    out["feature_prior_high_20_dist"] = ((safe_close - prior_high_20) / tick_size).where(prior_ready_20)
    out["feature_prior_low_20_dist"] = ((safe_close - prior_low_20) / tick_size).where(prior_ready_20)
    above = (safe_high > prior_high_20) & prior_ready_20 & valid
    below = (safe_low < prior_low_20) & prior_ready_20 & valid
    out["feature_breakout_above_20"] = above
    out["feature_breakout_below_20"] = below
    out["feature_failed_breakout_above_20"] = above & (safe_close <= prior_high_20)
    out["feature_failed_breakout_below_20"] = below & (safe_close >= prior_low_20)
    out["feature_close_back_inside_range_20"] = ((safe_close <= prior_high_20) & (safe_close >= prior_low_20) & prior_ready_20 & valid)
    out["feature_upper_wick_rejection"] = (out["feature_upper_wick_ratio"] > 0.5) & valid
    out["feature_lower_wick_rejection"] = (out["feature_lower_wick_ratio"] > 0.5) & valid

    for window in (30, 60):
        out[f"feature_realized_range_{window}"] = (rolling_max(safe_high, segment, window) - rolling_min(safe_low, segment, window)).where(valid_window_mask(valid, segment, window))
    range_30 = out["feature_realized_range_30"]
    range_120 = rolling_max(safe_high, segment, 120) - rolling_min(safe_low, segment, 120)
    out["feature_range_compression_30_vs_120"] = safe_div(range_30, range_120).where(valid_window_mask(valid, segment, 120))
    out["feature_chop_ratio_30"] = safe_div(rolling_sum(true_range, segment, 30), (safe_close - lag(safe_close, segment, 30)).abs())
    prev_high = lag(safe_high, segment, 1)
    prev_low = lag(safe_low, segment, 1)
    inside = ((safe_high <= prev_high) & (safe_low >= prev_low)).where(
        prev_high.notna() & prev_low.notna()
    )
    out["feature_inside_bar_count_20"] = rolling_true_count_full_valid(
        inside,
        valid,
        segment,
        20,
    )
    overlap = (pd.concat([safe_high, prev_high], axis=1).min(axis=1) - pd.concat([safe_low, prev_low], axis=1).max(axis=1)).clip(lower=0)
    union = (pd.concat([safe_high, prev_high], axis=1).max(axis=1) - pd.concat([safe_low, prev_low], axis=1).min(axis=1))
    out["feature_overlap_ratio_20"] = rolling_mean(safe_div(overlap, union), segment, 20)

    session_open = safe_open.groupby(segment, sort=False).transform("first")
    session_high = safe_high.groupby(segment, sort=False).cummax()
    session_low = safe_low.groupby(segment, sort=False).cummin()
    session_mid = (session_high + session_low) / 2.0
    session_range = session_high - session_low
    cum_pv = (safe_close * safe_volume).groupby(segment, sort=False).cumsum()
    cum_vol = safe_volume.groupby(segment, sort=False).cumsum()
    session_vwap = safe_div(cum_pv, cum_vol)
    out["feature_session_open_dist"] = (safe_close - session_open) / tick_size
    out["feature_session_high_dist"] = (safe_close - session_high) / tick_size
    out["feature_session_low_dist"] = (safe_close - session_low) / tick_size
    out["feature_session_mid_dist"] = (safe_close - session_mid) / tick_size
    out["feature_session_vwap_dist"] = (safe_close - session_vwap) / tick_size
    prior = prior_session_maps(out)
    out["feature_session_range_percentile"] = safe_div(session_range, prior["range"])
    or_ready, or_high, or_low, open_drive_up, open_drive_down = first_30_session_stats(out, valid, tick_size)
    out["feature_opening_range_30_ready"] = or_ready
    out["feature_opening_range_30_high_dist"] = ((safe_close - or_high) / tick_size).where(or_ready)
    out["feature_opening_range_30_low_dist"] = ((safe_close - or_low) / tick_size).where(or_ready)
    out["feature_opening_range_30_breakout_up"] = (safe_close > or_high) & or_ready & valid
    out["feature_opening_range_30_breakout_down"] = (safe_close < or_low) & or_ready & valid

    for window in (15, 60):
        out[f"feature_realized_vol_{window}"] = rolling_std(out["feature_ret_1"], segment, window)
    out["feature_vol_expansion_15_vs_60"] = safe_div(out["feature_realized_vol_15"], out["feature_realized_vol_60"])
    tr_mean_60 = rolling_mean(true_range, segment, 60)
    tr_std_60 = rolling_std(true_range, segment, 60)
    large_bar = (true_range > tr_mean_60 + 2.0 * tr_std_60) & valid
    large_bar_signal = (true_range > tr_mean_60 + 2.0 * tr_std_60).where(
        valid & tr_mean_60.notna() & tr_std_60.notna()
    )
    out["feature_large_bar_count_30"] = rolling_true_count_full_valid(
        large_bar_signal,
        valid,
        segment,
        30,
    )
    shock = (true_range > tr_mean_60 + 3.0 * tr_std_60) & valid_window_mask(valid, segment, 60)
    out["feature_shock_bar_flag"] = shock
    out["feature_bars_since_shock"] = bars_since(shock, valid, segment)
    vol_mean_60 = rolling_mean(safe_volume, segment, 60)
    vol_std_60 = rolling_std(safe_volume, segment, 60)
    out["feature_volume_z_60"] = safe_div(safe_volume - vol_mean_60, vol_std_60)
    range_z = safe_div(true_range - tr_mean_60, tr_std_60)
    out["feature_volume_surge_with_range"] = (out["feature_volume_z_60"] > 2.0) & (range_z > 1.0) & valid
    out["feature_volume_surge_without_progress"] = (out["feature_volume_z_60"] > 2.0) & (out["feature_efficiency_ratio_30"] < 0.2) & valid
    out["feature_range_per_volume"] = safe_div(true_range, safe_volume)
    climax = (out["feature_volume_z_60"] > 3.0) & valid
    out["feature_volume_climax_flag"] = climax
    out["feature_bars_since_volume_climax"] = bars_since(climax, valid, segment)

    out["feature_5m_ret_3"] = ret_over(close, valid, segment, 15)
    out["feature_15m_ret_4"] = ret_over(close, valid, segment, 60)
    slope_60 = (safe_close - lag(safe_close, segment, 60)) / 60.0
    out["feature_60m_trend_slope"] = slope_60.where(valid_window_mask(valid, segment, 60))
    out["feature_daily_open_dist"] = (safe_close - session_open) / tick_size
    out["feature_prior_session_high_dist"] = (safe_close - prior["high"]) / tick_size
    out["feature_prior_session_low_dist"] = (safe_close - prior["low"]) / tick_size
    out["feature_prior_session_close_dist"] = (safe_close - prior["close"]) / tick_size
    out["feature_prior_session_range_pct"] = safe_div(session_range, prior["range"])
    out["feature_overnight_gap_ticks"] = (session_open - prior["close"]) / tick_size

    minute = num_col(out, "minute_of_day")
    out["feature_time_bucket_globex_open"] = minute.between(1020, 1139) & valid
    out["feature_time_bucket_europe"] = minute.between(120, 479) & valid
    out["feature_time_bucket_us_open"] = minute.between(480, 569) & valid
    out["feature_time_bucket_midday"] = minute.between(660, 839) & valid
    out["feature_time_bucket_power_hour"] = minute.between(900, 959) & valid
    out["feature_first_30m_flag"] = num_col(out, "minutes_since_session_open").between(0, 29) & valid
    out["feature_last_30m_flag"] = num_col(out, "minutes_until_session_close").between(0, 29) & valid

    close_progress_30 = (safe_close - lag(safe_close, segment, 30)).abs()
    range_sum_30 = rolling_sum(true_range, segment, 30)
    volume_sum_30 = rolling_sum(safe_volume, segment, 30)
    out["feature_effort_result_30"] = safe_div(close_progress_30, range_sum_30)
    close_progress_ticks_30 = close_progress_30 / tick_size
    out["feature_volume_per_tick_progress_30"] = volume_sum_30 / close_progress_ticks_30.clip(lower=1.0)
    out["feature_range_without_close_progress_30"] = range_sum_30 / close_progress_30.clip(lower=tick_size)
    volume_per_range_30 = safe_div(volume_sum_30, range_sum_30)
    out["feature_absorption_proxy_30"] = volume_per_range_30 * (1.0 - out["feature_effort_result_30"].clip(0, 1))
    out["feature_exhaustion_proxy_30"] = volume_per_range_30 * out["feature_efficiency_ratio_30"] * (1.0 - safe_div(close_progress_30, range_sum_30).clip(0, 1))

    out["feature_open_drive_up"] = open_drive_up & or_ready & valid
    out["feature_open_drive_down"] = open_drive_down & or_ready & valid
    out["feature_open_drive_strength_30"] = safe_div((safe_close - session_open).abs(), session_range.clip(lower=tick_size)).where(or_ready)
    out["feature_session_one_wayness"] = safe_div((safe_close - session_open).abs(), session_range.clip(lower=tick_size))
    above_vwap = (safe_close > session_vwap).where(valid)
    below_vwap = (safe_close < session_vwap).where(valid)
    out["feature_bars_above_vwap_30"] = rolling_true_count_full_valid(
        above_vwap,
        valid,
        segment,
        30,
    )
    out["feature_bars_below_vwap_30"] = rolling_true_count_full_valid(
        below_vwap,
        valid,
        segment,
        30,
    )
    out["feature_vwap_side_persistence"] = pd.concat([out["feature_bars_above_vwap_30"], out["feature_bars_below_vwap_30"]], axis=1).max(axis=1) / 30.0
    extension = (safe_close - session_open).abs()
    adverse_from_extreme = np.where(safe_close >= session_open, session_high - safe_close, safe_close - session_low)
    out["feature_pullback_shallowness_30"] = 1.0 - safe_div(pd.Series(adverse_from_extreme, index=out.index), extension.clip(lower=tick_size))

    out["feature_session_range_extension_up"] = ((safe_close - prior["high"]).clip(lower=0.0) / tick_size).where(prior["high"].notna())
    out["feature_session_range_extension_down"] = ((prior["low"] - safe_close).clip(lower=0.0) / tick_size).where(prior["low"].notna())
    out["feature_session_acceptance_above_mid"] = rolling_mean_full_valid(
        (safe_close > session_mid).where(valid).astype(float),
        valid,
        segment,
        30,
    )
    out["feature_session_acceptance_below_mid"] = rolling_mean_full_valid(
        (safe_close < session_mid).where(valid).astype(float),
        valid,
        segment,
        30,
    )
    prior_session_high_so_far = session_high.groupby(segment, sort=False).shift(1)
    prior_session_low_so_far = session_low.groupby(segment, sort=False).shift(1)
    out["feature_failed_retest_session_high"] = (safe_high > prior_session_high_so_far) & (safe_close < prior_session_high_so_far) & valid
    out["feature_failed_retest_session_low"] = (safe_low < prior_session_low_so_far) & (safe_close > prior_session_low_so_far) & valid

    shock_direction = np.sign(safe_close - prev_close).where(shock, 0.0)
    out["feature_shock_direction"] = shock_direction
    up_shock = shock & (shock_direction > 0)
    down_shock = shock & (shock_direction < 0)
    out["feature_bars_since_up_shock"] = bars_since(up_shock, valid, segment)
    out["feature_bars_since_down_shock"] = bars_since(down_shock, valid, segment)
    retrace, continuation, decay = shock_decay_features(out, valid, segment, true_range, shock, shock_direction)
    out["feature_post_shock_retrace_pct"] = retrace
    out["feature_post_shock_continuation_pct"] = continuation
    out["feature_post_shock_range_decay"] = decay

    for feature in FEATURE_COLS:
        if feature not in out.columns:
            out[feature] = np.nan
        if out[feature].dtype == bool:
            out.loc[~valid, feature] = False
        else:
            out.loc[~valid, feature] = np.nan
    return out.copy()


def shock_decay_features(
    df: pd.DataFrame,
    valid: pd.Series,
    segment: pd.Series,
    true_range: pd.Series,
    shock: pd.Series,
    shock_direction: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    retrace_arr = np.full(len(df), np.nan, dtype=float)
    continuation_arr = np.full(len(df), np.nan, dtype=float)
    decay_arr = np.full(len(df), np.nan, dtype=float)
    close_arr = num_col(df, "close").to_numpy(dtype=float)
    high_arr = num_col(df, "high").to_numpy(dtype=float)
    low_arr = num_col(df, "low").to_numpy(dtype=float)
    valid_arr = valid.to_numpy(dtype=bool)
    shock_arr = shock.to_numpy(dtype=bool)
    shock_direction_arr = shock_direction.to_numpy(dtype=float)
    true_range_arr = true_range.to_numpy(dtype=float)

    for _, positions in df.groupby(segment, sort=False).indices.items():
        shock_close = math.nan
        shock_high = math.nan
        shock_low = math.nan
        shock_tr = math.nan
        direction = 0.0
        recent_tr: list[float] = []
        for pos in positions:
            if not valid_arr[pos]:
                shock_close = math.nan
                shock_high = math.nan
                shock_low = math.nan
                shock_tr = math.nan
                direction = 0.0
                recent_tr.clear()
                continue
            if shock_arr[pos]:
                shock_close = close_arr[pos]
                shock_high = high_arr[pos]
                shock_low = low_arr[pos]
                shock_tr = true_range_arr[pos]
                direction = shock_direction_arr[pos]
                recent_tr = [shock_tr]
            elif not math.isnan(shock_close):
                recent_tr.append(true_range_arr[pos])
            if math.isnan(shock_close) or direction == 0.0:
                continue
            if direction > 0:
                move = max(shock_close - shock_low, EPS)
                retrace_arr[pos] = max(0.0, shock_close - close_arr[pos]) / move
                continuation_arr[pos] = max(0.0, high_arr[pos] - shock_high) / move
            else:
                move = max(shock_high - shock_close, EPS)
                retrace_arr[pos] = max(0.0, close_arr[pos] - shock_close) / move
                continuation_arr[pos] = max(0.0, shock_low - low_arr[pos]) / move
            if shock_tr > EPS:
                decay_arr[pos] = np.nanmean(recent_tr[-5:]) / shock_tr
    return (
        pd.Series(retrace_arr, index=df.index, dtype=float),
        pd.Series(continuation_arr, index=df.index, dtype=float),
        pd.Series(decay_arr, index=df.index, dtype=float),
    )


def other_market_frame(path: Path, market: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    valid = compute_feature_input_valid(df)
    segment = df["session_segment_id"].astype("string")
    close = num_col(df, "close")
    ret15 = ret_over(close, valid, segment, 15)
    ret30 = ret_over(close, valid, segment, 30)
    ret60 = ret_over(close, valid, segment, 60)
    ret1 = ret_over(close, valid, segment, 1)
    return pd.DataFrame(
        {
            "ts": df["ts"],
            f"{market}_valid": valid,
            f"{market}_ret_1": ret1,
            f"{market}_ret_15": ret15,
            f"{market}_ret_30": ret30,
            f"{market}_ret_60": ret60,
        }
    )


def add_intermarket_features(
    df: pd.DataFrame,
    *,
    market: str,
    year: int,
    input_root: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    missing_rates: dict[str, float] = {}
    segment = out["session_segment_id"].astype("string")
    self_ret_15 = ret_over(num_col(out, "close"), out["feature_input_valid"], segment, 15)
    self_ret_30 = ret_over(num_col(out, "close"), out["feature_input_valid"], segment, 30)
    base = out[["ts", "session_segment_id", "feature_input_valid", "feature_ret_1"]].copy()
    base["self_ret_15"] = self_ret_15
    base["self_ret_30"] = self_ret_30
    other_data: dict[str, pd.DataFrame] = {}
    for other in TIER1_MARKETS:
        path = input_root / other / f"{year}.parquet"
        other_data[other] = other_market_frame(path, other) if other != market else None  # type: ignore[assignment]

    merged = base
    for other, frame in other_data.items():
        if frame is None:
            continue
        merged = merged.merge(frame, on="ts", how="left", sort=False)

    for other in TIER1_MARKETS:
        rel_col = f"feature_rel_ret_vs_{other}_15"
        corr_col = f"feature_corr_vs_{other}_60"
        if other == market or f"{other}_ret_15" not in merged:
            out[rel_col] = np.nan
            out[corr_col] = np.nan
            missing_rates[rel_col] = 1.0
            continue
        valid_pair = out["feature_input_valid"] & merged[f"{other}_ret_15"].notna()
        out[rel_col] = (self_ret_15 - merged[f"{other}_ret_15"]).where(valid_pair)
        out[corr_col] = rolling_corr(
            out["feature_ret_1"],
            merged[f"{other}_ret_1"],
            out["session_segment_id"].astype("string"),
            60,
        )
        missing_rates[rel_col] = float(out[rel_col].isna().mean())
        missing_rates[corr_col] = float(out[corr_col].isna().mean())

    es = merged.get("ES_ret_30")
    zn = merged.get("ZN_ret_30")
    cl = merged.get("CL_ret_30")
    out["feature_es_zn_divergence_30"] = (es - zn) if es is not None and zn is not None else np.nan
    out["feature_cl_es_divergence_30"] = (cl - es) if cl is not None and es is not None else np.nan
    out["feature_es_zn_risk_regime_30"] = out["feature_es_zn_divergence_30"]
    out["feature_cl_es_macro_divergence_30"] = out["feature_cl_es_divergence_30"]
    risk_components: list[pd.Series] = []
    if market != "ES" and es is not None:
        risk_components.append(es.rename("ES"))
    if market != "ZN" and zn is not None:
        risk_components.append((-zn).rename("ZN"))
    if market != "CL" and cl is not None:
        risk_components.append((0.5 * cl).rename("CL"))
    if risk_components:
        risk_frame = pd.concat(risk_components, axis=1)
        out["feature_tier1_risk_on_score_30"] = risk_frame.sum(
            axis=1,
            min_count=len(risk_components),
        )
    else:
        out["feature_tier1_risk_on_score_30"] = np.nan

    signs = []
    for other in TIER1_MARKETS:
        if other == market or f"{other}_ret_15" not in merged:
            continue
        signs.append(np.sign(merged[f"{other}_ret_15"]))
    if signs:
        sign_frame = pd.concat(signs, axis=1)
        self_sign = np.sign(self_ret_15)
        out["feature_tier1_direction_agreement_15"] = sign_frame.eq(self_sign, axis=0).sum(axis=1) / sign_frame.notna().sum(axis=1)
        ret_frame = pd.concat(
            [merged[f"{other}_ret_15"] for other in TIER1_MARKETS if other != market and f"{other}_ret_15" in merged],
            axis=1,
        )
        out["feature_tier1_return_dispersion_15"] = ret_frame.std(axis=1)
    else:
        out["feature_tier1_direction_agreement_15"] = np.nan
        out["feature_tier1_return_dispersion_15"] = np.nan

    for feature in FEATURE_FAMILIES["tier1_intermarket"] + FEATURE_FAMILIES["tier1_cross_market_regime"]:
        if feature not in out.columns:
            out[feature] = np.nan
        out.loc[~out["feature_input_valid"], feature] = np.nan
        missing_rates.setdefault(feature, float(out[feature].isna().mean()))
    return out, missing_rates


def target_columns(columns: Iterable[str]) -> list[str]:
    return sorted(
        col
        for col in columns
        if col.startswith("target_") or col in REGIME_LABEL_COLUMNS
    )


def metadata_columns(columns: Iterable[str]) -> list[str]:
    preferred = [
        "ts",
        "market",
        "year",
        "feature_input_valid",
        "feature_row_valid",
        "training_row_valid",
        "target_valid",
    ]
    return [col for col in preferred if col in columns]


def excluded_columns(columns: Iterable[str], feature_cols: list[str], target_cols: list[str], metadata_cols: list[str]) -> list[str]:
    assigned = set(feature_cols) | set(target_cols) | set(metadata_cols)
    return sorted(col for col in columns if col not in assigned)


def validate_registry(feature_cols: list[str]) -> list[str]:
    failures: list[str] = []
    bad_prefix = [col for col in feature_cols if not col.startswith("feature_")]
    if bad_prefix:
        failures.append(f"feature columns without feature_ prefix: {bad_prefix}")
    forbidden = [
        col
        for col in feature_cols
        if col in FORBIDDEN_FEATURE_COLUMNS or col.startswith("target_")
    ]
    if forbidden:
        failures.append(f"forbidden columns in feature_cols: {forbidden}")
    missing_family = [col for col in feature_cols if col not in FEATURE_TO_FAMILY]
    if missing_family:
        failures.append(f"feature columns missing family mapping: {missing_family}")
    return failures


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    profile: str,
    costs_config: Path = DEFAULT_COSTS_CONFIG,
    input_root: Path = DEFAULT_INPUT_ROOT,
) -> FeatureResult:
    market = input_path.parent.name
    year = int(input_path.stem)
    result = FeatureResult(
        profile=profile,
        market=market,
        year=year,
        input_path=relative_path(input_path),
        output_path=relative_path(output_path),
    )
    if not input_path.exists():
        result.failures.append(f"missing input file: {relative_path(input_path)}")
        return result

    df = pd.read_parquet(input_path)
    result.input_rows = int(len(df))
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        result.failures.append(f"missing required input columns: {','.join(missing)}")
        return result

    label_failures = validate_label_contract(df)
    if label_failures:
        result.failures.extend(label_failures)
        return result

    tick_size, tick_failure = resolve_market_tick_size(costs_config, market)
    if tick_failure:
        result.failures.append(tick_failure)
        return result

    out = add_base_market_features(df, tick_size)
    out, intermarket_missing = add_intermarket_features(
        out,
        market=market,
        year=year,
        input_root=input_root,
    )
    result.intermarket_missing_pct = intermarket_missing
    registry_failures = validate_registry(FEATURE_COLS)
    result.failures.extend(registry_failures)

    result.output_rows = int(len(out))
    result.feature_input_valid_rows = int(out["feature_input_valid"].sum())
    result.training_row_valid_rows = int(out["training_row_valid"].sum())
    result.target_valid_rows = int(bool_col(out, "target_valid").sum())
    result.feature_count = len(FEATURE_COLS)
    result.nan_counts = {feature: int(out[feature].isna().sum()) for feature in FEATURE_COLS}
    result.nan_pct = {
        feature: round(float(out[feature].isna().mean()), 6) for feature in FEATURE_COLS
    }
    unavailable = [feature for feature, pct in result.nan_pct.items() if pct >= 1.0]
    if unavailable:
        result.warnings.append(f"features fully unavailable: {','.join(unavailable)}")

    if result.failures:
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    out.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_registries(output_root: Path, reports_root: Path, columns: list[str]) -> dict[str, list[str] | dict[str, str]]:
    feature_cols = FEATURE_COLS.copy()
    target_cols = target_columns(columns)
    metadata_cols = metadata_columns(columns)
    excluded_cols = excluded_columns(columns, feature_cols, target_cols, metadata_cols)
    registry = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "metadata_cols": metadata_cols,
        "excluded_cols": excluded_cols,
        "feature_families": FEATURE_TO_FAMILY,
    }
    write_json(output_root / "feature_cols.json", feature_cols)
    write_json(output_root / "target_cols.json", target_cols)
    write_json(output_root / "metadata_cols.json", metadata_cols)
    write_json(output_root / "excluded_cols.json", excluded_cols)
    write_json(reports_root / "feature_registry.json", registry)
    return registry


def high_correlation_report(results: list[FeatureResult], feature_cols: list[str], reports_root: Path) -> int:
    frames: list[pd.DataFrame] = []
    remaining = 200_000
    for result in results:
        if result.status == "FAIL" or remaining <= 0:
            continue
        path = Path(result.output_path)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=[*feature_cols, "training_row_valid"])
        train = df.loc[df["training_row_valid"], feature_cols].dropna(how="all")
        if train.empty:
            continue
        take = min(remaining, len(train))
        frames.append(train.head(take))
        remaining -= take
    rows: list[dict[str, object]] = []
    if frames:
        sample = pd.concat(frames, ignore_index=True)
        corr = sample.corr(numeric_only=True)
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and abs(float(value)) >= 0.98:
                    rows.append({"feature_a": left, "feature_b": right, "corr": float(value)})
    reports_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["feature_a", "feature_b", "corr"]).to_csv(
        reports_root / "feature_correlation_report.csv",
        index=False,
    )
    return len(rows)


def write_reports(
    results: list[FeatureResult],
    *,
    profile: str,
    output_root: Path,
    reports_root: Path,
    profile_config: Path = DEFAULT_PROFILE_CONFIG,
    costs_config: Path = DEFAULT_COSTS_CONFIG,
) -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    output_columns: list[str] = []
    for result in results:
        path = Path(result.output_path)
        if path.exists():
            output_columns = pd.read_parquet(path, columns=[]).columns.tolist()
            output_columns = pd.read_parquet(path).columns.tolist()
            break
    if not output_columns:
        output_columns = FEATURE_COLS.copy()
    registry = write_registries(output_root, reports_root, output_columns)
    corr_pairs = high_correlation_report(results, FEATURE_COLS, reports_root)
    failures = [failure for result in results for failure in result.failures]
    warnings = [warning for result in results for warning in result.warnings]
    input_hashes = file_hash_map(Path(result.input_path) for result in results)
    output_hashes = file_hash_map(Path(result.output_path) for result in results)
    config_digest = config_hash([profile_config, costs_config])
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "script_path": relative_path(Path(__file__)),
        "script_hash": file_sha256(Path(__file__)),
        "config_hash": config_digest,
        "input_file_hashes": input_hashes,
        "output_file_hashes": output_hashes,
        "profile": profile,
        "markets": sorted({result.market for result in results}),
        "years": sorted({result.year for result in results}),
        "feature_count": len(FEATURE_COLS),
        "feature_family_counts": {family: len(features) for family, features in FEATURE_FAMILIES.items()},
        "forbidden_feature_leakage_failures": validate_registry(FEATURE_COLS),
        "warning_count": len(warnings),
        "failure_count": len(failures),
        "failures": failures,
        "outputs": [result.to_dict() for result in results],
        "registry": registry,
    }
    report = {
        "generated_at": manifest["generated_at"],
        "git_commit": manifest["git_commit"],
        "script_path": manifest["script_path"],
        "script_hash": manifest["script_hash"],
        "config_hash": config_digest,
        "input_file_hashes": input_hashes,
        "output_file_hashes": output_hashes,
        "profile": profile,
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "summary": {
            "file_count": len(results),
            "pass_count": sum(result.status == "PASS" for result in results),
            "warn_count": sum(result.status == "WARN" for result in results),
            "fail_count": sum(result.status == "FAIL" for result in results),
            "input_rows": sum(result.input_rows for result in results),
            "output_rows": sum(result.output_rows for result in results),
            "feature_input_valid_rows": sum(result.feature_input_valid_rows for result in results),
            "training_row_valid_rows": sum(result.training_row_valid_rows for result in results),
            "target_valid_rows": sum(result.target_valid_rows for result in results),
            "feature_count": len(FEATURE_COLS),
            "high_corr_pair_count": corr_pairs,
        },
        "files": [result.to_dict() for result in results],
        "warning_count": len(warnings),
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json(reports_root / "baseline_feature_manifest.json", manifest)
    write_json(reports_root / "baseline_feature_report.json", report)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT.as_posix())
    parser.add_argument("--profile-config", default=DEFAULT_PROFILE_CONFIG.as_posix())
    parser.add_argument("--costs-config", default=DEFAULT_COSTS_CONFIG.as_posix())
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    inputs = resolve_profile_inputs(args.profile, input_root, Path(args.profile_config))
    results: list[FeatureResult] = []
    for market, year, input_path in inputs:
        output_path = output_root / market / f"{year}.parquet"
        result = process_file(
            input_path,
            output_path,
            profile=args.profile,
            costs_config=Path(args.costs_config),
            input_root=input_root,
        )
        results.append(result)
        print(
            f"{result.status} {market} {year}: rows={result.output_rows} "
            f"features={result.feature_count} input_valid={result.feature_input_valid_rows} "
            f"training_valid={result.training_row_valid_rows} warnings={len(result.warnings)} "
            f"failures={len(result.failures)}"
        )
    write_reports(
        results,
        profile=args.profile,
        output_root=output_root,
        reports_root=reports_root,
        profile_config=Path(args.profile_config),
        costs_config=Path(args.costs_config),
    )
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
