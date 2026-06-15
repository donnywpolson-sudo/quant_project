#!/usr/bin/env python3
"""Build deterministic walk-forward split plans from Phase 4 feature matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from scripts.final_holdout.guard_final_holdout import (
    final_holdout_permission_failure,
    is_final_holdout_year_set,
)
from scripts.validation.model_registry import resolve_purge_bars, validate_purge_policy
from scripts.validation.check_tier_2_coverage import PRODUCT_AVAILABLE_START_YEAR


DEFAULT_PROFILE = "tier_1"
DEFAULT_INPUT_ROOT = Path("data/feature_matrices/baseline")
DEFAULT_REPORTS_ROOT = Path("reports/wfa")
DEFAULT_PROFILE_CONFIG = Path("configs/alpha_tiered.yaml")
DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")


@dataclass(frozen=True)
class ProfilePlan:
    requested_profile: str
    resolved_profile: str
    markets: list[str]
    years: list[int]
    settings_profile: str
    train_days: int
    test_days: int
    step_days: int
    final_holdout_years: list[int]
    forbid_research_use: bool
    intent: str


@dataclass(frozen=True)
class WfaPolicy:
    purge_bars: int
    resolved_purge_bars: int
    embargo_bars: int
    final_holdout_tuning_allowed: bool
    final_holdout_excluded_from_selection: bool


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hash_or_missing(path: Path) -> str:
    return _file_sha256(path) if path.exists() else "MISSING"


def _file_hash_map(paths: Iterable[Path]) -> dict[str, str]:
    return {_relative_path(path): _file_hash_or_missing(path) for path in paths}


def _config_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(_relative_path(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_hash_or_missing(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _resolve_alias(profile: str, aliases: Mapping[str, Any]) -> str:
    resolved = profile
    seen: set[str] = set()
    while resolved in aliases:
        if resolved in seen:
            raise SystemExit(f"Profile alias cycle detected at {resolved!r}")
        seen.add(resolved)
        resolved = str(aliases[resolved])
    return resolved


def load_profile_plan(profile: str, profile_config: Path) -> ProfilePlan:
    config = _read_yaml(profile_config)
    aliases = config.get("aliases", {})
    if not isinstance(aliases, Mapping):
        aliases = {}
    resolved_profile = _resolve_alias(profile, aliases)

    profiles = config.get("profiles", {})
    if not isinstance(profiles, Mapping) or resolved_profile not in profiles:
        known = sorted(str(key) for key in profiles) if isinstance(profiles, Mapping) else []
        raise SystemExit(f"Unknown profile {profile!r}. Known profiles: {', '.join(known)}")

    profile_entry = profiles[resolved_profile]
    if not isinstance(profile_entry, Mapping):
        raise SystemExit(f"Invalid profile config for {resolved_profile!r}")
    if bool(profile_entry.get("discovery", False)):
        raise SystemExit("Discovery profiles are inventory-only and cannot feed WFA splits")

    markets = profile_entry.get("markets", [])
    years = profile_entry.get("years", [])
    if not isinstance(markets, list) or not markets:
        raise SystemExit(f"Profile {resolved_profile!r} has no markets")
    if not isinstance(years, list) or not years:
        raise SystemExit(f"Profile {resolved_profile!r} has no years")

    settings_name = str(profile_entry.get("settings_profile", ""))
    profile_defaults = config.get("profile_defaults", {})
    if not isinstance(profile_defaults, Mapping) or settings_name not in profile_defaults:
        raise SystemExit(f"Profile {resolved_profile!r} references unknown settings profile {settings_name!r}")
    settings = profile_defaults[settings_name]
    if not isinstance(settings, Mapping):
        raise SystemExit(f"Invalid settings profile {settings_name!r}")

    defaults = config.get("defaults", {})
    if not isinstance(defaults, Mapping):
        defaults = {}
    final_holdout_years = defaults.get("final_holdout_years", [])
    if not isinstance(final_holdout_years, list):
        final_holdout_years = []

    return ProfilePlan(
        requested_profile=profile,
        resolved_profile=resolved_profile,
        markets=[str(market) for market in markets],
        years=[int(year) for year in years],
        settings_profile=settings_name,
        train_days=int(settings["train_days"]),
        test_days=int(settings["test_days"]),
        step_days=int(settings["step_days"]),
        final_holdout_years=[int(year) for year in final_holdout_years],
        forbid_research_use=bool(profile_entry.get("forbid_research_use", False)),
        intent=str(profile_entry.get("intent", "")),
    )


def load_wfa_policy(models_config: Path) -> WfaPolicy:
    config = _read_yaml(models_config)
    policy = config.get("policy", {})
    if not isinstance(policy, Mapping):
        raise SystemExit("models policy mapping missing")
    if policy.get("random_splits_allowed") is not False:
        raise SystemExit("random splits must be disabled for WFA")
    if policy.get("final_holdout_tuning_allowed") is not False:
        raise SystemExit("final-holdout tuning must be disabled for WFA")

    purge_errors = validate_purge_policy(config)
    if purge_errors:
        raise SystemExit("; ".join(purge_errors))
    purge = config["purge"]
    resolved = resolve_purge_bars(purge)

    selection = config.get("model_selection_reports", {})
    if not isinstance(selection, Mapping):
        raise SystemExit("model_selection_reports mapping missing")
    final_excluded = selection.get("final_holdout_excluded_from_selection")
    if final_excluded is not True:
        raise SystemExit("final holdout must be excluded from model selection")

    return WfaPolicy(
        purge_bars=resolved,
        resolved_purge_bars=resolved,
        embargo_bars=int(purge.get("embargo_bars", resolved)),
        final_holdout_tuning_allowed=False,
        final_holdout_excluded_from_selection=True,
    )


def resolve_input_paths(plan: ProfilePlan, input_root: Path) -> list[tuple[str, int, Path]]:
    return [
        (market, year, input_root / market / f"{year}.parquet")
        for market in plan.markets
        for year in plan.years
    ]


def _read_feature_rows(path: Path, market: str, year: int) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"missing feature matrix: {_relative_path(path)}"
    try:
        import pyarrow.parquet as pq

        available = list(pq.read_schema(path).names)
    except Exception as exc:
        return None, f"unreadable feature matrix schema: {_relative_path(path)}: {exc}"

    if "ts" not in available:
        return None, f"feature matrix missing ts column: {_relative_path(path)}"
    optional = ["market", "year", "training_row_valid", "target_valid", "feature_input_valid"]
    columns = ["ts", *[column for column in optional if column in available]]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        return None, f"unreadable feature matrix rows: {_relative_path(path)}: {exc}"

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.loc[frame["ts"].notna()].copy()
    frame["market"] = market
    frame["year"] = year

    if "training_row_valid" in frame:
        eligible = frame["training_row_valid"].fillna(False).astype(bool)
    elif {"target_valid", "feature_input_valid"}.issubset(frame.columns):
        eligible = frame["target_valid"].fillna(False).astype(bool) & frame[
            "feature_input_valid"
        ].fillna(False).astype(bool)
    elif "target_valid" in frame:
        eligible = frame["target_valid"].fillna(False).astype(bool)
    else:
        eligible = pd.Series(True, index=frame.index)
    frame["wfa_row_eligible"] = eligible

    return frame[["ts", "market", "year", "wfa_row_eligible"]], None


def _iso(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None:
        return None
    return timestamp.isoformat()


def _split_group(plan: ProfilePlan) -> tuple[str, bool, str | None]:
    final_years = set(plan.final_holdout_years)
    profile_years = set(plan.years)
    if profile_years & final_years and not profile_years <= final_years:
        return "invalid", False, "profile mixes research and final-holdout years"
    if profile_years <= final_years and final_years:
        return "final_holdout", True, None
    if "forward" in plan.intent:
        return "forward", False, None
    if plan.forbid_research_use:
        return "restricted", False, None
    return "research", False, None


def build_market_folds(
    market: str,
    frame: pd.DataFrame,
    plan: ProfilePlan,
    policy: WfaPolicy,
) -> tuple[list[dict[str, Any]], list[str]]:
    split_group, is_final_holdout, error = _split_group(plan)
    if error is not None:
        return [], [f"{market}: {error}"]

    frame = frame.sort_values("ts").reset_index(drop=True)
    if frame.empty:
        return [], [f"{market}: no feature rows available"]

    timestamps = frame["ts"]
    eligible = frame["wfa_row_eligible"].astype(bool)
    first_ts = timestamps.iloc[0]
    last_ts = timestamps.iloc[-1]
    test_start = first_ts + pd.Timedelta(days=plan.train_days)
    step = pd.Timedelta(days=plan.step_days)
    test_span = pd.Timedelta(days=plan.test_days)
    folds: list[dict[str, Any]] = []

    while test_start + test_span <= last_ts:
        test_end_exclusive = test_start + test_span
        train_start = test_start - pd.Timedelta(days=plan.train_days)
        train_before = eligible & (timestamps >= train_start) & (timestamps < test_start)
        test_mask = eligible & (timestamps >= test_start) & (timestamps < test_end_exclusive)
        test_start_pos = int(timestamps.searchsorted(test_start, side="left"))
        purge_end_pos = test_start_pos - policy.resolved_purge_bars

        if purge_end_pos > 0:
            purged_train_end = timestamps.iloc[purge_end_pos - 1]
            train_after = train_before & (timestamps <= purged_train_end)
        else:
            purged_train_end = None
            train_after = pd.Series(False, index=frame.index)

        train_rows_before = int(train_before.sum())
        train_rows_after = int(train_after.sum())
        test_rows = int(test_mask.sum())
        if train_rows_after > 0 and test_rows > 0 and purged_train_end is not None:
            test_rows_frame = frame.loc[test_mask, "ts"]
            train_before_frame = frame.loc[train_before, "ts"]
            train_after_frame = frame.loc[train_after, "ts"]
            test_end_inclusive = test_rows_frame.iloc[-1]
            embargo_start_pos = int(timestamps.searchsorted(test_end_exclusive, side="left"))
            embargo_end = None
            embargo_rows = 0
            if embargo_start_pos < len(frame):
                embargo_end_pos = min(len(frame) - 1, embargo_start_pos + policy.embargo_bars - 1)
                embargo_end = timestamps.iloc[embargo_end_pos]
                embargo_rows = int(embargo_end_pos - embargo_start_pos + 1)

            fold_number = len(folds) + 1
            folds.append(
                {
                    "market": market,
                    "fold_id": f"{market}_{split_group}_{fold_number:04d}",
                    "fold_number": fold_number,
                    "split_group": split_group,
                    "train_start": _iso(train_before_frame.iloc[0]),
                    "train_end": _iso(train_before_frame.iloc[-1]),
                    "purged_train_end": _iso(train_after_frame.iloc[-1]),
                    "test_start": _iso(test_rows_frame.iloc[0]),
                    "test_end": _iso(test_end_inclusive),
                    "embargo_end": _iso(embargo_end),
                    "train_rows_before_purge": train_rows_before,
                    "train_rows_after_purge": train_rows_after,
                    "purged_train_rows": train_rows_before - train_rows_after,
                    "test_rows": test_rows,
                    "embargo_rows": embargo_rows,
                    "purge_bars": policy.purge_bars,
                    "resolved_purge_bars": policy.resolved_purge_bars,
                    "embargo_bars": policy.embargo_bars,
                    "is_final_holdout": is_final_holdout,
                    "final_holdout": is_final_holdout,
                    "selection_allowed": split_group == "research",
                }
            )
        test_start += step

    if not folds:
        return [], [f"{market}: no non-empty WFA folds"]
    return folds, []


def build_split_plan(
    *,
    profile: str,
    input_root: Path,
    reports_root: Path,
    profile_config: Path,
    models_config: Path,
    allow_final_holdout: bool = False,
) -> dict[str, Any]:
    plan = load_profile_plan(profile, profile_config)
    permission_failure = final_holdout_permission_failure(
        is_final_holdout=is_final_holdout_year_set(plan.years, plan.final_holdout_years),
        allow_final_holdout=allow_final_holdout,
        action=f"final-holdout split-plan generation for profile {plan.requested_profile!r}",
    )
    if permission_failure is not None:
        raise SystemExit(permission_failure)
    policy = load_wfa_policy(models_config)
    inputs = resolve_input_paths(plan, input_root)
    frames_by_market: dict[str, list[pd.DataFrame]] = {market: [] for market in plan.markets}
    failures: list[str] = []
    skipped_inputs: list[dict[str, Any]] = []
    hashed_inputs: list[Path] = []

    for market, year, path in inputs:
        available_start = PRODUCT_AVAILABLE_START_YEAR.get(market)
        if available_start is not None and year < available_start:
            skipped_inputs.append(
                {
                    "market": market,
                    "year": year,
                    "path": _relative_path(path),
                    "reason": f"product_unavailable_before_{available_start}",
                }
            )
            continue
        hashed_inputs.append(path)
        frame, failure = _read_feature_rows(path, market, year)
        if failure is not None:
            failures.append(failure)
            continue
        assert frame is not None
        frames_by_market[market].append(frame)

    folds: list[dict[str, Any]] = []
    for market, frames in frames_by_market.items():
        if not frames:
            failures.append(f"{market}: no readable feature matrices")
            continue
        market_folds, market_failures = build_market_folds(
            market,
            pd.concat(frames, ignore_index=True),
            plan,
            policy,
        )
        folds.extend(market_folds)
        failures.extend(market_failures)

    split_rows = pd.DataFrame(folds)
    reports_root.mkdir(parents=True, exist_ok=True)
    split_rows.to_csv(reports_root / "split_plan.csv", index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_path": _relative_path(Path(__file__)),
        "script_hash": _file_sha256(Path(__file__)),
        "config_hash": _config_hash([profile_config, models_config]),
        "input_file_hashes": _file_hash_map(hashed_inputs),
        "profile": plan.requested_profile,
        "resolved_profile": plan.resolved_profile,
        "input_root": _relative_path(input_root),
        "output_root": _relative_path(reports_root),
        "reports_root": _relative_path(reports_root),
        "markets": plan.markets,
        "years": plan.years,
        "settings_profile": plan.settings_profile,
        "window_policy": {
            "train_days": plan.train_days,
            "test_days": plan.test_days,
            "step_days": plan.step_days,
        },
        "purge_policy": {
            "purge_bars": policy.purge_bars,
            "resolved_purge_bars": policy.resolved_purge_bars,
            "embargo_bars": policy.embargo_bars,
        },
        "final_holdout_policy": {
            "final_holdout_years": plan.final_holdout_years,
            "final_holdout_tuning_allowed": policy.final_holdout_tuning_allowed,
            "final_holdout_excluded_from_selection": policy.final_holdout_excluded_from_selection,
        },
        "fold_count": len(folds),
        "fold_count_by_market": split_rows.groupby("market").size().to_dict() if folds else {},
        "skipped_input_count": len(skipped_inputs),
        "skipped_inputs": skipped_inputs,
        "warning_count": 0,
        "failure_count": len(failures),
        "failures": failures,
        "folds": folds,
    }
    (reports_root / "split_plan.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT.as_posix())
    parser.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT.as_posix())
    parser.add_argument("--profile-config", default=DEFAULT_PROFILE_CONFIG.as_posix())
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG.as_posix())
    parser.add_argument("--allow-final-holdout", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_split_plan(
        profile=args.profile,
        input_root=Path(args.input_root),
        reports_root=Path(args.reports_root),
        profile_config=Path(args.profile_config),
        models_config=Path(args.models_config),
        allow_final_holdout=args.allow_final_holdout,
    )
    status = "FAIL" if manifest["failure_count"] else "PASS"
    print(
        f"{status} WFA split plan: folds={manifest['fold_count']} "
        f"markets={len(manifest['markets'])} failures={manifest['failure_count']}"
    )
    return 1 if manifest["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
