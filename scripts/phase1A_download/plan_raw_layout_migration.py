#!/usr/bin/env python3
"""Dry-run planner for migrating legacy Phase 1A DBN archives.

This script only reports proposed moves. It never creates, moves, deletes, or
rewrites data files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from scripts.phase1_raw_contract import (
    EXPECTED_COMPRESSION,
    EXPECTED_ENCODING,
    REQUIRED_DATASET,
    REQUIRED_MANIFEST_FIELDS,
    SCHEMA_PATHS,
    VENDOR,
)

SCHEMA_OHLCV = "ohlcv-1m"
SCHEMA_DEFINITION = "definition"
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_DBN_ROOT = Path("data/dbn")


@dataclass
class MigrationItem:
    source_path: str
    source_manifest_path: str
    target_path: str | None
    target_manifest_path: str | None
    schema: str | None
    market: str | None
    year: int | None
    start: str | None
    end: str | None
    source_sha256: str | None
    action: str
    unsafe_reasons: list[str]
    manifest_path_update_required: bool


def is_dbn_file(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".dbn.zst") or path.name.endswith(".dbn"))


def sidecar_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dbn_suffix(path: Path) -> str:
    if path.name.endswith(".dbn.zst"):
        return ".dbn.zst"
    if path.name.endswith(".dbn"):
        return ".dbn"
    raise ValueError(f"not a DBN file: {path}")


def discover_legacy_dbn_files(raw_root: Path) -> list[Path]:
    paths: list[Path] = []
    definition_root = raw_root / SCHEMA_PATHS[SCHEMA_DEFINITION]
    if raw_root.exists():
        for market_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            if market_dir == definition_root:
                continue
            paths.extend(sorted(path for path in market_dir.iterdir() if is_dbn_file(path)))
    if definition_root.exists():
        for market_dir in sorted(path for path in definition_root.iterdir() if path.is_dir()):
            paths.extend(sorted(path for path in market_dir.iterdir() if is_dbn_file(path)))
    return paths


def infer_legacy_path_fields(path: Path, raw_root: Path) -> tuple[str | None, str | None, int | None, list[str]]:
    reasons: list[str] = []
    try:
        parts = path.relative_to(raw_root).parts
    except ValueError:
        parts = path.parts
    schema: str | None
    market: str | None
    filename: str | None
    if len(parts) == 2:
        schema = SCHEMA_OHLCV
        market = parts[0]
        filename = parts[1]
    elif len(parts) == 3 and parts[0] == SCHEMA_PATHS[SCHEMA_DEFINITION]:
        schema = SCHEMA_DEFINITION
        market = parts[1]
        filename = parts[2]
    else:
        return None, None, None, [f"unsupported legacy path layout: {path.as_posix()}"]
    year_text = filename.removesuffix(".dbn.zst").removesuffix(".dbn") if filename else ""
    year = int(year_text) if year_text.isdigit() else None
    if year is None:
        reasons.append("legacy DBN filename is not a numeric year")
    return schema, market, year, reasons


def read_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = sidecar_manifest_path(path)
    if not manifest_path.exists():
        return None, [f"missing manifest: {manifest_path.as_posix()}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"unreadable manifest: {exc}"]
    if not isinstance(payload, dict):
        return None, ["manifest is not a JSON object"]
    return payload, []


def validate_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    schema: str | None,
    market: str | None,
    year: int | None,
    source_hash: str,
) -> list[str]:
    reasons: list[str] = []
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        reasons.append("manifest missing fields: " + ",".join(missing))
    if manifest.get("vendor") != VENDOR:
        reasons.append("manifest vendor mismatch")
    if manifest.get("dataset") != REQUIRED_DATASET:
        reasons.append("manifest dataset mismatch")
    if schema is not None and manifest.get("schema") != schema:
        reasons.append("manifest schema mismatch")
    if market is not None and manifest.get("market") != market:
        reasons.append("manifest market mismatch")
    if manifest.get("encoding") != EXPECTED_ENCODING:
        reasons.append("manifest encoding mismatch")
    if manifest.get("compression") != EXPECTED_COMPRESSION:
        reasons.append("manifest compression mismatch")
    if manifest.get("path") != path.as_posix():
        reasons.append("manifest path mismatch")
    if int(manifest.get("file_size_bytes") or 0) != path.stat().st_size:
        reasons.append("manifest file_size_bytes mismatch")
    if manifest.get("file_sha256") != source_hash:
        reasons.append("manifest file_sha256 mismatch")
    if year is not None:
        try:
            start = date.fromisoformat(str(manifest.get("start")))
            end = date.fromisoformat(str(manifest.get("end")))
            if start.year != year or end <= start or end > date(year + 1, 1, 1):
                reasons.append("manifest time range does not match legacy path year")
        except Exception:
            reasons.append("manifest time range invalid")
    return reasons


def target_path_for_manifest(
    target_dbn_root: Path,
    manifest: dict[str, Any],
    *,
    schema: str,
    market: str,
    year: int,
    source_path: Path,
) -> Path | None:
    start = str(manifest.get("start") or "")
    end = str(manifest.get("end") or "")
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError:
        return None
    if start_date.year != year or end_date <= start_date or end_date > date(year + 1, 1, 1):
        return None
    schema_dir = SCHEMA_PATHS[schema]
    return target_dbn_root / schema_dir / market / str(year) / f"{start}_{end}{dbn_suffix(source_path)}"


def plan_item(path: Path, raw_root: Path, target_dbn_root: Path) -> MigrationItem:
    manifest_path = sidecar_manifest_path(path)
    schema, market, year, reasons = infer_legacy_path_fields(path, raw_root)
    if not path.name.endswith(".dbn.zst"):
        reasons.append("legacy DBN file is not .dbn.zst")
    source_hash = file_sha256(path)
    manifest, manifest_reasons = read_manifest(path)
    reasons.extend(manifest_reasons)
    target_path: Path | None = None
    if manifest is not None:
        reasons.extend(
            validate_manifest(
                path,
                manifest,
                schema=schema,
                market=market,
                year=year,
                source_hash=source_hash,
            )
        )
        if schema is not None and market is not None and year is not None:
            target_path = target_path_for_manifest(
                target_dbn_root,
                manifest,
                schema=schema,
                market=market,
                year=year,
                source_path=path,
            )
            if target_path is None:
                reasons.append("cannot build target path from manifest dates")
    if target_path is not None and target_path.exists():
        target_hash = file_sha256(target_path)
        if target_hash == source_hash:
            reasons.append("target already exists with identical hash")
        else:
            reasons.append("target already exists with different hash")
    target_manifest = sidecar_manifest_path(target_path) if target_path is not None else None
    if target_manifest is not None and target_manifest.exists():
        reasons.append("target manifest already exists")
    unsafe = [reason for reason in reasons if not reason.startswith("target already exists with identical hash")]
    if unsafe:
        action = "unsafe"
    elif target_path is not None and target_path.exists():
        action = "skip_target_exists_same_hash"
    else:
        action = "plan_move"
    return MigrationItem(
        source_path=path.as_posix(),
        source_manifest_path=manifest_path.as_posix(),
        target_path=target_path.as_posix() if target_path is not None else None,
        target_manifest_path=target_manifest.as_posix() if target_manifest is not None else None,
        schema=schema,
        market=market,
        year=year,
        start=str(manifest.get("start")) if manifest is not None and manifest.get("start") is not None else None,
        end=str(manifest.get("end")) if manifest is not None and manifest.get("end") is not None else None,
        source_sha256=source_hash,
        action=action,
        unsafe_reasons=unsafe,
        manifest_path_update_required=(
            manifest is not None and target_path is not None and manifest.get("path") != target_path.as_posix()
        ),
    )


def mark_duplicate_targets(items: list[MigrationItem]) -> None:
    by_target: dict[str, list[MigrationItem]] = {}
    for item in items:
        if item.target_path is not None:
            by_target.setdefault(item.target_path, []).append(item)
    for target, target_items in by_target.items():
        if len(target_items) <= 1:
            continue
        hashes = {item.source_sha256 for item in target_items}
        reason = (
            "duplicate planned target with identical source hashes"
            if len(hashes) == 1
            else "duplicate planned target with different source hashes"
        )
        for item in target_items:
            if reason not in item.unsafe_reasons:
                item.unsafe_reasons.append(reason)
            item.action = "unsafe"


def build_plan(raw_root: Path, target_dbn_root: Path) -> dict[str, Any]:
    sources = discover_legacy_dbn_files(raw_root)
    items = [plan_item(path, raw_root, target_dbn_root) for path in sources]
    mark_duplicate_targets(items)
    counts = {
        "total": len(items),
        "plan_move": sum(1 for item in items if item.action == "plan_move"),
        "unsafe": sum(1 for item in items if item.action == "unsafe"),
        "skip_target_exists_same_hash": sum(
            1 for item in items if item.action == "skip_target_exists_same_hash"
        ),
    }
    return {
        "dry_run": True,
        "raw_root": raw_root.as_posix(),
        "target_dbn_root": target_dbn_root.as_posix(),
        "counts": counts,
        "items": [asdict(item) for item in items],
    }


def print_text_report(plan: dict[str, Any]) -> None:
    counts = plan["counts"]
    print(
        "DRY_RUN raw_layout_migration "
        f"total={counts['total']} plan_move={counts['plan_move']} "
        f"unsafe={counts['unsafe']} skip={counts['skip_target_exists_same_hash']}"
    )
    for item in plan["items"]:
        target = item["target_path"] or "<no-target>"
        print(f"{item['action'].upper()} {item['source_path']} -> {target}")
        for reason in item["unsafe_reasons"]:
            print(f"  UNSAFE {reason}")
        if item["manifest_path_update_required"]:
            print("  NOTE manifest path field must be rewritten during an approved migration")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT.as_posix())
    parser.add_argument("--target-dbn-root", default=DEFAULT_DBN_ROOT.as_posix())
    parser.add_argument("--json", action="store_true", help="Print the dry-run plan as JSON.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    plan = build_plan(Path(args.raw_root), Path(args.target_dbn_root))
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print_text_report(plan)
    return 1 if plan["counts"]["unsafe"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
