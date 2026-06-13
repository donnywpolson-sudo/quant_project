from __future__ import annotations

import json
from pathlib import Path

from scripts.phase1A_download.plan_raw_layout_migration import (
    build_plan,
    file_sha256,
    sidecar_manifest_path,
)


def write_manifest(
    path: Path,
    *,
    schema: str,
    market: str = "ES",
    start: str = "2024-01-01",
    end: str = "2025-01-01",
) -> None:
    payload = {
        "vendor": "databento",
        "dataset": "GLBX.MDP3",
        "schema": schema,
        "market": market,
        "symbols_requested": [f"{market}.v.0" if schema == "ohlcv-1m" else f"{market}.FUT"],
        "start": start,
        "end": end,
        "stype_in": "continuous" if schema == "ohlcv-1m" else "parent",
        "stype_out": "instrument_id",
        "encoding": "dbn",
        "compression": "zstd",
        "downloaded_at": "2026-01-01T00:00:00+00:00",
        "path": path.as_posix(),
        "file_size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "job_id": "job-test",
        "api_client_version": "test",
        "request_status": "ok",
    }
    sidecar_manifest_path(path).write_text(json.dumps(payload), encoding="utf-8")


def test_build_plan_maps_legacy_dbn_and_definition_to_canonical_targets(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    target_root = tmp_path / "data" / "dbn"
    ohlcv = raw_root / "ES" / "2024.dbn.zst"
    definition = raw_root / "definition" / "ES" / "2024.dbn.zst"
    ohlcv.parent.mkdir(parents=True)
    definition.parent.mkdir(parents=True)
    ohlcv.write_bytes(b"ohlcv")
    definition.write_bytes(b"definition")
    write_manifest(ohlcv, schema="ohlcv-1m")
    write_manifest(definition, schema="definition")

    plan = build_plan(raw_root, target_root)

    assert plan["counts"] == {"total": 2, "plan_move": 2, "unsafe": 0, "skip_target_exists_same_hash": 0}
    items = {item["schema"]: item for item in plan["items"]}
    assert items["ohlcv-1m"]["target_path"].endswith(
        "data/dbn/ohlcv_1m/ES/2024/2024-01-01_2025-01-01.dbn.zst"
    )
    assert items["definition"]["target_path"].endswith(
        "data/dbn/definition/ES/2024/2024-01-01_2025-01-01.dbn.zst"
    )
    assert items["ohlcv-1m"]["manifest_path_update_required"] is True
    assert items["definition"]["manifest_path_update_required"] is True


def test_build_plan_reports_target_hash_collision(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    target_root = tmp_path / "data" / "dbn"
    source = raw_root / "ES" / "2024.dbn.zst"
    target = target_root / "ohlcv_1m" / "ES" / "2024" / "2024-01-01_2025-01-01.dbn.zst"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    target.write_bytes(b"different")
    write_manifest(source, schema="ohlcv-1m")

    plan = build_plan(raw_root, target_root)

    assert plan["counts"]["unsafe"] == 1
    item = plan["items"][0]
    assert item["action"] == "unsafe"
    assert "target already exists with different hash" in item["unsafe_reasons"]


def test_build_plan_reports_missing_manifest_and_hash_mismatch(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    target_root = tmp_path / "data" / "dbn"
    missing_manifest = raw_root / "ES" / "2024.dbn.zst"
    bad_hash = raw_root / "NQ" / "2024.dbn.zst"
    missing_manifest.parent.mkdir(parents=True)
    bad_hash.parent.mkdir(parents=True)
    missing_manifest.write_bytes(b"missing")
    bad_hash.write_bytes(b"before")
    write_manifest(bad_hash, schema="ohlcv-1m", market="NQ")
    bad_hash.write_bytes(b"after")

    plan = build_plan(raw_root, target_root)

    assert plan["counts"]["unsafe"] == 2
    reasons = {
        item["source_path"]: item["unsafe_reasons"]
        for item in plan["items"]
    }
    assert any("missing manifest" in reason for reason in reasons[missing_manifest.as_posix()])
    assert "manifest file_sha256 mismatch" in reasons[bad_hash.as_posix()]


def test_build_plan_reports_non_zstd_dbn_as_unsafe(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    target_root = tmp_path / "data" / "dbn"
    source = raw_root / "ES" / "2024.dbn"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"dbn")
    write_manifest(source, schema="ohlcv-1m")

    plan = build_plan(raw_root, target_root)

    assert plan["counts"]["unsafe"] == 1
    assert plan["items"][0]["action"] == "unsafe"
    assert "legacy DBN file is not .dbn.zst" in plan["items"][0]["unsafe_reasons"]
