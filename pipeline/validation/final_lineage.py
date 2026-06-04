from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lineage_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".lineage.json")


def write_lineage(
    artifact_path: str | Path,
    *,
    run_id: str,
    profile: str,
    source_stage: str,
    source_artifact_path: str | Path,
    frozen_feature_manifest_path: str | Path = "data/frozen_features/phase5_v1/manifest.json",
    selected_feature_count: int | str = "",
    expected_symbols: list[str] | None = None,
    expected_splits: int | str = "",
    expected_rows: int | str = "",
    actual_rows: int | str = "",
    producing_code_version: str = "",
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    source = Path(source_artifact_path)
    frozen = Path(frozen_feature_manifest_path)
    payload = {
        "run_id": str(run_id),
        "profile": str(profile),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stage": source_stage,
        "source_artifact_path": str(source),
        "source_artifact_checksum": file_sha256(source) if source.exists() else "",
        "frozen_feature_manifest_hash": file_sha256(frozen) if frozen.exists() else "",
        "selected_feature_count": selected_feature_count,
        "expected_symbols": expected_symbols or [],
        "expected_splits": expected_splits,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "producing_code_version": producing_code_version,
    }
    out = lineage_path(artifact)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def read_json_or_sidecar(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    candidates = [p, lineage_path(p)]
    merged: dict[str, Any] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict):
            merged.update(raw)
    return merged


def validate_lineage_freshness(
    *,
    artifact_path: str | Path,
    source_artifact_path: str | Path,
    stage_name: str,
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    source = Path(source_artifact_path)
    if not artifact.exists():
        return {"status": "MISSING", "reason": f"artifact missing: {artifact}"}
    if not source.exists():
        return {"status": "MISSING", "reason": f"source artifact missing: {source}"}
    meta = read_json_or_sidecar(artifact)
    required = ["run_id", "profile", "created_at", "source_artifact_path", "source_artifact_checksum"]
    missing = [k for k in required if not meta.get(k)]
    if missing:
        return {"status": "STALE", "reason": f"{stage_name} missing lineage fields: {','.join(missing)}"}
    if Path(str(meta.get("source_artifact_path"))).as_posix() != source.as_posix():
        return {
            "status": "STALE",
            "reason": f"{stage_name} source path mismatch: recorded={meta.get('source_artifact_path')} expected={source}",
        }
    checksum = file_sha256(source)
    if str(meta.get("source_artifact_checksum")) != checksum:
        return {"status": "STALE", "reason": f"{stage_name} source checksum mismatch"}
    created_at = _parse_dt(meta.get("created_at"))
    source_created = _source_created_at(source)
    if created_at and source_created and created_at < source_created:
        return {"status": "STALE", "reason": f"{stage_name} created_at older than source artifact"}
    source_meta = read_json_or_sidecar(source)
    for key in ["run_id", "profile"]:
        if source_meta.get(key) and str(meta.get(key)) != str(source_meta.get(key)):
            return {"status": "STALE", "reason": f"{stage_name} {key} mismatch with source"}
    return {"status": "PASS", "reason": "ok", "lineage": meta}


def _source_created_at(path: Path):
    meta = read_json_or_sidecar(path)
    return _parse_dt(meta.get("created_at") or meta.get("created_at_utc"))


def _parse_dt(value: Any):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
