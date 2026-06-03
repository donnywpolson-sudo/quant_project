from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from pipeline.data_gate.manifest import DatasetGateError, build_manifest, file_record, validate_dataset_gate, write_manifest


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_dataset_gate_passes_matching_manifest(tmp_path: Path) -> None:
    f = tmp_path / "ES" / "2024.parquet"
    _write(f, "abc")
    manifest = tmp_path / "audit_manifest.json"
    write_manifest([file_record(f, audit_status="PASS")], manifest)

    validate_dataset_gate([f], ["ES"], manifest)


def test_dataset_gate_fails_missing_record(tmp_path: Path) -> None:
    f = tmp_path / "ES" / "2024.parquet"
    _write(f, "abc")
    manifest = tmp_path / "audit_manifest.json"
    write_manifest([], manifest)

    with pytest.raises(DatasetGateError, match="missing_manifest_record"):
        validate_dataset_gate([f], ["ES"], manifest)


def test_dataset_gate_fails_if_file_changed(tmp_path: Path) -> None:
    f = tmp_path / "ES" / "2024.parquet"
    _write(f, "abc")
    manifest = tmp_path / "audit_manifest.json"
    write_manifest([file_record(f, audit_status="PASS")], manifest)
    time.sleep(0.01)
    _write(f, "changed")

    with pytest.raises(DatasetGateError, match="changed"):
        validate_dataset_gate([f], ["ES"], manifest)


def test_dataset_gate_fails_if_audit_not_pass(tmp_path: Path) -> None:
    f = tmp_path / "ES" / "2024.parquet"
    _write(f, "abc")
    manifest = tmp_path / "audit_manifest.json"
    write_manifest([file_record(f, audit_status="FAIL")], manifest)

    with pytest.raises(DatasetGateError, match="audit_not_pass"):
        validate_dataset_gate([f], ["ES"], manifest)


def test_dataset_gate_missing_manifest_can_be_non_required(tmp_path: Path) -> None:
    f = tmp_path / "ES" / "2024.parquet"
    _write(f, "abc")

    validate_dataset_gate([f], ["ES"], tmp_path / "missing.json", required=False)


def test_manifest_build_treats_audit_warn_as_pass(tmp_path: Path) -> None:
    f = tmp_path / "ES" / "2024.parquet"
    _write(f, "abc")
    audit_dir = tmp_path / "audit"
    _write(
        audit_dir / "core_summary.csv",
        "severity,market,year\nWARN,ES,2024\n",
    )
    manifest = tmp_path / "audit_manifest.json"

    build_manifest([f], audit_dir=audit_dir, out_path=manifest)

    validate_dataset_gate([f], ["ES"], manifest)
