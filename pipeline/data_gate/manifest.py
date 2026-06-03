from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows


class DatasetGateError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_data_manifest(root: str | Path, stage: str | None = None) -> dict[str, Any]:
    root = Path(root)
    rows = []
    for p in sorted(root.glob("*/*.parquet")):
        row = {"path": str(p.as_posix()), "market": p.parent.name, "year": p.stem, "bytes": p.stat().st_size, "sha256": _sha256(p)}
        try:
            row["rows"] = pl.scan_parquet(p).select(pl.len()).collect().item()
        except Exception:
            row["rows"] = None
        rows.append(row)
    payload = {"status": "PASS", "stage": stage or root.name, "root": str(root), "files": rows}
    atomic_write_json(root / "manifest.json", payload)
    write_csv_rows(root / "_manifest.csv", rows or [{"path": "", "market": "", "year": "", "bytes": 0, "sha256": "", "rows": 0}])
    return payload


def validate_dataset_gate(
    files: list[Path],
    *,
    symbols: list[str] | None = None,
    manifest_path: str | Path = "reports/validation/audit_manifest.json",
    required: bool = True,
    check_hash: bool = True,
) -> dict:
    rows = []
    failures = []
    allowed = set(symbols or [])
    for raw in files:
        p = Path(raw)
        status = "PASS"
        notes = []
        if not p.exists():
            status = "FAIL"
            notes.append("missing file")
        if allowed and p.parent.name not in allowed:
            status = "FAIL"
            notes.append(f"symbol {p.parent.name} not in configured symbols")
        sha = _sha256(p) if p.exists() and check_hash else ""
        try:
            nrows = pl.scan_parquet(p).select(pl.len()).collect().item() if p.exists() else 0
            if required and nrows <= 0:
                status = "FAIL"
                notes.append("empty parquet")
        except Exception as exc:
            status = "FAIL"
            nrows = 0
            notes.append(f"read failure: {exc}")
        if status == "FAIL":
            failures.append(f"{p}: {'; '.join(notes)}")
        rows.append({"path": str(p), "symbol": p.parent.name, "status": status, "rows": nrows, "sha256": sha, "notes": "; ".join(notes)})
    if required and not rows:
        failures.append("no dataset files supplied")
    report = {"status": "FAIL" if failures else "PASS", "files": rows, "failures": failures, "checks": ["exists", "configured_symbol", "readable", "non_empty", "sha256" if check_hash else "hash_skipped"]}
    atomic_write_json(manifest_path, report)
    write_csv_rows(Path(manifest_path).with_suffix(".csv"), rows or [{"path": "", "symbol": "", "status": "FAIL", "rows": 0, "sha256": "", "notes": "; ".join(failures)}])
    if failures and required:
        raise DatasetGateError("; ".join(failures))
    return report
