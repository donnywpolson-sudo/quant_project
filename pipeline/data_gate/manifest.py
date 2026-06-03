from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_MANIFEST_PATH = Path("reports/validation/audit_manifest.json")


class DatasetGateError(RuntimeError):
    pass


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def _market_from_path(path: Path) -> str:
    return path.parent.name


def _schema_summary(path: Path) -> dict:
    try:
        import polars as pl

        schema = pl.scan_parquet(path).collect_schema()
        stats = (
            pl.scan_parquet(path)
            .select(
                [
                    pl.len().alias("rows"),
                    pl.col("ts_event").min().alias("ts_min"),
                    pl.col("ts_event").max().alias("ts_max"),
                ]
            )
            .collect()
        )
        return {
            "columns": {k: str(v) for k, v in schema.items()},
            "schema_hash": hashlib.sha256(
                json.dumps({k: str(v) for k, v in schema.items()}, sort_keys=True).encode()
            ).hexdigest(),
            "rows": int(stats["rows"][0]),
            "ts_min": str(stats["ts_min"][0]),
            "ts_max": str(stats["ts_max"][0]),
        }
    except Exception as exc:
        return {"schema_error": repr(exc)}


def file_record(path: str | Path, *, audit_status: str = "UNKNOWN", audit_source: str = "") -> dict:
    p = Path(path)
    st = p.stat()
    rec = {
        "path": _norm_path(p),
        "market": _market_from_path(p),
        "file_name": p.name,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "sha256": _sha256_file(p),
        "audit_status": str(audit_status).upper(),
        "audit_source": audit_source,
    }
    rec.update(_schema_summary(p))
    return rec


def write_manifest(records: list[dict], out_path: str | Path = DEFAULT_MANIFEST_PATH) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": sorted(records, key=lambda r: r["path"]),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    return out


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise DatasetGateError(f"DATASET GATE FAIL: audit manifest missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _audit_status_lookup(audit_dir: Path) -> dict[tuple[str, str], str]:
    """Return {(market, file_name): PASS|FAIL|WARN|UNKNOWN} from known audit CSVs."""
    out: dict[tuple[str, str], str] = {}

    # L0 report shape: market/year in core_summary.csv + session_summary.csv.
    for name in ("core_summary.csv", "session_summary.csv"):
        for row in _read_csv_rows(audit_dir / name):
            market = str(row.get("market", ""))
            year = str(row.get("year", ""))
            severity = str(row.get("severity", "UNKNOWN")).upper()
            if market and year:
                key = (market, f"{year}.parquet")
                prev = out.get(key, "PASS")
                out[key] = "FAIL" if "FAIL" in (prev, severity) else "PASS"

    # L1 report shape: market/file in summary.csv.
    for row in _read_csv_rows(audit_dir / "summary.csv"):
        market = str(row.get("market", ""))
        file_name = str(row.get("file", ""))
        severity = str(row.get("severity", "UNKNOWN")).upper()
        if market and file_name:
            out[(market, file_name)] = "FAIL" if severity == "FAIL" else "PASS"

    return out


def build_manifest(
    files: Iterable[str | Path],
    *,
    audit_dir: str | Path,
    out_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    audit_dir = Path(audit_dir)
    statuses = _audit_status_lookup(audit_dir)
    records = []
    for f in files:
        p = Path(f)
        key = (_market_from_path(p), p.name)
        records.append(
            file_record(
                p,
                audit_status=statuses.get(key, "UNKNOWN"),
                audit_source=str(audit_dir),
            )
        )
    return write_manifest(records, out_path)


def validate_dataset_gate(
    files: Iterable[str | Path],
    symbols: Iterable[str] | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    required: bool = True,
    check_hash: bool = True,
) -> None:
    files = list(files)
    path = Path(manifest_path)
    if not path.exists():
        if required:
            raise DatasetGateError(f"DATASET GATE FAIL: audit manifest missing: {path}")
        print(f"[DATASET-GATE] WARN audit manifest missing; gate skipped: {path}", flush=True)
        return

    manifest = load_manifest(path)
    by_path = {str(r.get("path")): r for r in manifest.get("files", [])}
    allowed = {s.upper() for s in symbols or []}
    failures: list[str] = []

    for raw in files:
        p = Path(raw)
        market = _market_from_path(p).upper()
        if allowed and market not in allowed:
            continue
        key = _norm_path(p)
        rec = by_path.get(key)
        if rec is None:
            failures.append(f"missing_manifest_record:{p}")
            continue
        if str(rec.get("audit_status", "")).upper() != "PASS":
            failures.append(f"audit_not_pass:{p}:{rec.get('audit_status')}")
        if not p.exists():
            failures.append(f"file_missing:{p}")
            continue
        st = p.stat()
        if int(rec.get("size", -1)) != int(st.st_size):
            failures.append(f"size_changed:{p}")
        if int(rec.get("mtime_ns", -1)) != int(st.st_mtime_ns):
            failures.append(f"mtime_changed:{p}")
        if check_hash and str(rec.get("sha256")) != _sha256_file(p):
            failures.append(f"hash_changed:{p}")

    if failures:
        sample = "; ".join(failures[:10])
        raise DatasetGateError(f"DATASET GATE FAIL: {len(failures)} issue(s): {sample}")

    print(f"[DATASET-GATE] PASS files={len(files)} manifest={path}", flush=True)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Build/validate dataset audit manifest.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--audit-dir", required=True)
    b.add_argument("--out", default=str(DEFAULT_MANIFEST_PATH))
    b.add_argument("--pattern", default="*.parquet")
    v = sub.add_parser("validate")
    v.add_argument("--root", required=True)
    v.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    v.add_argument("--pattern", default="*.parquet")
    v.add_argument("--no-hash", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.root).rglob(args.pattern))
    if args.cmd == "build":
        out = build_manifest(files, audit_dir=args.audit_dir, out_path=args.out)
        print(f"Wrote audit manifest: {out}")
    elif args.cmd == "validate":
        validate_dataset_gate(files, manifest_path=args.manifest, check_hash=not args.no_hash)


if __name__ == "__main__":
    _main()
