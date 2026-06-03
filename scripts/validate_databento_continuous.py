from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows
from pipeline.data_gate.manifest import build_data_manifest


REQUIRED = ["ts_event", "open", "high", "low", "close", "volume"]


def validate_raw_to_validated(raw_root: str | Path = "data/raw", validated_root: str | Path = "data/validated", *, write_validated: bool = False, clean_policy: str = "drop-invalid") -> dict:
    raw_root = Path(raw_root)
    validated_root = Path(validated_root)
    rows = []
    failures = []
    for p in sorted(raw_root.glob("*/*.parquet")):
        status = "PASS"
        note = ""
        try:
            df = pl.read_parquet(p)
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                status = "FAIL"
                note = f"missing columns: {missing}"
            elif df["ts_event"].n_unique() != df.height:
                status = "FAIL"
                note = "duplicate ts_event"
            elif {"open", "high", "low", "close"}.issubset(df.columns) and df.filter(
                (pl.col("high") < pl.col("low"))
                | (pl.col("open") > pl.col("high"))
                | (pl.col("open") < pl.col("low"))
                | (pl.col("close") > pl.col("high"))
                | (pl.col("close") < pl.col("low"))
            ).height:
                status = "FAIL"
                note = "invalid OHLC ordering"
            elif write_validated:
                out = validated_root / p.parent.name / p.name
                out.parent.mkdir(parents=True, exist_ok=True)
                clean = df.drop_nulls(["ts_event"]).sort("ts_event") if clean_policy == "drop-invalid" else df
                clean.write_parquet(out)
        except Exception as exc:
            status = "FAIL"
            note = str(exc)
        if status == "FAIL":
            failures.append(str(p))
        rows.append({"path": str(p), "market": p.parent.name, "year": p.stem, "status": status, "note": note})
    report = {"status": "FAIL" if failures else "PASS", "raw_root": str(raw_root), "validated_root": str(validated_root), "files": rows, "failures": failures}
    atomic_write_json("reports/validation/raw_validation_report.json", report)
    write_csv_rows("reports/validation/raw_validation_summary.csv", rows or [{"path": "", "market": "", "year": "", "status": "WARN", "note": "no raw files"}])
    if write_validated:
        build_data_manifest(validated_root, stage="validated")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--write-validated", action="store_true")
    p.add_argument("--clean-policy", choices=["drop-invalid", "none"], default="drop-invalid")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--validated-root", default="data/validated")
    args = p.parse_args()
    if not args.audit_only and not args.write_validated:
        args.audit_only = True
    report = validate_raw_to_validated(args.raw_root, args.validated_root, write_validated=args.write_validated, clean_policy=args.clean_policy)
    print(report["status"])


if __name__ == "__main__":
    main()
