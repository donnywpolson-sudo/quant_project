from __future__ import annotations

from pathlib import Path

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows
from pipeline.data_gate.manifest import build_data_manifest


METADATA_PREFIXES = ("roll_", "continuous_", "front_contract", "back_contract")


def causal_gate_df(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    if "ts_event" in df.columns:
        ts_dtype = df["ts_event"].dtype
        exprs.append(pl.col("ts_event").alias("prediction_time"))
        if df["ts_event"].dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            exprs.append((pl.col("ts_event") + 1).cast(ts_dtype).alias("earliest_execution_time"))
        else:
            exprs.append((pl.col("ts_event") + pl.duration(minutes=1)).cast(ts_dtype).alias("earliest_execution_time"))
    for c in df.columns:
        if c.endswith("_available_at"):
            exprs.append((pl.col(c) <= pl.col("ts_event")).fill_null(False).alias(f"{c}_is_available"))
    out = df.with_columns(exprs) if exprs else df
    metadata = [c for c in out.columns if c.startswith(METADATA_PREFIXES) or c.endswith("_available_at")]
    return out.with_columns(pl.lit(",".join(metadata)).alias("non_model_metadata_columns"))


def causal_gate_root(in_root: str | Path = "data/session_normalized", out_root: str | Path = "data/causally_gated_normalized") -> dict:
    in_root = Path(in_root)
    out_root = Path(out_root)
    rows = []
    failures = []
    for p in sorted(in_root.glob("*/*.parquet")):
        try:
            out = out_root / p.parent.name / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            causal_gate_df(pl.read_parquet(p)).write_parquet(out)
            rows.append({"input": str(p), "output": str(out), "status": "PASS"})
        except Exception as exc:
            failures.append(str(p))
            rows.append({"input": str(p), "output": "", "status": "FAIL", "note": str(exc)})
    report = {"status": "FAIL" if failures else "PASS", "files": rows, "failures": failures}
    atomic_write_json("reports/causal_gating/causal_gating_report.json", report)
    write_csv_rows("reports/causal_gating/causal_gating_summary.csv", rows or [{"input": "", "output": "", "status": "WARN", "note": "no files"}])
    build_data_manifest(out_root, stage="causally_gated_normalized")
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in-root", default="data/session_normalized")
    p.add_argument("--out-root", default="data/causally_gated_normalized")
    args = p.parse_args()
    report = causal_gate_root(args.in_root, args.out_root)
    print(report["status"])


if __name__ == "__main__":
    main()
