from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pipeline.common.io_safe import atomic_write_json, write_csv_rows
from pipeline.common.cache import build_cache_metadata, cache_is_fresh, write_cache_metadata
from pipeline.data_gate.manifest import build_data_manifest


def add_labels(df: pl.DataFrame, *, horizon: int = 15, entry_lag_bars: int = 1, target_col: str = "target_15m_ret", target_scale_factor: float = 1.0, price_col: str = "open") -> pl.DataFrame:
    if price_col not in df.columns:
        price_col = "close"
    if price_col not in df.columns:
        raise ValueError("missing open/close for label generation")
    entry = int(entry_lag_bars)
    exit_lag = entry + int(horizon)
    return df.with_columns(
        ((pl.col(price_col).shift(-exit_lag) / pl.col(price_col).shift(-entry)).log() * float(target_scale_factor)).alias(target_col),
        pl.lit(entry_lag_bars).alias("label_entry_lag_bars"),
        pl.lit(horizon).alias("label_horizon_bars"),
        pl.lit(float(target_scale_factor)).alias("label_target_scale_factor"),
    )


def label_root(in_root: str | Path = "data/causally_gated_normalized", out_root: str | Path = "data/labeled", config: Any | None = None) -> dict:
    in_root = Path(in_root)
    out_root = Path(out_root)
    horizon = getattr(getattr(config, "target", object()), "target_15m_horizon", 15)
    scale = getattr(getattr(config, "target", object()), "target_scale_factor", 1.0)
    lag = getattr(getattr(config, "execution", object()), "entry_lag_bars", 1)
    rows = []
    src_manifest = in_root / "manifest.json"
    for p in sorted(in_root.glob("*/*.parquet")):
        out = out_root / p.parent.name / p.name
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = build_cache_metadata(
            out,
            source_stage="causally_gated_normalized",
            output_stage="labeled",
            source_paths=[src_manifest if src_manifest.exists() else p],
            config=config,
            config_sections=["target", "execution", "pipeline"],
            code_paths=[__file__],
            symbol=p.parent.name,
            year=p.stem,
        ) if config is not None else None
        fresh, reason = cache_is_fresh(out, meta, config) if meta else (False, "no config")
        if fresh:
            rows.append({"input": str(p), "output": str(out), "status": "COMPLETED_CACHED"})
            continue
        add_labels(pl.read_parquet(p), horizon=horizon, entry_lag_bars=lag, target_scale_factor=scale).write_parquet(out)
        if meta:
            write_cache_metadata(out, meta)
        rows.append({"input": str(p), "output": str(out), "status": "PASS"})
    report = {"status": "PASS", "files": rows}
    atomic_write_json("reports/validation/label_generation_report.json", report)
    write_csv_rows("reports/validation/label_generation_summary.csv", rows or [{"input": "", "output": "", "status": "WARN"}])
    build_data_manifest(out_root, stage="labeled")
    return report
