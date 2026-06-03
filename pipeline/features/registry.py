from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipeline.common.io_safe import atomic_write_json
from pipeline.common.cache import build_cache_metadata, cache_is_fresh, write_cache_metadata


TARGET_PREFIXES = ("target_", "label_")
METADATA = {
    "ts_event",
    "date",
    "session",
    "session_id",
    "session_date",
    "symbol",
    "market",
    "session_timezone",
    "session_calendar_accuracy",
    "rtype",
    "publisher_id",
    "instrument_id",
    "prediction_time",
    "earliest_execution_time",
    "execution_time",
    "non_model_metadata_columns",
    "open",
    "high",
    "low",
    "close",
    "volume",
}
FORBIDDEN_PREFIXES = ("future_", "roll_", "continuous_", "front_contract", "back_contract")


def build_column_registry(df: pl.DataFrame, source_stage: str = "") -> dict:
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIXES)]
    forbidden = [c for c in df.columns if c.startswith(FORBIDDEN_PREFIXES)]
    availability = [c for c in df.columns if c.endswith("_available_at")]
    metadata = sorted(set([c for c in df.columns if c in METADATA or c.endswith("_is_available")] + availability + forbidden))
    feature_cols = [
        c for c, dtype in zip(df.columns, df.dtypes)
        if dtype.is_numeric() and c not in target_cols and c not in metadata and not c.startswith(FORBIDDEN_PREFIXES)
    ]
    schema = "|".join(f"{c}:{t}" for c, t in zip(df.columns, df.dtypes))
    return {
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "metadata_columns": metadata,
        "forbidden_model_columns": sorted(set(target_cols + forbidden + metadata)),
        "availability_timestamp_columns": availability,
        "source_stage": source_stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_hash": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
    }


def write_column_registry(df: pl.DataFrame, path: str | Path, source_stage: str = "", config=None, source_paths: list[str | Path] | None = None) -> dict:
    if config is not None:
        meta = build_cache_metadata(
            path,
            source_stage=source_stage,
            output_stage="column_registry",
            source_paths=source_paths or [],
            config=config,
            config_sections=["features", "target", "pipeline"],
            code_paths=[__file__],
        )
        fresh, _ = cache_is_fresh(path, meta, config)
        if fresh:
            import json
            return json.loads(Path(path).read_text(encoding="utf-8"))
    registry = build_column_registry(df, source_stage)
    atomic_write_json(path, registry)
    if config is not None:
        write_cache_metadata(path, meta)
    return registry
