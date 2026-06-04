from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


DIAGNOSTIC_STRING_FIELDS = {
    "run_id",
    "profile",
    "config_env",
    "symbol",
    "split",
    "threshold_type",
    "threshold_mode",
    "acceptance_status",
    "rejection_reason",
}
STRING_KEY_FIELDS = DIAGNOSTIC_STRING_FIELDS


def stringify_key_fields(row: dict, key_fields: Iterable[str] = STRING_KEY_FIELDS) -> dict:
    out = dict(row)
    for key in key_fields:
        if key in out and out[key] is not None:
            out[key] = str(out[key])
    return out


def stringify_diagnostic_keys(row: dict, key_fields: Iterable[str] = STRING_KEY_FIELDS) -> dict:
    return stringify_key_fields(row, key_fields)


def write_csv_json(
    rows: list[dict],
    *,
    csv_path: str | Path,
    json_path: str | Path,
    fields: list[str],
    key_fields: Iterable[str] = STRING_KEY_FIELDS,
) -> tuple[Path, Path]:
    csv_out = Path(csv_path)
    json_out = Path(json_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    safe_rows = [stringify_key_fields(row, key_fields) for row in rows]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in safe_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    json_out.write_text(json.dumps(safe_rows, indent=2, default=str), encoding="utf-8")
    return csv_out, json_out


def read_json_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def read_diagnostic_csv(path: str | Path):
    import pandas as pd

    return pd.read_csv(path, dtype=str, keep_default_na=False)
