from __future__ import annotations

from typing import Any

from pipeline.audit.pipeline_coverage import stage_catalog


START_STAGE_NUM = {
    "raw": 1,
    "validated": 5,
    "session_normalized": 7,
    "causally_gated_normalized": 9,
    "labeled": 11,
    "baseline_feature_matrix": 14,
    "expanded_feature_matrix": 21,
    "final_wfa": 24,
}


def normalize_start_stage(stage: str | None) -> str:
    raw = (stage or "raw").strip().lower().replace(" ", "_")
    aliases = {
        "causally_normalized": "causally_gated_normalized",
        "causally_normalised": "causally_gated_normalized",
        "casually_normalized": "causally_gated_normalized",
        "casually_gated_normalized": "causally_gated_normalized",
        "baseline_matrix": "baseline_feature_matrix",
        "expanded_matrix": "expanded_feature_matrix",
        "final": "final_wfa",
        "final_wfa_with_frozen_features": "final_wfa",
    }
    return aliases.get(raw, raw)


def build_stage_plan(start_stage: str, config: Any) -> list[dict]:
    start_stage = normalize_start_stage(start_stage)
    if start_stage not in START_STAGE_NUM:
        raise ValueError(f"unsupported start_stage={start_stage}")
    first_pending = START_STAGE_NUM[start_stage]
    checkpoint_root = getattr(getattr(config, "pipeline", object()), "checkpoint_root", None) or getattr(getattr(config, "data", object()), "root", None)
    rows = []
    for s in stage_catalog():
        rows.append({
            "stage_num": s.number,
            "stage_name": s.name,
            "stage_key": s.manifest_key,
            "status": "SKIPPED_CHECKPOINT" if s.number < first_pending else "PENDING",
            "input_root": checkpoint_root if s.number == first_pending else "",
            "output_root": "",
            "command": s.callable_or_command,
            "module": s.module_or_script,
            "report_paths": [p for p in s.output_paths if "reports/" in p or p.endswith("_report.json")],
            "skip_reason": f"start_stage={start_stage}" if s.number < first_pending else "",
        })
    return rows
