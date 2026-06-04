from __future__ import annotations

from pathlib import Path
from typing import Any


def build_final_safety_summary(
    *,
    run_id: str,
    profile: str,
    symbols: list[str],
    total_splits: int,
    artifact_rows: list[dict],
    deployment_report: dict,
) -> dict:
    counts = {"ACCEPT": 0, "REJECT": 0, "WARN": 0, "MISSING": 0}
    for row in artifact_rows:
        status = row.get("acceptance_status", "MISSING")
        counts[status if status in counts else "MISSING"] += 1
    successful = sum(1 for r in artifact_rows if r.get("status") == "OK")
    expected_rows = len(symbols) * int(total_splits)
    return {
        "run_id": run_id,
        "profile": profile,
        "symbols": list(symbols),
        "splits": total_splits,
        "expected_symbol_split_rows": expected_rows,
        "successful": successful,
        "failed": max(expected_rows - successful, 0),
        "total_reports_written": sum(
            1
            for row in artifact_rows
            for key in ["leakage_report", "execution_trace_report", "metrics_report", "stress_report", "acceptance_report"]
            if Path(row.get(key, "")).exists()
        ),
        "acceptance": counts,
        "deployment": deployment_report.get("status", "MISSING"),
        "deployment_mode": deployment_report.get("deployment", {}).get("mode"),
    }


def print_final_safety_summary(summary: dict) -> None:
    counts = summary["acceptance"]
    print("\n[FINAL SAFETY SUMMARY]", flush=True)
    print(f"run_id={summary['run_id']}", flush=True)
    print(f"profile={summary['profile']}", flush=True)
    print(f"symbols={','.join(summary['symbols'])}", flush=True)
    print(
        f"splits={summary['splits']} expected_rows={summary['expected_symbol_split_rows']} "
        f"successful={summary['successful']} failed={summary['failed']}",
        flush=True,
    )
    print(
        "acceptance: "
        f"ACCEPT={counts['ACCEPT']} REJECT={counts['REJECT']} WARN={counts['WARN']} MISSING={counts['MISSING']}",
        flush=True,
    )
    print(f"deployment={summary['deployment']} mode={summary['deployment_mode']}", flush=True)
