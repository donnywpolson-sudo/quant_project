from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from pipeline.common.config import config as flat_config, load_config
from pipeline.common.io_safe import atomic_write_json
from pipeline.session.normalize import SESSION_KEYS, load_session_config


def _session_sections(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections: dict[str, dict[str, Any]] = {}
    for market in sorted((raw.get("markets") or {}).keys()):
        sections[market] = load_session_config(path, market)
    for name, section in (raw.get("profiles") or {}).items():
        sections[f"profile:{name}"] = {k: v for k, v in (section or {}).items() if k in SESSION_KEYS}
    if not sections and "default" in raw:
        sections["default"] = load_session_config(path, "")
    return sections


def run_session_roll_audit(config_path: str = "configs/raw_data_validation.yaml", out: str = "reports/session_normalization/session_roll_audit.json") -> dict[str, Any]:
    checks = []
    failures = []
    path = Path(config_path)
    if not path.exists():
        failures.append(f"missing validation/session config: {path}")
    sections = _session_sections(path)
    if not sections and path.exists():
        failures.append(f"no session sections found in {path}")
    for name, section in sections.items():
        for field in ["timezone", "week_start_day", "week_start_time", "week_end_day", "week_end_time"]:
            if field not in section:
                failures.append(f"{name}: missing {field}")
        if "daily_break" not in section and "daily_breaks" not in section:
            checks.append({"name": f"{name}_daily_break", "status": "WARN", "message": "no daily break configured"})
    root_cfg = load_config()
    roll = root_cfg.roll_policy if root_cfg is not None else None
    roll_dict = roll.model_dump() if roll is not None else getattr(flat_config, "ROLL_POLICY", {})
    method = roll_dict.get("method")
    adjustment = roll_dict.get("adjustment")
    if method not in {"volume_or_days_before_expiry", "days_before_expiry"}:
        failures.append(f"unsupported roll_policy.method={method}")
    if adjustment not in {"back_adjusted", "none", "ratio_adjusted"}:
        failures.append(f"unsupported roll_policy.adjustment={adjustment}")
    report = {
        "status": "FAIL" if failures else "PASS",
        "checks": checks,
        "failures": failures,
        "config_path": str(path),
        "session_sections": sorted(sections.keys()),
        "roll_policy": roll_dict,
    }
    atomic_write_json(out, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/raw_data_validation.yaml")
    p.add_argument("--out", default="reports/session_normalization/session_roll_audit.json")
    args = p.parse_args()
    print(run_session_roll_audit(args.config, args.out)["status"])


if __name__ == "__main__":
    main()
