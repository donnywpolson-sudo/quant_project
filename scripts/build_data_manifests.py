from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.data_gate.manifest import build_data_manifest


STAGE_DIRS = {
    "raw": "raw",
    "validated": "validated",
    "session_normalized": "session_normalized",
    "causally_gated_normalized": "causally_gated_normalized",
    "labeled": "labeled",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=list(STAGE_DIRS))
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    for stage in args.stages:
        if stage not in STAGE_DIRS:
            raise SystemExit(f"unknown stage: {stage}")
        root = Path(args.data_root) / STAGE_DIRS[stage]
        root.mkdir(parents=True, exist_ok=True)
        report = build_data_manifest(root, stage=stage)
        print(f"{stage}: {len(report['files'])} files")


if __name__ == "__main__":
    main()
