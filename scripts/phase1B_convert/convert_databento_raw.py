#!/usr/bin/env python3
"""Phase 1B entry point for DBN archive to raw Parquet conversion."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase1A_download.download_databento_raw import main


def phase1b_main() -> int:
    if not any(arg == "--mode" or arg.startswith("--mode=") for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--mode", "convert-parquet"]
    return main()


if __name__ == "__main__":
    raise SystemExit(phase1b_main())
