#!/usr/bin/env python3
"""Reject generated artifacts in staged changes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath


BLOCKED_SUFFIXES = {".parquet", ".dbn", ".zst", ".pkl"}
BLOCKED_DIRS = {"reports", "logs", "cache"}


def is_blocked(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalized).parts
    return (
        PurePosixPath(normalized).suffix.lower() in BLOCKED_SUFFIXES
        or any(part in BLOCKED_DIRS for part in parts[:-1])
    )


def staged_paths() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str]) -> int:
    paths = argv or staged_paths()
    blocked = sorted(path for path in paths if is_blocked(path))
    if not blocked:
        return 0

    print("Generated artifacts must not be committed:")
    for path in blocked:
        print(f"  {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
