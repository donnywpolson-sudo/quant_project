#!/usr/bin/env python3
"""One-command safe pull from GitHub."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_REMOTE_URL = "https://github.com/donnywpolson-sudo/quant_project.git"


def run(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], text=True, capture_output=capture)
    if check and result.returncode:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)
    return result


def repo_root() -> Path:
    result = run(["rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def branch_name() -> str:
    result = run(["branch", "--show-current"])
    branch = result.stdout.strip()
    if not branch:
        print("STOP: detached HEAD. Switch to a branch first.")
        sys.exit(1)
    return branch


def ensure_clean_worktree() -> None:
    result = run(["status", "--porcelain=v1", "--untracked-files=all"])
    if result.stdout.strip():
        print("STOP: local changes exist. Push/commit/stash them before pulling.")
        print(result.stdout)
        sys.exit(1)


def normalize_remote_url(url: str) -> str:
    value = url.strip().lower()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    return value.removesuffix(".git")


def ensure_origin() -> None:
    result = run(["remote", "get-url", "origin"], check=False)
    if result.returncode or not result.stdout.strip():
        print("STOP: no origin remote configured.")
        sys.exit(1)
    origin = result.stdout.strip()
    if normalize_remote_url(origin) != normalize_remote_url(DEFAULT_REMOTE_URL):
        print(f"STOP: origin points somewhere unexpected: {origin}")
        print(f"Expected: {DEFAULT_REMOTE_URL}")
        sys.exit(1)
    print(f"Origin: {origin}")


def create_backup_branch(branch: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = f"backup/{branch}-{stamp}"
    run(["branch", backup, "HEAD"], capture=False)
    print(f"Backup branch created: {backup}")
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    root = repo_root()
    branch = branch_name()
    print(f"Repo: {root}")
    print(f"Branch: {branch}")
    ensure_origin()
    ensure_clean_worktree()
    create_backup_branch(branch)

    print("Fetching from GitHub...")
    run(["fetch", "origin"], capture=False)
    print("Pulling latest GitHub changes...")
    run(["pull", "--rebase", "origin", branch], capture=False)
    run(["status", "--short", "--branch"], capture=False)


if __name__ == "__main__":
    main()
