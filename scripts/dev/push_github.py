#!/usr/bin/env python3
"""One-command safe push to GitHub."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_MESSAGE = "sync updates"
DEFAULT_REMOTE_URL = "https://github.com/donnywpolson-sudo/quant_project.git"
RISKY_SUFFIXES = (
    ".pem",
    ".key",
    ".parquet",
    ".dbn",
    ".zst",
    ".pkl",
    ".joblib",
    ".npy",
    ".npz",
    ".duckdb",
    ".sqlite",
)
RISKY_FILENAMES = {
    ".env",
    ".env.local",
    ".envrc",
    "credentials.json",
    "secrets.json",
}
RISKY_DIRS = (
    "/cache/",
    "/data/",
    "/reports/",
    "/logs/",
    "/models/",
    "/artifacts/",
)


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


def status_lines() -> list[str]:
    result = run(["status", "--porcelain=v1", "--untracked-files=all"])
    return [line for line in result.stdout.splitlines() if line.strip()]


def changed_paths(lines: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for line in lines:
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        paths.append(path)
    return paths


def looks_risky(path: str) -> bool:
    normalized = "/" + path.replace("\\", "/").lstrip("/").lower()
    name = Path(path).name.lower()
    return (
        name in RISKY_FILENAMES
        or any(name.endswith(suffix) for suffix in RISKY_SUFFIXES)
        or any(marker in normalized for marker in RISKY_DIRS)
    )


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


def normalize_remote_url(url: str) -> str:
    value = url.strip().lower()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    return value.removesuffix(".git")


def create_backup_branch(branch: str, label: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = f"backup/{branch}-{label}-{stamp}"
    run(["branch", backup, "HEAD"], capture=False)
    print(f"Backup branch created: {backup}")
    return backup


def run_tests(skip_tests: bool) -> None:
    if skip_tests:
        print("Skipping tests.")
        return
    print("Running tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], text=True)
    if result.returncode:
        print("STOP: tests failed. Not pushing.")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    branch = branch_name()
    print(f"Repo: {root}")
    print(f"Branch: {branch}")
    ensure_origin()

    lines = status_lines()
    paths = changed_paths(lines)
    if paths:
        print("Changed files:")
        for path in paths:
            print(f"  - {path}")
        risky = [path for path in paths if looks_risky(path)]
        if risky:
            print("STOP: risky data/secret/output files detected:")
            for path in risky:
                print(f"  - {path}")
            print("Add them to .gitignore or remove them before pushing.")
            sys.exit(1)
    else:
        print("No local changes.")

    run_tests(args.skip_tests)

    if paths:
        create_backup_branch(branch, "before-commit")
        run(["add", "-A"], capture=False)
        if run(["diff", "--cached", "--quiet"], check=False).returncode:
            run(["commit", "-m", args.message], capture=False)

    create_backup_branch(branch, "before-rebase")

    print("Pulling latest GitHub changes...")
    pull = run(["pull", "--rebase", "origin", branch], check=False)
    if pull.returncode:
        print(pull.stdout)
        print(pull.stderr)
        print("STOP: pull/rebase failed. Resolve conflicts, then rerun.")
        sys.exit(pull.returncode)
    if pull.stdout.strip():
        print(pull.stdout.strip())

    print("Pushing to GitHub...")
    run(["push", "-u", "origin", branch], capture=False)
    run(["status", "--short", "--branch"], capture=False)


if __name__ == "__main__":
    main()
