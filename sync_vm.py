#!/usr/bin/env python3
"""
sync_vm.py

Safe GitHub sync script for your virtual machine / Codespaces environment.

This script refuses to run unless the machine matches the script:
- sync_local.py only runs on a machine marked as "local".
- sync_vm.py runs in GitHub Codespaces automatically, or on a machine marked as "vm".

One-time setup:
  Local machine:
    python sync_local.py --mark-local

  Regular VM, not Codespaces:
    python sync_vm.py --mark-vm

Codespaces:
  No setup needed. GitHub sets CODESPACES=true automatically.

What the sync does:
1. Shows your current branch and changed files.
2. If there are changes, asks whether to commit them.
3. Pulls latest GitHub changes with rebase.
4. Pushes your branch back to GitHub.

For a new GitHub repo with no origin remote yet:
  python sync_vm.py

If the new GitHub repo was initialized with a README/license commit:
  python sync_vm.py --bootstrap-new-remote

GitHub remains the source of truth.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


EXPECTED_MACHINE = "vm"
MACHINE_LABEL = "VIRTUAL MACHINE / CODESPACES"
DEFAULT_COMMIT_MESSAGE = "sync from virtual machine"
DEFAULT_REMOTE_URL = "https://github.com/donnywpolson-sudo/quant_project.git"
MARKER_PATH = Path.home() / ".quant_sync_machine"

# These are blocked from auto-commit unless you explicitly type YES.
# Edit this list if you intentionally track any of these file types.
RISKY_SUFFIXES = (
    ".env",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    ".sqlite",
    ".db",
    ".csv",
    ".parquet",
)

RISKY_DIR_PARTS = (
    "/data/",
    "/datasets/",
    "/logs/",
    "/output/",
    "/outputs/",
    "/backtests/",
    "/secrets/",
)


def is_codespaces() -> bool:
    return os.environ.get("CODESPACES", "").strip().lower() == "true"


def normalize_machine(value: str | None) -> str | None:
    if not value:
        return None

    value = value.strip().lower().replace("-", "_").replace(" ", "_")

    if value in {"local", "laptop", "desktop", "pc", "mac", "windows"}:
        return "local"

    if value in {
        "vm",
        "virtual",
        "virtual_machine",
        "remote",
        "codespace",
        "codespaces",
        "github_codespaces",
    }:
        return "vm"

    return None


def read_marker_file() -> str | None:
    if not MARKER_PATH.exists():
        return None

    raw_value = MARKER_PATH.read_text(encoding="utf-8").strip()
    machine = normalize_machine(raw_value)

    if machine not in {"local", "vm"}:
        print(f"ERROR: {MARKER_PATH} contains {raw_value!r}.")
        print("It should contain either: local or vm")
        sys.exit(1)

    return machine


def detect_machine() -> tuple[str | None, str]:
    # Codespaces should always behave like the VM side.
    if is_codespaces():
        return "vm", "CODESPACES=true"

    env_machine = normalize_machine(os.environ.get("QUANT_SYNC_MACHINE"))
    if env_machine in {"local", "vm"}:
        return env_machine, "QUANT_SYNC_MACHINE"

    marker_machine = read_marker_file()
    if marker_machine in {"local", "vm"}:
        return marker_machine, str(MARKER_PATH)

    return None, "no machine marker found"


def write_machine_marker(machine: str) -> None:
    MARKER_PATH.write_text(machine + "\n", encoding="utf-8")
    print(f"Saved machine marker: {MARKER_PATH}")
    print(f"This machine is now marked as: {machine}")


def clear_machine_marker() -> None:
    if MARKER_PATH.exists():
        MARKER_PATH.unlink()
        print(f"Removed machine marker: {MARKER_PATH}")
    else:
        print(f"No machine marker existed at: {MARKER_PATH}")


def handle_machine_commands(args: argparse.Namespace) -> None:
    if args.show_machine:
        detected, source = detect_machine()
        print(f"Detected machine: {detected or 'unknown'}")
        print(f"Source: {source}")
        print(f"Marker path: {MARKER_PATH}")
        sys.exit(0)

    if args.clear_machine_marker:
        clear_machine_marker()
        sys.exit(0)

    if args.mark_local:
        detected, source = detect_machine()
        if detected == "vm":
            print(f"STOP: This machine is already marked as vm ({source}).")
            print("Use sync_vm.py on this machine.")
            sys.exit(1)

        if is_codespaces():
            print("STOP: This is GitHub Codespaces, so it should not be marked local.")
            sys.exit(1)

        write_machine_marker("local")
        sys.exit(0)

    if args.mark_vm:
        detected, source = detect_machine()
        if detected == "local":
            print(f"STOP: This machine is already marked as local ({source}).")
            print("Do not mark your local machine as a VM.")
            sys.exit(1)

        write_machine_marker("vm")
        sys.exit(0)


def ensure_correct_machine() -> None:
    detected, source = detect_machine()

    if detected == EXPECTED_MACHINE:
        print(f"Machine check passed: {detected} ({source})")
        return

    script_name = Path(__file__).name

    if detected is None:
        print("\nSTOP: I do not know whether this is your local machine or your VM.")
        print(f"This script is: {script_name}")
        print(f"This script is only for: {EXPECTED_MACHINE}")
        print("\nRun the correct one-time setup command first:")
        print("  On your local machine:")
        print("    python sync_local.py --mark-local")
        print("\n  On a regular VM:")
        print("    python sync_vm.py --mark-vm")
        print("\n  In GitHub Codespaces:")
        print("    no setup is needed")
        sys.exit(1)

    correct_script = "sync_local.py" if detected == "local" else "sync_vm.py"

    print("\nSTOP: Wrong sync script for this machine.")
    print(f"This machine is marked as: {detected} ({source})")
    print(f"You ran: {script_name}")
    print(f"Use this instead: python {correct_script}")
    sys.exit(1)


def run_git(
    repo: Path,
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git command inside the repo."""
    cmd = ["git", *args]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo,
            text=True,
            capture_output=capture,
        )
    except FileNotFoundError:
        print("ERROR: Git is not installed or is not on your PATH.")
        sys.exit(127)

    if check and result.returncode != 0:
        print(f"\nERROR running: {' '.join(cmd)}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)

    return result


def try_repo_root(start: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print("ERROR: Git is not installed or is not on your PATH.")
        sys.exit(127)

    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())

    return None


def find_repo_root() -> Path:
    """Find the repo root from either the current directory or the script location."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent]

    for start in candidates:
        repo = try_repo_root(start)
        if repo is not None:
            return repo

    print("This script must be run from inside a Git repo, or placed inside the repo.")
    sys.exit(1)


def current_branch(repo: Path) -> str:
    result = run_git(repo, ["branch", "--show-current"])
    branch = result.stdout.strip()

    if not branch:
        print("You are in detached HEAD state. Switch to a normal branch first.")
        sys.exit(1)

    return branch


def status_lines(repo: Path) -> list[str]:
    result = run_git(repo, ["status", "--porcelain=v1"])
    return [line for line in result.stdout.splitlines() if line.strip()]


def changed_paths(lines: Iterable[str]) -> list[str]:
    paths: list[str] = []

    for line in lines:
        # Porcelain format: two status columns, space, then path.
        path = line[3:].strip()

        # Rename format: old_name -> new_name
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()

        paths.append(path)

    return paths


def looks_risky(path: str) -> bool:
    normalized = "/" + path.replace("\\", "/").lstrip("/")
    lower = normalized.lower()
    name = Path(path).name.lower()

    if name == ".env" or name.startswith(".env."):
        return True

    if any(name.endswith(suffix) for suffix in RISKY_SUFFIXES):
        return True

    if any(part in lower for part in RISKY_DIR_PARTS):
        return True

    return False


def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix} ").strip().lower()

    if not answer:
        return default

    return answer in {"y", "yes"}


def remote_branch_exists(repo: Path, branch: str) -> bool:
    result = run_git(repo, ["ls-remote", "--heads", "origin", branch], check=False)
    return bool(result.stdout.strip())


def branches_share_history(repo: Path, local_branch: str, remote_ref: str) -> bool:
    result = run_git(
        repo,
        ["merge-base", local_branch, remote_ref],
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def origin_remote_url(repo: Path) -> str | None:
    result = run_git(repo, ["remote", "get-url", "origin"], check=False)

    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    return None


def normalize_remote_url(url: str) -> str:
    value = url.strip().lower()

    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")

    return value.removesuffix(".git")


def same_remote_url(left: str, right: str) -> bool:
    return normalize_remote_url(left) == normalize_remote_url(right)


def ensure_origin_remote(repo: Path, remote_url: str | None) -> None:
    existing_url = origin_remote_url(repo)
    target_url = remote_url or DEFAULT_REMOTE_URL

    if existing_url:
        if remote_url and not same_remote_url(existing_url, remote_url):
            print(f"Origin exists: {existing_url}")
            print(f"Updating origin to: {remote_url}")
            run_git(repo, ["remote", "set-url", "origin", remote_url])
            return

        if not same_remote_url(existing_url, DEFAULT_REMOTE_URL):
            print(f"\nSTOP: origin points somewhere else: {existing_url}")
            print(f"Expected: {DEFAULT_REMOTE_URL}")
            print("If this is intentional, update it explicitly:")
            print(f"  python sync_vm.py --remote-url {DEFAULT_REMOTE_URL}")
            sys.exit(1)

        print(f"Origin: {existing_url}")
        return

    print(f"\nNo origin remote found. Adding origin: {target_url}")
    run_git(repo, ["remote", "add", "origin", target_url])


def print_changed_files(paths: list[str]) -> None:
    print("\nChanged files:")

    for path in paths:
        print(f"  - {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Safely sync this repo from the {MACHINE_LABEL}."
    )

    parser.add_argument(
        "-m",
        "--message",
        help="Commit message to use if changes are committed.",
    )
    parser.add_argument(
        "--skip-risk-check",
        action="store_true",
        help="Allow auto-commit even if data/secret-looking files changed.",
    )
    parser.add_argument(
        "--mark-local",
        action="store_true",
        help="One-time setup: mark this computer as your local machine.",
    )
    parser.add_argument(
        "--mark-vm",
        action="store_true",
        help="One-time setup: mark this computer as your VM.",
    )
    parser.add_argument(
        "--clear-machine-marker",
        action="store_true",
        help="Remove the local/VM marker from this computer.",
    )
    parser.add_argument(
        "--show-machine",
        action="store_true",
        help="Show what this computer is currently marked as.",
    )
    parser.add_argument(
        "--remote-url",
        help="For a new repo: add this URL as origin before syncing.",
    )
    parser.add_argument(
        "--bootstrap-new-remote",
        action="store_true",
        help=(
            "For a disposable new GitHub repo only: force-with-lease push this "
            "local branch over an unrelated remote initial commit."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handle_machine_commands(args)
    ensure_correct_machine()

    repo = find_repo_root()
    branch = current_branch(repo)

    print(f"\n=== {MACHINE_LABEL} REPO SYNC ===")
    print(f"Repo:   {repo}")
    print(f"Branch: {branch}")

    if branch == "main":
        print(
            "\nNote: You are on 'main'. This script works on main, "
            "but feature branches are safer."
        )

    ensure_origin_remote(repo, args.remote_url)

    print("\nFetching from GitHub...")
    run_git(repo, ["fetch", "origin"])

    lines = status_lines(repo)

    if lines:
        paths = changed_paths(lines)
        print_changed_files(paths)

        risky = [path for path in paths if looks_risky(path)]

        if risky and not args.skip_risk_check:
            print("\nThese files look like secrets, data, logs, databases, or outputs:")

            for path in risky:
                print(f"  - {path}")

            print("\nI will not auto-commit these unless you type YES.")
            answer = input("Type YES to continue, or press Enter to stop: ").strip()

            if answer != "YES":
                print(
                    "\nStopped. Add risky files to .gitignore or commit them "
                    "intentionally by hand."
                )
                sys.exit(1)

        if ask_yes_no("\nCommit all current changes now?", default=True):
            commit_message = (
                args.message
                or input(f"Commit message [{DEFAULT_COMMIT_MESSAGE}]: ").strip()
                or DEFAULT_COMMIT_MESSAGE
            )

            print("\nStaging all changes...")
            run_git(repo, ["add", "-A"])

            diff = run_git(repo, ["diff", "--cached", "--quiet"], check=False)

            if diff.returncode == 0:
                print("\nNo staged changes to commit.")
            else:
                print("\nCommitting changes...")
                run_git(repo, ["commit", "-m", commit_message], capture=False)
        else:
            print("\nStopped. Commit or stash your local changes before syncing.")
            sys.exit(0)
    else:
        print("\nNo local changes to commit.")

    if remote_branch_exists(repo, branch):
        remote_ref = f"origin/{branch}"

        if not branches_share_history(repo, branch, remote_ref):
            if args.bootstrap_new_remote:
                print(
                    f"\nBootstrapping new remote: replacing {remote_ref} "
                    f"with local {branch} using --force-with-lease."
                )
                run_git(
                    repo,
                    ["push", "--force-with-lease", "-u", "origin", branch],
                    capture=False,
                )
                print("\nFinal status:")
                run_git(repo, ["status", "--short"], capture=False)
                print("\nDone.")
                return

            print(f"\nSTOP: {branch} and {remote_ref} do not share history.")
            print("This usually means the new GitHub repo has an initial README/license commit.")
            print("If that remote commit is disposable, run:")
            print("  python sync_vm.py --bootstrap-new-remote")
            sys.exit(1)

        print(f"\nPulling latest changes from origin/{branch} using rebase...")
        pull = run_git(
            repo,
            ["pull", "--rebase", "origin", branch],
            check=False,
            capture=True,
        )

        if pull.returncode != 0:
            print("\nPull/rebase stopped, probably because of a conflict.")

            if pull.stdout:
                print(pull.stdout)

            if pull.stderr:
                print(pull.stderr)

            print("Resolve the conflict, then run:")
            print("  git status")
            print("  git rebase --continue")
            print("Then run this script again.")
            sys.exit(pull.returncode)

        if pull.stdout:
            print(pull.stdout.strip())
    else:
        print(f"\nNo remote branch origin/{branch} yet. Creating it on push.")

    print(f"\nPushing {branch} to GitHub...")
    run_git(repo, ["push", "-u", "origin", branch], capture=False)

    print("\nFinal status:")
    run_git(repo, ["status", "--short"], capture=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
