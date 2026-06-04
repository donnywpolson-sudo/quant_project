from __future__ import annotations

import os
import sys
from typing import Any


def build_child_env(base_env: dict[str, str] | None, *, run_id: str, profile: str) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["PARENT_RUN_ID"] = run_id
    env["QUANT_RUN_ID"] = run_id
    env["CONFIG_ENV"] = profile
    env["QUANT_RUN_PROFILE"] = profile
    env["TQDM_DISABLE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def build_discovery_command(train_glob: str, manifest_path: str, *, train_start=None, train_end=None) -> list[str]:
    cmd = [sys.executable, "-m", "pipeline.cli", "discover", "--data", train_glob, "--out", str(manifest_path)]
    if train_start and train_end:
        cmd.extend(["--start", _iso(train_start), "--end", _iso(train_end)])
    return cmd


def build_run_command(
    *,
    action: str,
    data_arg: str,
    manifest_path: str,
    out_dir: str,
    train_start=None,
    train_end=None,
    test_start=None,
    test_end=None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pipeline.cli",
        action,
        "--data",
        str(data_arg),
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_dir),
    ]
    if train_start and train_end:
        cmd.extend(["--train-start", _iso(train_start), "--train-end", _iso(train_end)])
    if test_start and test_end:
        cmd.extend(["--start", _iso(test_start), "--end", _iso(test_end)])
    return cmd
