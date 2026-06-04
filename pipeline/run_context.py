from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_id(seed: float | str | None = None) -> str:
    """Create the canonical run id used by diagnostics and child processes."""
    value = str(time.time() if seed is None else seed)
    return "run_" + hashlib.sha256(value.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class RunContext:
    run_start: float
    run_dt: datetime
    run_id: str
    profile: str
    symbols: tuple[str, ...]
    data_root: str | None = None

    @classmethod
    def create(cls, config: Any | None = None, *, profile: str | None = None, data_root: str | None = None) -> "RunContext":
        run_start = time.time()
        run_dt = datetime.now()
        resolved_profile = profile or getattr(config, "ACTIVE_PROFILE", None) or "unknown"
        symbols = tuple(getattr(config, "symbols", ()) or ())
        return cls(
            run_start=run_start,
            run_dt=run_dt,
            run_id=create_run_id(run_start),
            profile=str(resolved_profile),
            symbols=tuple(str(s) for s in symbols),
            data_root=data_root,
        )

    @property
    def file_ts(self) -> str:
        return self.run_dt.strftime("%Y-%m-%d_%H-%M-%S")

    def expected_rows(self, split_count: int) -> int:
        return len(self.symbols) * int(split_count)

    def child_env(self, base_env: dict[str, str] | None = None, *, config_env: str | None = None) -> dict[str, str]:
        env = dict(base_env or os.environ)
        env["PARENT_RUN_ID"] = self.run_id
        env["QUANT_RUN_ID"] = self.run_id
        if config_env:
            env["CONFIG_ENV"] = config_env
        return env

    def log_path(self, log_dir: str | Path = "output/logs") -> Path:
        return Path(log_dir) / f"{self.file_ts}_UNKNOWN_{self.run_id}.log"
