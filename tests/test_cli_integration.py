import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl


REPO = Path(__file__).resolve().parents[1]


def _run(args, cwd=None, extra_env=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("CONFIG_ENV", "tier_0_smoke_pipeline")
    env.setdefault("QUANT_MODELING_MODE", "minimal_compatible")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "pipeline.cli", *args],
        cwd=cwd or REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_pipeline_cli_imports():
    import pipeline.cli  # noqa: F401


def test_pipeline_cli_help_returns_zero():
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr


def test_pipeline_cli_subcommand_help_returns_zero():
    for cmd in ["discover", "run", "aggregate"]:
        result = _run([cmd, "--help"])
        assert result.returncode == 0, result.stderr


def test_run_py_references_existing_pipeline_cli():
    assert (REPO / "pipeline" / "cli.py").exists()
    assert "pipeline.cli" in (REPO / "run.py").read_text(encoding="utf-8")


def test_minimal_synthetic_run_writes_audit_artifacts(tmp_path):
    data_dir = tmp_path / "data" / "ES"
    data_dir.mkdir(parents=True)
    path = data_dir / "2024.parquet"
    n = 40
    df = pl.DataFrame(
        {
            "ts_event": pl.datetime_range(
                pl.datetime(2024, 1, 1, 9, 30),
                pl.datetime(2024, 1, 1, 10, 9),
                "1m",
                eager=True,
            ),
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [100.2 + i * 0.1 for i in range(n)],
            "low": [99.8 + i * 0.1 for i in range(n)],
            "close": [100.05 + i * 0.1 for i in range(n)],
            "volume": [100 + i for i in range(n)],
        }
    )
    df.write_parquet(path)
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "out"
    discover = _run(["discover", "--data", str(path), "--out", str(manifest)], cwd=tmp_path)
    assert discover.returncode == 0, discover.stderr
    discovery_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert discovery_payload["discovery_data"]
    assert Path(discovery_payload["discovery_data"]).exists()
    assert discovery_payload["discovery_scope"] == "input_window"
    run = _run(["run", "--data", str(path), "--manifest", str(manifest), "--out", str(out)], cwd=tmp_path)
    assert run.returncode == 0, run.stderr

    assert (out / "backtest_results.parquet").exists()
    assert (out / "oos_predictions.parquet").exists()
    assert (out / "execution_trace_report.json").exists()
    assert list((tmp_path / "reports" / "metrics").glob("*_metrics_report.json"))
    assert list((tmp_path / "reports" / "leakage").glob("*.json"))
    assert list((tmp_path / "reports" / "stress").glob("*_stress_report.json"))
    assert list((tmp_path / "reports" / "acceptance").glob("*_acceptance_gate.json"))
    manifest_path = tmp_path / "artifacts" / "run_manifests" / "out.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["modeling_mode"] == "minimal_compatible"
    split = manifest_data["splits"][0]
    for key in ["backtest_results", "oos_predictions", "leakage_report", "execution_trace_report", "metrics_report", "stress_report", "acceptance_report"]:
        assert (tmp_path / split[key]).exists() if not Path(split[key]).is_absolute() else Path(split[key]).exists()


def test_synthetic_forbidden_future_feature_fails_before_modeling(tmp_path):
    path = tmp_path / "ES" / "2024.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts_event": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "open": [1.0] * 17,
            "high": [1.0] * 17,
            "low": [1.0] * 17,
            "close": [1.0] * 17,
            "volume": [1] * 17,
            "future_bad": [1.0] * 17,
        }
    ).write_parquet(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"selected_features": ["future_bad"]}), encoding="utf-8")
    result = _run(["run", "--data", str(path), "--manifest", str(manifest), "--out", str(tmp_path / "out")], cwd=tmp_path)
    assert result.returncode != 0
    assert "LEAKAGE FAIL" in result.stderr or "LEAKAGE FAIL" in result.stdout


def test_hard_fail_mode_preserves_reports_on_reject(tmp_path):
    path = tmp_path / "ES" / "2024.parquet"
    path.parent.mkdir(parents=True)
    n = 40
    pl.DataFrame(
        {
            "ts_event": list(range(n)),
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [100 + i for i in range(n)],
        }
    ).write_parquet(path)
    manifest = tmp_path / "manifest.json"
    discover = _run(["discover", "--data", str(path), "--out", str(manifest)], cwd=tmp_path)
    assert discover.returncode == 0, discover.stderr
    result = _run(
        ["run", "--data", str(path), "--manifest", str(manifest), "--out", str(tmp_path / "out")],
        cwd=tmp_path,
        extra_env={"QUANT_ACCEPTANCE_GATE_REQUIRED": "1"},
    )
    assert result.returncode != 0
    assert "ACCEPTANCE GATE REJECT" in result.stderr
    assert list((tmp_path / "reports" / "acceptance").glob("*_acceptance_gate.json"))
