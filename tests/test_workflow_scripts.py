import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_direct_workflow_scripts_help_from_repo_root():
    for script in [
        "scripts/validate_databento_continuous.py",
        "scripts/build_data_manifests.py",
        "scripts/session_normalize.py",
        "scripts/causal_gate_normalized.py",
    ]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{script}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_session_normalize_default_points_to_merged_config():
    text = (REPO / "scripts" / "session_normalize.py").read_text(encoding="utf-8")
    assert 'default="configs/raw_data_validation.yaml"' in text
