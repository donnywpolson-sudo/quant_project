from __future__ import annotations

import json
from pathlib import Path

from pipeline.common.io import canonical
from pipeline.data_gate.manifest import DEFAULT_MANIFEST_PATH
from scripts import validate_databento_continuous


def test_canonical_paths_match_phase1_layout() -> None:
    assert canonical.RAW_DATA_ROOT == Path("data/raw")
    assert canonical.VALIDATED_DATA_ROOT == Path("data/validated")
    assert canonical.SESSION_NORMALIZED_DATA_ROOT == Path("data/session_normalized")
    assert canonical.CAUSALLY_GATED_NORMALIZED_DATA_ROOT == Path("data/causally_gated_normalized")
    assert canonical.VALIDATION_REPORTS_ROOT == Path("reports/validation")
    assert canonical.SESSION_NORMALIZATION_REPORTS_ROOT == Path("reports/session_normalization")
    assert canonical.CAUSAL_GATING_REPORTS_ROOT == Path("reports/causal_gating")
    assert canonical.WFA_REPORTS_ROOT == Path("reports/wfa")
    assert canonical.METRICS_REPORTS_ROOT == Path("reports/metrics")
    assert canonical.MODELS_ARTIFACTS_ROOT == Path("artifacts/models")
    assert canonical.SCALERS_ARTIFACTS_ROOT == Path("artifacts/scalers")
    assert canonical.SELECTORS_ARTIFACTS_ROOT == Path("artifacts/selectors")
    assert canonical.RUN_MANIFESTS_ARTIFACTS_ROOT == Path("artifacts/run_manifests")
    assert canonical.BACKTESTS_ARTIFACTS_ROOT == Path("artifacts/backtests")


def test_report_defaults_point_to_reports_tree() -> None:
    assert validate_databento_continuous.DEFAULT_REPORT_OUT.as_posix().endswith("reports/validation")
    session_source = Path("scripts/session_normalize.py").read_text(encoding="utf-8")
    assert 'default="reports/session_normalization"' in session_source
    assert DEFAULT_MANIFEST_PATH == Path("reports/validation/audit_manifest.json")


def test_data_stage_roots_do_not_contain_report_csvs() -> None:
    stage_roots = [
        Path("data/raw"),
        Path("data/validated"),
        Path("data/session_normalized"),
        Path("data/causally_gated_normalized"),
    ]
    forbidden = {
        "core_summary.csv",
        "core_issues.csv",
        "session_summary.csv",
        "session_issues.csv",
        "reconciliation_raw_validated_normalized.csv",
        "unnormalized_validated_files.csv",
        "unvalidated_raw_files.csv",
        "_removed_rows.parquet",
        "_cleaning_manifest.csv",
    }
    offenders = [
        p.as_posix()
        for root in stage_roots
        if root.exists()
        for p in root.glob("*.csv")
        if p.name in forbidden
    ]
    assert offenders == []
    assert not (Path("data") / "validation_reports").exists()


def test_validated_manifest_matches_current_parquet_audit() -> None:
    root = Path("data/validated")
    parquets = sorted(root.glob("*/*.parquet")) if root.exists() else []
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["file_count"] == len(parquets)
        if not parquets:
            assert any("no validated parquet files" in x for x in manifest["blocking_issues"])


def test_alpha_tiered_is_only_active_config_reference() -> None:
    assert Path("configs/alpha_tiered.yaml").exists()
    assert not (Path("configs") / "alpha.yaml").exists()
    checked = [
        Path("README.md"),
        Path("project_layout.md"),
        Path("scripts/validate_databento_continuous.py"),
        Path("scripts/session_normalize.py"),
        Path("scripts/build_data_manifests.py"),
        Path("pipeline/common/config.py"),
    ]
    stale_config = "configs/" + "alpha.yaml"
    stale = [p.as_posix() for p in checked if stale_config in p.read_text(encoding="utf-8")]
    assert stale == []
