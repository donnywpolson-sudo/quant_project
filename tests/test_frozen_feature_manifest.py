import json
from pathlib import Path

import polars as pl
import pytest

from pipeline.features.discovery import (
    _is_selectable_feature_name,
    apply_frozen_feature_manifest,
    load_frozen_feature_manifest,
)


def _write_manifest(path: Path, feature_names: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "manifest_type": "frozen_feature_manifest",
                "frozen": True,
                "feature_names": feature_names,
                "selected_K": len(feature_names),
                "discovery_status": "completed",
            }
        ),
        encoding="utf-8",
    )


def _matrix() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_event": [1, 2],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [100, 101],
            "session_id": ["a", "a"],
            "target_15m_dir": [1, 0],
            "target_15m_ret": [0.1, -0.1],
            "ret_1": [0.01, 0.02],
            "ratio_x": [1.0, 2.0],
            "continuous_price": [10.5, 11.5],
        }
    )


def test_apply_frozen_feature_manifest_keeps_only_selected_features(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, ["ret_1"])

    out = apply_frozen_feature_manifest(_matrix(), str(manifest), "target_15m_dir")

    assert "ret_1" in out.columns
    assert "ratio_x" not in out.columns
    assert "target_15m_dir" in out.columns
    assert "continuous_price" not in out.columns


def test_frozen_manifest_rejects_missing_selected_feature(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, ["feature_missing"])

    with pytest.raises(RuntimeError, match="selected features missing"):
        apply_frozen_feature_manifest(_matrix(), str(manifest), "target_15m_dir")


def test_frozen_manifest_rejects_target_as_feature(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, ["target_15m_ret"])

    with pytest.raises(RuntimeError, match="invalid selected features"):
        load_frozen_feature_manifest(str(manifest))


def test_discovery_feature_filter_rejects_continuous_metadata():
    rejected = [
        "continuous_price",
        "continuous_open",
        "continuous_high",
        "continuous_low",
        "continuous_close",
        "cumulative_factor",
    ]
    assert all(not _is_selectable_feature_name(c) for c in rejected)
    assert _is_selectable_feature_name("ret_1")
