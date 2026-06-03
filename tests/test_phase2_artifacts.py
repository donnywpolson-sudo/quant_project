from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import polars as pl

from scripts.build_phase2_artifacts import (
    HORIZON_BARS,
    TARGET_COLS,
    add_session_safe_targets,
    build_one,
    feature_columns,
)


def _causal_df(lengths: tuple[int, ...] = (20, 20)) -> pl.DataFrame:
    rows = []
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    for seg, n in enumerate(lengths):
        for i in range(n):
            rows.append(
                {
                    "ts_event": ts,
                    "bar_start_ts": ts,
                    "bar_end_ts": ts + timedelta(minutes=1),
                    "prediction_ts": ts + timedelta(minutes=1),
                    "open": price,
                    "high": price + 0.25,
                    "low": price - 0.25,
                    "close": price,
                    "volume": 10 + i,
                    "dt_min": 1.0,
                    "is_contiguous_1m": i > 0,
                    "is_session_gap": i == 0,
                    "session_segment_id": seg,
                    "is_observed_bar": True,
                    "is_synthetic_empty_bar": False,
                    "has_causal_price": True,
                    "was_forward_filled": False,
                    "minutes_since_observed": 0.0,
                    "observed_bar_count_30m": i + 1,
                    "synthetic_bar_count_30m": 0,
                    "market": "ES",
                    "year": 2025,
                    "tick_size": 0.25,
                    "point_value": 50.0,
                    "contract_multiplier": 50.0,
                    "source_type": "continuous",
                    "continuous_method": "databento_continuous",
                    "adjustment_method": "none",
                    "roll_rule": "test",
                    "bar_timestamp": "start",
                    "bar_density_in": "sparse",
                    "eligible_for_features": True,
                    "eligible_for_entry": True,
                    "is_tradable_bar": True,
                    "causal_source_cutoff_ts": ts + timedelta(minutes=1),
                    "causally_gated": True,
                    "causal_gate_reason": "observed_bar_closed",
                    "ready_for_feature_discovery": True,
                    "ready_for_feature_expansion": True,
                    "ready_for_wfa": False,
                }
            )
            ts += timedelta(minutes=1)
            price += 1.0
        ts += timedelta(hours=1)
    return pl.DataFrame(rows)


def test_target_ret_15m_is_same_session_close_forward_return() -> None:
    out = add_session_safe_targets(_causal_df())
    expected = 115.0 / 100.0 - 1.0
    assert out["target_ret_15m"][0] == pytest.approx(expected)
    assert out["target_direction_15m"][0] == 1
    assert out["target_valid"][0] is True
    assert out["target_horizon_bars"][0] == HORIZON_BARS


def test_targets_do_not_cross_session_segments_and_last_15_invalid() -> None:
    out = add_session_safe_targets(_causal_df((20, 20)))
    for seg in (0, 1):
        part = out.filter(pl.col("session_segment_id") == seg)
        assert part.tail(HORIZON_BARS)["target_valid"].to_list() == [False] * HORIZON_BARS
        assert part.tail(HORIZON_BARS)["target_ret_15m"].null_count() == HORIZON_BARS
        assert part["target_valid"].sum() == 5


def test_target_columns_are_excluded_from_feature_cols(tmp_path: Path) -> None:
    src = tmp_path / "input" / "ES" / "2025.parquet"
    src.parent.mkdir(parents=True)
    _causal_df().write_parquet(src)

    rec = build_one(
        src,
        tmp_path / "data" / "labeled",
        tmp_path / "data" / "features_baseline",
        tmp_path / "data" / "feature_matrices" / "baseline",
    )
    matrix = pl.read_parquet(rec["feature_matrix_path"])
    features = feature_columns(matrix)
    assert set(features).isdisjoint(TARGET_COLS)
    assert set(TARGET_COLS).issubset(matrix.columns)
    assert {"market", "prediction_ts", "session_segment_id", "bar_index_in_session"}.issubset(matrix.columns)


def test_phase2_manifests_and_registries_are_written(tmp_path: Path) -> None:
    from scripts.build_phase2_artifacts import main
    import sys

    root = tmp_path / "data" / "causally_gated_normalized"
    src = root / "ES" / "2025.parquet"
    src.parent.mkdir(parents=True)
    _causal_df().write_parquet(src)

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_phase2_artifacts.py",
            "--root",
            str(root),
            "--labeled-root",
            str(tmp_path / "data" / "labeled"),
            "--features-root",
            str(tmp_path / "data" / "features_baseline"),
            "--matrix-root",
            str(tmp_path / "data" / "feature_matrices" / "baseline"),
            "--markets",
            "ES",
            "--years",
            "2025",
        ]
        main()
    finally:
        sys.argv = old_argv

    for rel in [
        "data/labeled/manifest.json",
        "data/labeled/_manifest.csv",
        "data/features_baseline/manifest.json",
        "data/features_baseline/_manifest.csv",
        "data/feature_matrices/baseline/manifest.json",
        "data/feature_matrices/baseline/_manifest.csv",
        "data/feature_matrices/baseline/feature_cols.json",
        "data/feature_matrices/baseline/target_cols.json",
        "data/feature_matrices/baseline/metadata_cols.json",
        "data/feature_matrices/baseline/excluded_cols.json",
    ]:
        assert (tmp_path / rel).exists()

    features = json.loads((tmp_path / "data/feature_matrices/baseline/feature_cols.json").read_text())
    targets = json.loads((tmp_path / "data/feature_matrices/baseline/target_cols.json").read_text())
    assert features
    assert set(features).isdisjoint(targets)
