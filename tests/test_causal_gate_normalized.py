from pathlib import Path

import pandas as pd
import pytest

from scripts.causal_gate_normalized import causal_gate_one


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="min")
    return pd.DataFrame(
        {
            "ts_event": ts,
            "open": [None, 100.0, 100.0, 101.0],
            "high": [None, 101.0, 100.0, 102.0],
            "low": [None, 99.0, 100.0, 100.0],
            "close": [None, 100.0, 100.0, 101.0],
            "volume": [0, 10, 0, 20],
            "dt_min": [None, 1.0, 1.0, 1.0],
            "is_contiguous_1m": [False, True, True, True],
            "is_session_gap": [False, False, False, False],
            "session_segment_id": [0, 0, 0, 0],
            "is_observed_bar": [False, True, False, True],
            "is_synthetic_empty_bar": [True, False, True, False],
            "has_causal_price": [False, True, True, True],
            "was_forward_filled": [False, False, True, False],
            "minutes_since_observed": [None, 0.0, 1.0, 0.0],
            "observed_bar_count_30m": [0, 1, 1, 2],
            "synthetic_bar_count_30m": [1, 1, 2, 2],
            "market": ["ES"] * 4,
            "year": [2024] * 4,
            "tick_size": [0.25] * 4,
            "point_value": [50.0] * 4,
            "contract_multiplier": [50.0] * 4,
            "source_type": ["continuous"] * 4,
            "continuous_method": ["databento_continuous"] * 4,
            "adjustment_method": ["none"] * 4,
            "roll_rule": ["databento_continuous_mapping"] * 4,
            "bar_timestamp": ["start"] * 4,
            "bar_density_in": ["sparse"] * 4,
            "eligible_for_features": [False, True, True, True],
            "eligible_for_entry": [False, True, False, True],
            "is_tradable_bar": [False, True, False, True],
        }
    )


def test_causal_gate_filters_noncausal_padding_and_marks_prediction_ts(tmp_path: Path) -> None:
    src = tmp_path / "ES" / "2024.parquet"
    dst = tmp_path / "out" / "ES" / "2024.parquet"
    src.parent.mkdir()
    _frame().to_parquet(src, index=False)

    rec = causal_gate_one(src, dst)
    out = pd.read_parquet(dst)

    assert rec["dropped_noncausal_padding_rows"] == 1
    assert len(out) == 3
    assert out["causally_gated"].all()
    assert out["prediction_ts"].equals(out["ts_event"] + pd.Timedelta(minutes=1))
    assert out["ready_for_feature_discovery"].all()
    assert not out["ready_for_wfa"].any()


def test_causal_gate_rejects_noncausal_row_after_observed(tmp_path: Path) -> None:
    src = tmp_path / "ES" / "2024.parquet"
    dst = tmp_path / "out" / "ES" / "2024.parquet"
    src.parent.mkdir()
    df = _frame()
    df.loc[2, "has_causal_price"] = False
    df.loc[2, "eligible_for_features"] = False
    df.to_parquet(src, index=False)

    with pytest.raises(RuntimeError, match="non-causal rows after observed"):
        causal_gate_one(src, dst)


def test_causal_gate_rejects_target_columns(tmp_path: Path) -> None:
    src = tmp_path / "ES" / "2024.parquet"
    dst = tmp_path / "out" / "ES" / "2024.parquet"
    src.parent.mkdir()
    df = _frame()
    df["target_15m_ret"] = 0.0
    df.to_parquet(src, index=False)

    with pytest.raises(RuntimeError, match="forbidden post-label/model columns"):
        causal_gate_one(src, dst)
