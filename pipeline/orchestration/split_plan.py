from __future__ import annotations

from collections import Counter
from typing import Any


def split_window_parts(split_data: Any) -> tuple[list[int], list[int], Any, Any, Any, Any]:
    """Normalize legacy and day-based split tuple shapes."""
    train = split_data[0]
    test = split_data[1]
    train_start = split_data[2] if len(split_data) > 2 else None
    train_end = split_data[3] if len(split_data) > 3 else None
    test_start = split_data[4] if len(split_data) > 4 else None
    test_end = split_data[5] if len(split_data) > 5 else None
    return train, test, train_start, train_end, test_start, test_end


def assert_execution_plan_unique(plan_rows: list[dict], symbols: list[str], splits: list) -> None:
    expected = len(symbols) * len(splits)
    assert len(plan_rows) == expected, (
        f"SPLIT PLAN ROW COUNT FAIL: rows={len(plan_rows)} expected={expected} "
        f"symbols={symbols} splits={len(splits)}"
    )
    keys = [(r.get("symbol"), int(r.get("split", 0))) for r in plan_rows]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    assert len(keys) == len(set(keys)), f"SPLIT PLAN DUPLICATE FAIL: duplicate (symbol, split) rows={dupes}"
