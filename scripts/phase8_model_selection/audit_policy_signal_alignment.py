#!/usr/bin/env python3
"""Audit Phase 8 policy direction, gate, and signal alignment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.phase8_model_selection.audit_policy_failure import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREDICTIONS,
    DEFAULT_RUN,
    _relative_path,
    _write_csv,
    _write_json,
)
from scripts.phase8_model_selection.audit_trade_failure_drilldown import _empty_unavailable
from scripts.phase8_model_selection.evaluate_predictions import (
    DEFAULT_COSTS_CONFIG,
    PolicyConfig,
    build_policy_frame,
)


OUTPUT_SUFFIXES = {
    "summary": "policy_signal_alignment_summary.json",
    "overall": "policy_signal_alignment_overall.csv",
    "market_side": "policy_signal_alignment_by_market_side.csv",
    "market_fold": "policy_signal_alignment_by_market_fold.csv",
    "hour_side": "policy_signal_alignment_by_hour_side.csv",
    "signal_buckets": "policy_signal_alignment_by_signal_bucket.csv",
    "gate_effect": "policy_signal_gate_effect.csv",
    "inversion_check": "policy_signal_inversion_check.csv",
    "readme": "policy_signal_alignment_readme.md",
}


def _output_paths(output_root: Path, run: str) -> dict[str, Path]:
    return {key: output_root / f"{run}_{suffix}" for key, suffix in OUTPUT_SUFFIXES.items()}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _sum_float(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _rate_or_none(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def _direction_accuracy(position: pd.Series, actual: pd.Series) -> float | None:
    aligned = pd.DataFrame({"position": position, "actual": actual}).dropna()
    aligned = aligned[aligned["position"].ne(0) & aligned["actual"].isin([-1, 0, 1])]
    if aligned.empty:
        return None
    return float(aligned["position"].eq(aligned["actual"]).mean())


def _inverted_rate(position: pd.Series, actual: pd.Series) -> float | None:
    aligned = pd.DataFrame({"position": position, "actual": actual}).dropna()
    aligned = aligned[aligned["position"].ne(0) & aligned["actual"].isin([-1, 1])]
    if aligned.empty:
        return None
    return float(aligned["position"].eq(-aligned["actual"]).mean())


def _argmax_direction_accuracy(frame: pd.DataFrame) -> float | None:
    p_long = _numeric(frame, "p_long")
    p_short = _numeric(frame, "p_short")
    p_flat = _numeric(frame, "p_flat")
    if p_long.isna().all() or p_short.isna().all() or p_flat.isna().all():
        return None
    stacked = np.vstack(
        [
            p_short.fillna(-np.inf).to_numpy(),
            p_flat.fillna(-np.inf).to_numpy(),
            p_long.fillna(-np.inf).to_numpy(),
        ]
    ).T
    predicted = np.array([-1, 0, 1])[np.argmax(stacked, axis=1)]
    actual = _numeric(frame, "target_direction").to_numpy()
    mask = np.isfinite(actual)
    if not bool(mask.any()):
        return None
    return float((predicted[mask] == actual[mask]).mean())


def _realized_direction(frame: pd.DataFrame) -> pd.Series:
    move = _numeric(frame, "execution_close") - _numeric(frame, "execution_open")
    return pd.Series(np.sign(move.to_numpy(dtype=float)), index=frame.index, dtype=float)


def _bucket(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels).astype("string").fillna("missing")


def _attach_alignment_fields(policy_frame: pd.DataFrame) -> pd.DataFrame:
    frame = policy_frame.copy()
    frame["side"] = frame["position"].map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    frame["base_side"] = frame["base_position"].map({-1: "short", 0: "flat", 1: "long"}).fillna("unknown")
    frame["realized_direction"] = _realized_direction(frame)
    frame["target_direction"] = _numeric(frame, "observed_direction_target")
    frame["abs_direction_margin"] = _numeric(frame, "direction_margin").abs()
    frame["signal_margin_bucket"] = _bucket(
        frame["abs_direction_margin"],
        [-np.inf, 0.05, 0.10, 0.20, 0.40, np.inf],
        ["<=0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">0.40"],
    )
    frame["p_long_bucket"] = _bucket(
        _numeric(frame, "p_long"),
        [-np.inf, 0.25, 0.40, 0.55, 0.70, 0.85, np.inf],
        ["<=0.25", "0.25-0.40", "0.40-0.55", "0.55-0.70", "0.70-0.85", ">0.85"],
    )
    frame["p_short_bucket"] = _bucket(
        _numeric(frame, "p_short"),
        [-np.inf, 0.25, 0.40, 0.55, 0.70, 0.85, np.inf],
        ["<=0.25", "0.25-0.40", "0.40-0.55", "0.55-0.70", "0.70-0.85", ">0.85"],
    )
    frame["p_fade_bucket"] = _bucket(
        _numeric(frame, "p_fade_success"),
        [-np.inf, 0.50, 0.70, 0.85, 0.95, np.inf],
        ["<0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", ">0.95"],
    )
    frame["p_trend_bucket"] = _bucket(
        _numeric(frame, "p_trend_danger"),
        [-np.inf, 0.10, 0.25, 0.50, 0.75, np.inf],
        ["<0.10", "0.10-0.25", "0.25-0.50", "0.50-0.75", ">0.75"],
    )
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["hour_utc"] = timestamp.dt.hour
    frame["gate_category"] = "other"
    frame.loc[frame["position"].ne(0), "gate_category"] = "traded"
    frame.loc[frame["no_direction_signal"], "gate_category"] = "no_direction"
    frame.loc[frame["blocked_by_fade_filter"], "gate_category"] = "fade_filter_block"
    frame.loc[frame["blocked_by_trend_danger"], "gate_category"] = "trend_danger_block"
    frame["position_matches_base_signal"] = frame["position"].eq(frame["base_position"]) | frame[
        "position"
    ].eq(0)
    frame["base_gross_dollars"] = (
        frame["base_position"] * _numeric(frame, "price_move").fillna(0.0) * _numeric(frame, "point_value").fillna(0.0)
    )
    frame["base_cost_dollars"] = np.where(
        frame["base_position"].ne(0),
        _numeric(frame, "round_turn_cost_dollars").fillna(0.0),
        0.0,
    )
    frame["base_net_dollars"] = frame["base_gross_dollars"] - frame["base_cost_dollars"]
    frame["inverted_position"] = -frame["position"]
    frame["inverted_gross_dollars"] = (
        frame["inverted_position"]
        * _numeric(frame, "price_move").fillna(0.0)
        * _numeric(frame, "point_value").fillna(0.0)
    )
    frame["inverted_net_dollars"] = np.where(
        frame["position"].ne(0),
        frame["inverted_gross_dollars"] - _numeric(frame, "round_turn_cost_dollars").fillna(0.0),
        0.0,
    )
    return frame


def _alignment_metrics(frame: pd.DataFrame, scope: str, keys: Mapping[str, Any]) -> dict[str, Any]:
    traded = frame[frame["position"].ne(0)]
    base_signals = frame[frame["base_position"].ne(0)]
    blocked_base = frame[frame["base_position"].ne(0) & frame["position"].eq(0)]
    gross = _sum_float(frame["gross_dollars"])
    cost = _sum_float(frame["cost_dollars"])
    net = _sum_float(frame["net_dollars"])
    base_gross = _sum_float(frame["base_gross_dollars"])
    base_net = _sum_float(frame["base_net_dollars"])
    return {
        "scope": scope,
        **dict(keys),
        "row_count": int(len(frame)),
        "base_signal_count": int(frame["base_position"].ne(0).sum()),
        "trade_count": int(frame["position"].ne(0).sum()),
        "blocked_base_signal_count": int(len(blocked_base)),
        "long_count": int(frame["position"].eq(1).sum()),
        "short_count": int(frame["position"].eq(-1).sum()),
        "gross_return_dollars": gross,
        "cost_dollars": cost,
        "net_return_dollars": net,
        "base_signal_gross_dollars": base_gross,
        "base_signal_net_dollars": base_net,
        "gate_net_delta_vs_base_signal": net - base_net,
        "trade_rate_of_base_signals": (
            float(frame["position"].ne(0).sum() / frame["base_position"].ne(0).sum())
            if frame["base_position"].ne(0).any()
            else None
        ),
        "traded_target_direction_accuracy": _direction_accuracy(
            traded["position"], traded["target_direction"]
        ),
        "traded_argmax_direction_accuracy": _argmax_direction_accuracy(traded),
        "all_row_argmax_direction_accuracy": _argmax_direction_accuracy(frame),
        "traded_realized_direction_accuracy": _direction_accuracy(
            traded["position"], traded["realized_direction"]
        ),
        "traded_target_inverted_rate_nonzero": _inverted_rate(
            traded["position"], traded["target_direction"]
        ),
        "traded_realized_inverted_rate_nonzero": _inverted_rate(
            traded["position"], traded["realized_direction"]
        ),
        "base_target_direction_accuracy": _direction_accuracy(
            base_signals["base_position"], base_signals["target_direction"]
        ),
        "blocked_base_target_direction_accuracy": _direction_accuracy(
            blocked_base["base_position"], blocked_base["target_direction"]
        ),
        "avg_abs_direction_margin_traded": _mean_or_none(traded["abs_direction_margin"]),
        "avg_p_long_traded": _mean_or_none(traded["p_long"]),
        "avg_p_short_traded": _mean_or_none(traded["p_short"]),
        "avg_p_fade_success_traded": _mean_or_none(traded["p_fade_success"]),
        "avg_p_trend_danger_traded": _mean_or_none(traded["p_trend_danger"]),
        "position_base_signal_mismatch_count": int((~frame["position_matches_base_signal"]).sum()),
    }


def _group_alignment(
    frame: pd.DataFrame,
    *,
    scope: str,
    group_cols: list[str],
    sort_by: str = "net_return_dollars",
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        records.append(_alignment_metrics(group, scope, dict(zip(group_cols, keys))))
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(sort_by, ascending=True, na_position="last").reset_index(drop=True)


def _signal_bucket_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    group_specs = [
        ("signal_margin_bucket", ["signal_margin_bucket", "side"]),
        ("p_long_bucket", ["p_long_bucket", "side"]),
        ("p_short_bucket", ["p_short_bucket", "side"]),
        ("p_fade_bucket", ["p_fade_bucket", "side"]),
        ("p_trend_bucket", ["p_trend_bucket", "side"]),
    ]
    records: list[dict[str, Any]] = []
    for scope, cols in group_specs:
        for keys, group in frame.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            records.append(_alignment_metrics(group, scope, dict(zip(cols, keys))))
    if not records:
        return _empty_unavailable("no signal bucket records")
    return pd.DataFrame(records).sort_values("net_return_dollars").reset_index(drop=True)


def _gate_effect(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["gate_category", "base_side"], dropna=False):
        gate_category, base_side = keys
        records.append(
            _alignment_metrics(
                group,
                "gate_category_base_side",
                {"gate_category": gate_category, "base_side": base_side},
            )
        )
    if not records:
        return _empty_unavailable("no gate effect records")
    return pd.DataFrame(records).sort_values("gate_net_delta_vs_base_signal").reset_index(drop=True)


def _inversion_check(frame: pd.DataFrame) -> pd.DataFrame:
    traded = frame[frame["position"].ne(0)].copy()
    if traded.empty:
        return _empty_unavailable("no traded rows")
    actual_net = _sum_float(traded["net_dollars"])
    inverted_net = _sum_float(traded["inverted_net_dollars"])
    actual_gross = _sum_float(traded["gross_dollars"])
    inverted_gross = _sum_float(traded["inverted_gross_dollars"])
    return pd.DataFrame(
        [
            {
                "scope": "traded_rows",
                "trade_count": int(len(traded)),
                "actual_gross_dollars": actual_gross,
                "actual_net_dollars": actual_net,
                "inverted_gross_dollars": inverted_gross,
                "inverted_net_dollars": inverted_net,
                "inverted_net_delta": inverted_net - actual_net,
                "actual_target_direction_accuracy": _direction_accuracy(
                    traded["position"], traded["target_direction"]
                ),
                "inverted_target_direction_accuracy": _direction_accuracy(
                    -traded["position"], traded["target_direction"]
                ),
                "actual_realized_direction_accuracy": _direction_accuracy(
                    traded["position"], traded["realized_direction"]
                ),
                "inverted_realized_direction_accuracy": _direction_accuracy(
                    -traded["position"], traded["realized_direction"]
                ),
            }
        ]
    )


def _decision(overall: Mapping[str, Any], gate_effect: pd.DataFrame, inversion: pd.DataFrame) -> str:
    if int(overall.get("position_base_signal_mismatch_count") or 0) > 0:
        return "policy_logic_position_mismatch"
    traded_acc = overall.get("traded_target_direction_accuracy")
    base_acc = overall.get("base_target_direction_accuracy")
    blocked_acc = overall.get("blocked_base_target_direction_accuracy")
    argmax_acc = overall.get("all_row_argmax_direction_accuracy")
    inversion_delta = None
    inverted_acc = None
    if "unavailable_reason" not in inversion.columns and not inversion.empty:
        inversion_delta = inversion.iloc[0].get("inverted_net_delta")
        inverted_acc = inversion.iloc[0].get("inverted_target_direction_accuracy")
    if traded_acc is not None and float(traded_acc) < 0.50:
        if argmax_acc is not None and float(argmax_acc) >= 0.50:
            return "direction_edge_calibration_issue_not_policy_logic_bug"
        if base_acc is not None and float(base_acc) < 0.50:
            return "valid_but_weak_direction_signal"
    if (
        traded_acc is not None
        and base_acc is not None
        and float(traded_acc) < 0.50
        and float(base_acc) >= 0.50
    ):
        if blocked_acc is not None and float(blocked_acc) > float(traded_acc):
            return "policy_gates_select_bad_direction_subset"
        return "policy_selected_subset_direction_alignment_bad"
    if (
        inversion_delta is not None
        and float(inversion_delta) > 0.0
        and inverted_acc is not None
        and traded_acc is not None
        and float(inverted_acc) > float(traded_acc)
    ):
        return "traded_subset_direction_inversion"
    if not gate_effect.empty and "gate_net_delta_vs_base_signal" in gate_effect.columns:
        if float(gate_effect["gate_net_delta_vs_base_signal"].sum()) < 0.0:
            return "gates_reduce_base_signal_pnl"
    return "valid_but_weak_signal_or_cost_drag"


def _top_findings(
    *,
    overall: Mapping[str, Any],
    market_side: pd.DataFrame,
    gate_effect: pd.DataFrame,
    inversion: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    findings.append(
        "All-row argmax direction accuracy is "
        f"{float(overall.get('all_row_argmax_direction_accuracy') or 0.0):.4f}; "
        "traded target-direction accuracy is "
        f"{float(overall.get('traded_target_direction_accuracy') or 0.0):.4f}; "
        "base-signal accuracy is "
        f"{float(overall.get('base_target_direction_accuracy') or 0.0):.4f}."
    )
    findings.append(
        "Blocked base-signal target accuracy is "
        f"{float(overall.get('blocked_base_target_direction_accuracy') or 0.0):.4f}."
    )
    if "unavailable_reason" not in inversion.columns and not inversion.empty:
        row = inversion.iloc[0]
        findings.append(
            "Inverting traded directions would change net by "
            f"{float(row['inverted_net_delta']):.2f}."
        )
    if not market_side.empty and "unavailable_reason" not in market_side.columns:
        row = market_side.sort_values("net_return_dollars").iloc[0]
        findings.append(
            f"Worst market/side is {row.get('market')} {row.get('side')} "
            f"at {float(row.get('net_return_dollars')):.2f}."
        )
    if "unavailable_reason" not in gate_effect.columns and not gate_effect.empty:
        row = gate_effect.sort_values("gate_net_delta_vs_base_signal").iloc[0]
        findings.append(
            f"Worst gate bucket is {row.get('gate_category')} {row.get('base_side')} "
            f"with gate delta {float(row.get('gate_net_delta_vs_base_signal')):.2f}."
        )
    return findings[:5]


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    findings = "\n".join(f"- {item}" for item in summary["top_findings"])
    files = "\n".join(
        f"- `{value}`" for key, value in sorted(summary["outputs"].items()) if key != "readme"
    )
    text = f"""# Phase 8 Policy Signal Alignment

Run: `{summary['run']}`

This diagnostic recomputes Phase 8 policy rows from saved predictions and
audits direction alignment, gate effects, and traded-vs-blocked signal quality.
It does not alter features, predictions, model training, WFA splits, or policy
behavior.

## Top Findings

{findings}

## Files

{files}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_policy_signal_alignment(
    *,
    predictions_path: Path,
    costs_config: Path,
    output_root: Path,
    run: str,
    policy: PolicyConfig,
) -> dict[str, Any]:
    if not predictions_path.exists():
        raise SystemExit(f"prediction parquet missing: {_relative_path(predictions_path)}")
    predictions = pd.read_parquet(predictions_path)
    policy_frame, failures, warnings = build_policy_frame(predictions, costs_config, policy)
    if failures:
        raise SystemExit("; ".join(failures))
    frame = _attach_alignment_fields(policy_frame)
    traded = frame[frame["position"].ne(0)].copy()

    overall = _alignment_metrics(frame, "overall", {})
    overall_frame = pd.DataFrame([overall])
    market_side = _group_alignment(traded, scope="market_side", group_cols=["market", "side"])
    market_fold = _group_alignment(traded, scope="market_fold", group_cols=["market", "fold_id"])
    hour_side = _group_alignment(traded, scope="hour_side", group_cols=["hour_utc", "side"])
    signal_buckets = _signal_bucket_alignment(traded)
    gate_effect = _gate_effect(frame)
    inversion = _inversion_check(frame)
    decision = _decision(overall, gate_effect, inversion)

    output_root.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_root, run)
    _write_csv(paths["overall"], overall_frame)
    _write_csv(paths["market_side"], market_side)
    _write_csv(paths["market_fold"], market_fold)
    _write_csv(paths["hour_side"], hour_side)
    _write_csv(paths["signal_buckets"], signal_buckets)
    _write_csv(paths["gate_effect"], gate_effect)
    _write_csv(paths["inversion_check"], inversion)
    outputs = {key: _relative_path(path) for key, path in paths.items()}
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "prediction_path": _relative_path(predictions_path),
        "prediction_count": int(len(predictions)),
        "policy_row_count": int(len(frame)),
        "trade_count": int(frame["position"].ne(0).sum()),
        "decision": decision,
        "overall": overall,
        "worst_market_side_rows": market_side.head(10).to_dict(orient="records"),
        "worst_market_fold_rows": market_fold.head(10).to_dict(orient="records"),
        "worst_signal_bucket_rows": signal_buckets.head(10).to_dict(orient="records"),
        "gate_effect_rows": gate_effect.to_dict(orient="records"),
        "inversion_check": inversion.to_dict(orient="records"),
        "top_findings": _top_findings(
            overall=overall,
            market_side=market_side,
            gate_effect=gate_effect,
            inversion=inversion,
        ),
        "warnings": warnings,
        "outputs": outputs,
    }
    _write_json(paths["summary"], summary)
    _write_readme(paths["readme"], summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS.as_posix())
    parser.add_argument("--costs-config", default=DEFAULT_COSTS_CONFIG.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--long-short-margin", type=float, default=0.05)
    parser.add_argument("--min-fade-success", type=float, default=0.50)
    parser.add_argument("--max-trend-danger", type=float, default=0.50)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = build_policy_signal_alignment(
        predictions_path=Path(args.predictions),
        costs_config=Path(args.costs_config),
        output_root=Path(args.output_root),
        run=args.run,
        policy=PolicyConfig(
            long_short_margin=args.long_short_margin,
            min_fade_success=args.min_fade_success,
            max_trend_danger=args.max_trend_danger,
        ),
    )
    overall = summary["overall"]
    print(
        "PASS policy signal alignment: "
        f"rows={summary['policy_row_count']} "
        f"trades={summary['trade_count']} "
        f"traded_accuracy={overall['traded_target_direction_accuracy']} "
        f"decision={summary['decision']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
