from __future__ import annotations

import argparse
import importlib.util
import importlib
import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.analytics.aggregate import compute_backtest_metrics, write_metrics_report
from pipeline.analytics.failure_attribution import write_failure_attribution_report
from pipeline.audit.execution_trace import build_execution_trace, validate_execution_trace, write_execution_trace_outputs
from pipeline.audit.leakage import run_leakage_audit
from pipeline.audit.run_manifest import write_run_manifest
from pipeline.common.config import RootConfig, config as flat_config, load_config
from pipeline.common.io_safe import atomic_write_json, write_csv_rows
from pipeline.common.cache import build_cache_metadata, cache_is_fresh, write_cache_metadata
from pipeline.execution.cost_model import attach_execution_cost_model
from pipeline.gates.acceptance import run_acceptance_gate
from pipeline.gates.deployment import run_deployment_readiness
from pipeline.stress.stress_tests import run_stress_tests
from pipeline.data_gate.checkpoint import validate_checkpoint_stage
from pipeline.orchestration.stage_plan import normalize_start_stage


EXCLUDE_COLS = {
    "ts_event", "date", "session", "session_id", "session_date", "symbol", "market",
    "session_timezone", "session_calendar_accuracy", "rtype", "publisher_id", "instrument_id",
    "open", "high", "low", "close", "volume",
    "prediction_time", "earliest_execution_time", "execution_time", "non_model_metadata_columns",
    "pnl", "gross_pnl", "net_pnl", "fees", "slippage",
    "position_before", "position_after", "position_delta", "raw_signal", "prediction_prob",
}

MINIMAL_MODELING_WARNING = "minimal_compatible modeling validates pipeline wiring only; it is not strategy evidence"


def _load_cfg() -> RootConfig:
    cfg = load_config(os.environ.get("CONFIG_ENV") or os.environ.get("QUANT_ENV"))
    if cfg is None:
        # load_config is idempotent and may return None after first call in tests.
        cfg = RootConfig()
    override = os.environ.get("QUANT_MODELING_MODE")
    if override:
        if override not in {"minimal_compatible", "full_research"}:
            raise ValueError(f"unsupported QUANT_MODELING_MODE={override!r}")
        cfg.pipeline.modeling_mode = override
        flat_config.MODELING_MODE = override
    return cfg


def _read_data(pattern: str, start: str | None = None, end: str | None = None) -> pl.DataFrame:
    paths = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
    if not paths:
        raise FileNotFoundError(f"no data files matched: {pattern}")
    frames = [pl.read_parquet(p) for p in paths]
    df = pl.concat(frames, how="diagonal") if len(frames) > 1 else frames[0]
    if "ts_event" in df.columns:
        if start:
            df = df.filter(pl.col("ts_event") >= pl.lit(_parse_like_ts(start, df["ts_event"].dtype)).cast(df["ts_event"].dtype))
        if end:
            df = df.filter(pl.col("ts_event") < pl.lit(_parse_like_ts(end, df["ts_event"].dtype)).cast(df["ts_event"].dtype))
    return df.sort("ts_event") if "ts_event" in df.columns else df


def _parse_like_ts(value: str, dtype: pl.DataType) -> Any:
    if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return int(float(value))
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if getattr(dtype, "time_zone", None):
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        return dt.replace(tzinfo=None)
    except Exception:
        return value


def _symbol_from_path(path: str) -> str:
    p = Path(path)
    return p.parent.name if p.parent.name else p.stem.split("_")[0]


def _ensure_target(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    if target_col in df.columns:
        return df
    price = "open" if "open" in df.columns else "close"
    if price not in df.columns:
        raise ValueError(f"cannot derive {target_col}: missing open/close")
    return df.with_columns(((pl.col(price).shift(-16) / pl.col(price).shift(-1)).log()).alias(target_col))


def _enforce_safe_label_end(df: pl.DataFrame, end: str | None, cfg: RootConfig) -> pl.DataFrame:
    if not end or "ts_event" not in df.columns:
        return df
    dtype = df["ts_event"].dtype
    horizon_min = int(cfg.target.target_15m_horizon) * 5
    if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        cutoff = int(float(end)) - horizon_min
        return df.filter(pl.col("ts_event") < cutoff)
    try:
        from datetime import timedelta

        cutoff = _parse_like_ts(end, dtype) - timedelta(minutes=horizon_min)
        return df.filter(pl.col("ts_event") < pl.lit(cutoff).cast(dtype))
    except Exception:
        return df.head(max(df.height - horizon_min, 0))


def _add_basic_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    if "close" in df.columns:
        ret = pl.col("close").pct_change()
        exprs.append(pl.when(ret.is_finite()).then(ret).otherwise(0).fill_null(0).alias("ret_1"))
    if "volume" in df.columns:
        vol = pl.col("volume").cast(pl.Float64).pct_change()
        exprs.append(pl.when(vol.is_finite()).then(vol).otherwise(0).fill_null(0).alias("volume_chg"))
    return df.with_columns(exprs) if exprs else df


def _feature_cols(df: pl.DataFrame, target_col: str, cfg: RootConfig) -> list[str]:
    forbidden = list(cfg.leakage_audit.forbidden_feature_prefixes) + list(cfg.leakage_audit.forbidden_model_metadata_prefixes)
    cols = []
    for c, dtype in zip(df.columns, df.dtypes):
        if c == target_col or c in EXCLUDE_COLS:
            continue
        if any(c.startswith(p) for p in forbidden):
            continue
        if dtype.is_numeric():
            cols.append(c)
    return cols


def _forbidden_input_columns(df: pl.DataFrame, target_col: str, cfg: RootConfig) -> list[str]:
    bad = []
    for c in df.columns:
        if c == target_col:
            continue
        if c.startswith("future_") or c.startswith("label_") or c.startswith("target_"):
            bad.append(c)
    return bad


def _window_name(args: argparse.Namespace) -> str:
    raw = "_".join(str(x or "none") for x in [getattr(args, "start", None), getattr(args, "end", None)])
    return "".join(ch if ch.isalnum() else "-" for ch in raw)[:80]


def cmd_discover(args: argparse.Namespace) -> None:
    cfg = _load_cfg()
    df = _add_basic_features(_read_data(args.data, args.start, args.end))
    target_col = getattr(flat_config, "DISCOVERY_TARGET", cfg.walkforward.discovery_target)
    df = _ensure_target(df, target_col)
    df = _enforce_safe_label_end(df, args.end, cfg)
    features = _feature_cols(df, target_col, cfg)
    payload = {
        "status": "PASS",
        "target_col": target_col,
        "feature_cols": features,
        "selected_features": features,
        "data": args.data,
        "rows": df.height,
        "start": args.start,
        "end": args.end,
    }
    atomic_write_json(args.out, payload)
    print(f"[CLI] discovery manifest: {args.out} features={len(features)} rows={df.height}")


def _load_manifest_features(path: str | None) -> list[str]:
    raw = _load_manifest_payload(path)
    return list(raw.get("selected_features") or raw.get("feature_cols") or [])


def _load_manifest_payload(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_minimal_compatible_modeling(df: pl.DataFrame, features: list[str], target_col: str, cfg: RootConfig, symbol: str | None = None) -> pl.DataFrame:
    if not features:
        raise ValueError("no model features available after safe feature selection")
    score_expr = sum([pl.col(c).fill_null(0).cast(pl.Float64) for c in features]) / float(len(features))
    df = df.with_columns(score_expr.tanh().alias("_score"))
    df = df.with_columns(
        pl.col("_score").alias("prediction"),
        (0.5 + 0.25 * pl.col("_score")).clip(0.0, 1.0).alias("prediction_prob"),
        pl.when(pl.col("_score") > float(cfg.execution.prediction_entry_threshold)).then(1)
        .when(pl.col("_score") < -float(cfg.execution.prediction_entry_threshold)).then(-1)
        .otherwise(0).alias("raw_signal"),
        pl.lit(float(cfg.execution.prediction_entry_threshold)).alias("signal_entry_threshold"),
    )
    if "ts_event" in df.columns:
        if df["ts_event"].dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            exec_time = pl.col("ts_event") + int(cfg.execution.entry_lag_bars)
        else:
            exec_time = pl.col("ts_event") + pl.duration(minutes=int(cfg.execution.entry_lag_bars))
        df = df.with_columns(pl.col("ts_event").alias("prediction_time"), exec_time.alias("execution_time"))
    df = attach_execution_cost_model(df, target_col=target_col, config=cfg, symbol=symbol, feature_set_id="minimal_compatible")
    return df.drop_nulls([target_col, "prediction_prob", "pnl"])


def _run_full_research_modeling(df: pl.DataFrame, features: list[str], target_col: str, cfg: RootConfig, context: dict[str, Any]) -> pl.DataFrame:
    required = [
        ("pipeline.features.engine", "load_or_build_feature_target_matrix"),
        ("pipeline.features.discovery", "select_features_train_only"),
        ("pipeline.features.preprocessing", "fit_apply_train_scaler"),
        ("pipeline.walkforward.walkforward", "apply_walkforward_contract"),
        ("pipeline.modeling.full_research", "run_full_research_modeling"),
    ]
    for mod_name, attr in required:
        if importlib.util.find_spec(mod_name) is None:
            raise RuntimeError(f"FULL_RESEARCH MODELING FAIL: missing {mod_name}.{attr}")
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, attr):
            raise RuntimeError(f"FULL_RESEARCH MODELING FAIL: missing {mod_name}.{attr}")
    from pipeline.modeling.full_research import run_full_research_modeling

    result, artifacts = run_full_research_modeling(
        df,
        features,
        target_col,
        context.get("train_start"),
        context.get("train_end"),
        context.get("test_start"),
        context.get("test_end"),
        context,
    )
    context.setdefault("modeling_artifacts", {}).update(artifacts)
    return result


def run_modeling_pipeline(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_start: str | None,
    train_end: str | None,
    test_start: str | None,
    test_end: str | None,
    context: dict[str, Any],
) -> pl.DataFrame:
    cfg: RootConfig = context["config"]
    context.setdefault("train_start", train_start)
    context.setdefault("train_end", train_end)
    context.setdefault("test_start", test_start)
    context.setdefault("test_end", test_end)
    mode = getattr(cfg.pipeline, "modeling_mode", "minimal_compatible")
    if mode == "minimal_compatible":
        return _run_minimal_compatible_modeling(df, feature_cols, target_col, cfg, context.get("symbol"))
    if mode == "full_research":
        return _run_full_research_modeling(df, feature_cols, target_col, cfg, context)
    raise ValueError(f"unsupported pipeline.modeling_mode={mode!r}; expected minimal_compatible or full_research")


def _write_oos_predictions(
    result: pl.DataFrame,
    out_path: Path,
    *,
    target_col: str,
    symbol: str,
    split_id: str,
    train_start: str | None,
    train_end: str | None,
    test_start: str | None,
    test_end: str | None,
    modeling_mode: str,
) -> None:
    cols = []
    for name in ["ts_event", "prediction_time", "execution_time", "prediction", "prediction_prob"]:
        if name in result.columns:
            cols.append(pl.col(name))
    cols.extend(
        [
            pl.lit(target_col).alias("target_col"),
            pl.col(target_col).alias("target_value") if target_col in result.columns else pl.lit(None).alias("target_value"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(split_id).alias("split_id"),
            pl.lit(train_start).alias("train_start"),
            pl.lit(train_end).alias("train_end"),
            pl.lit(test_start).alias("test_start"),
            pl.lit(test_end).alias("test_end"),
            pl.lit(modeling_mode).alias("modeling_mode"),
            pl.col("feature_set_id") if "feature_set_id" in result.columns else pl.lit("").alias("feature_set_id"),
        ]
    )
    result.select(cols).write_parquet(out_path)


def _json_status(path: Path) -> str:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status", "UNKNOWN")
    except Exception:
        return "MISSING"


def cmd_run(args: argparse.Namespace, hmm: bool = False) -> None:
    cfg = _load_cfg()
    command = "run-hmm" if hmm else "run"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from_stage = normalize_start_stage(getattr(args, "from_stage", None) or getattr(cfg.pipeline, "start_stage", "raw"))
    data_root = getattr(args, "data_root", None)
    if from_stage != "raw":
        if not data_root:
            data_root = getattr(cfg.pipeline, "checkpoint_root", None) or getattr(cfg.data, "root", None)
        gate = validate_checkpoint_stage(from_stage, data_root, cfg, cfg.symbols, cfg.start_year, cfg.end_year)
        if gate["status"] != "PASS":
            raise SystemExit(f"CHECKPOINT GATE FAIL: stage={from_stage} root={data_root} reason={'; '.join(gate['failures'][:3])}")
        if not getattr(args, "data", None):
            files = sorted(Path(data_root).glob("*/*.parquet"))
            if not files:
                raise SystemExit(f"CHECKPOINT GATE FAIL: stage={from_stage} root={data_root} reason=no parquet files")
            args.data = str(files[0])
        if not getattr(args, "manifest", None):
            args.manifest = str(out_dir / "checkpoint_discovery_manifest.json")
            cmd_discover(argparse.Namespace(data=args.data, out=args.manifest, start=getattr(args, "train_start", None), end=getattr(args, "train_end", None)))
    symbol = _symbol_from_path(args.data)
    window = _window_name(args)
    profile = getattr(flat_config, "ACTIVE_PROFILE", cfg.__class__.__name__)
    modeling_mode = getattr(cfg.pipeline, "modeling_mode", "minimal_compatible")
    if modeling_mode == "minimal_compatible":
        print(f"[CLI] WARNING modeling_mode=minimal_compatible: {MINIMAL_MODELING_WARNING}")
    df = _add_basic_features(_read_data(args.data, args.start, args.end))
    target_col = getattr(flat_config, "WALKFORWARD_TARGET", cfg.walkforward.walkforward_target)
    df = _ensure_target(df, target_col)
    df = _enforce_safe_label_end(df, args.end, cfg)
    bad = _forbidden_input_columns(df, target_col, cfg)
    if bad:
        raise SystemExit(f"LEAKAGE FAIL: forbidden input columns present before modeling: {bad}")
    manifest_payload = _load_manifest_payload(args.manifest)
    manifest_features = list(manifest_payload.get("selected_features") or manifest_payload.get("feature_cols") or [])
    safe_features = _feature_cols(df, target_col, cfg)
    features = [c for c in manifest_features if c in safe_features] or safe_features
    leakage_path = Path(cfg.leakage_audit.report_dir) / f"{profile}_{symbol}_{command}_{window}.json"
    leakage = run_leakage_audit(df, features, target_col, context={"out": str(leakage_path), "symbol": symbol, "command": command})
    if cfg.leakage_audit.fail_on_error and leakage["status"] == "FAIL":
        raise SystemExit(f"LEAKAGE FAIL: {leakage_path}")
    split_id = Path(out_dir).name.split("_")[-1] if "split" in Path(out_dir).name else "1"
    modeling_context = {
        "config": cfg,
        "symbol": symbol,
        "command": command,
        "run_id": Path(out_dir).name,
        "split_id": split_id,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_start": args.start,
        "test_end": args.end,
    }
    modeling_df = df
    if modeling_mode == "full_research":
        train_data = manifest_payload.get("data")
        if not train_data:
            raise SystemExit("FULL_RESEARCH MODELING FAIL: missing train data path in discovery manifest")
        train_df = _add_basic_features(_read_data(train_data, args.train_start, args.train_end))
        modeling_df = pl.concat([train_df, df], how="diagonal_relaxed")
    result_path = out_dir / ("backtest_results_hmm.parquet" if hmm else "backtest_results.parquet")
    result_meta = build_cache_metadata(
        result_path,
        source_stage=from_stage if from_stage != "raw" else "input_data",
        output_stage="backtest_results",
        source_paths=[args.data, args.manifest],
        config=cfg,
        config_sections=["target", "features", "execution", "walkforward", "pipeline"],
        code_paths=[__file__, "pipeline/modeling/full_research.py"],
        symbol=symbol,
        split_id=split_id,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.start,
        test_end=args.end,
    )
    fresh_result, _ = cache_is_fresh(result_path, result_meta, cfg)
    if fresh_result:
        result = pl.read_parquet(result_path)
    else:
        result = run_modeling_pipeline(
            modeling_df,
            features,
            target_col,
            args.train_start,
            args.train_end,
            args.start,
            args.end,
            modeling_context,
        )
        result.write_parquet(result_path)
        write_cache_metadata(result_path, result_meta)
    _write_oos_predictions(
        result,
        out_dir / "oos_predictions.parquet",
        target_col=target_col,
        symbol=symbol,
        split_id=split_id,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.start,
        test_end=args.end,
        modeling_mode=modeling_mode,
    )
    for pth, out_stage in [(out_dir / "oos_predictions.parquet", "oos_predictions")]:
        write_cache_metadata(pth, build_cache_metadata(
            pth, source_stage="backtest_results", output_stage=out_stage,
            source_paths=[result_path], config=cfg,
            config_sections=["target", "execution", "pipeline"],
            code_paths=[__file__], symbol=symbol, split_id=split_id,
            train_start=args.train_start, train_end=args.train_end, test_start=args.start, test_end=args.end,
        ))
    stage_oos_name = "final_oos_predictions.parquet" if modeling_mode == "full_research" else "baseline_oos_predictions.parquet"
    _write_oos_predictions(
        result,
        out_dir / stage_oos_name,
        target_col=target_col,
        symbol=symbol,
        split_id=split_id,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.start,
        test_end=args.end,
        modeling_mode=modeling_mode,
    )
    write_cache_metadata(out_dir / stage_oos_name, build_cache_metadata(
        out_dir / stage_oos_name, source_stage="backtest_results", output_stage=stage_oos_name.replace(".parquet", ""),
        source_paths=[result_path], config=cfg, config_sections=["target", "execution", "pipeline"],
        code_paths=[__file__], symbol=symbol, split_id=split_id,
        train_start=args.train_start, train_end=args.train_end, test_start=args.start, test_end=args.end,
    ))
    metrics = compute_backtest_metrics(result)
    metrics["modeling_mode"] = modeling_mode
    if modeling_mode == "minimal_compatible":
        metrics["warnings"] = [MINIMAL_MODELING_WARNING]
    metrics_path = Path("reports/metrics") / f"{profile}_{symbol}_{command}_{window}_metrics_report.json"
    metrics_meta = build_cache_metadata(
        metrics_path, source_stage="backtest_results", output_stage="metrics_report",
        source_paths=[result_path], config=cfg,
        config_sections=["execution", "walkforward", "pipeline"],
        code_paths=[__file__, "pipeline/analytics/aggregate.py"], symbol=symbol, split_id=split_id,
    )
    fresh_metrics, _ = cache_is_fresh(metrics_path, metrics_meta, cfg)
    if fresh_metrics:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        atomic_write_json(metrics_path, metrics)
        write_csv_rows(metrics_path.with_suffix(".csv"), [metrics])
        write_cache_metadata(metrics_path, metrics_meta)
    diag_prefix = Path("reports/diagnostics") / f"{profile}_{symbol}_{command}_{window}_failure_attribution"
    diag_report = write_failure_attribution_report(
        result,
        diag_prefix,
        symbol=symbol,
        split_id=split_id,
        modeling_mode=modeling_mode,
    )
    trace = build_execution_trace(result, max_rows=cfg.execution.execution_trace_rows)
    exec_report = validate_execution_trace(trace, cfg)
    write_execution_trace_outputs(trace, exec_report, out_dir)
    if cfg.acceptance_gate.fail_on_execution_trace_error and exec_report["status"] == "FAIL":
        raise SystemExit(f"EXECUTION TRACE FAIL: {out_dir / 'execution_trace_report.json'}")
    stress = None
    stress_path = None
    if cfg.stress_tests.enabled:
        stress_prefix = Path(cfg.stress_tests.report_dir) / f"{profile}_{symbol}_{command}_{window}_stress_report"
        stress_path = stress_prefix.with_suffix(".json")
        stress_meta = build_cache_metadata(
            stress_path, source_stage="backtest_results", output_stage="stress_report",
            source_paths=[result_path], config=cfg,
            config_sections=["execution", "stress_tests", "pipeline"],
            code_paths=["pipeline/stress/stress_tests.py"], symbol=symbol, split_id=split_id,
        )
        fresh_stress, _ = cache_is_fresh(stress_path, stress_meta, cfg)
        if fresh_stress:
            stress = json.loads(stress_path.read_text(encoding="utf-8"))
        else:
            stress = run_stress_tests(result, cfg, stress_prefix)
            write_cache_metadata(stress_path, stress_meta)
    acceptance_path = Path(cfg.acceptance_gate.report_dir) / f"{profile}_{symbol}_{command}_{window}_acceptance_gate.json"
    acceptance_meta = build_cache_metadata(
        acceptance_path, source_stage="metrics_report", output_stage="acceptance_report",
        source_paths=[metrics_path, stress_path or ""], config=cfg,
        config_sections=["acceptance_gate", "pipeline"],
        code_paths=["pipeline/gates/acceptance.py"], symbol=symbol, split_id=split_id,
    )
    fresh_acceptance, _ = cache_is_fresh(acceptance_path, acceptance_meta, cfg)
    if fresh_acceptance:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    else:
        acceptance = run_acceptance_gate(metrics, stress, leakage, exec_report, context={"config": cfg, "out": str(acceptance_path), "symbol": symbol, "command": command, "modeling_mode": modeling_mode})
        write_cache_metadata(acceptance_path, acceptance_meta)
    if acceptance["status"] == "REJECT":
        failed = [g["name"] for g in acceptance.get("gates", []) if g.get("status") == "FAIL"]
        print(f"[CLI] ACCEPTANCE REJECT: {acceptance_path} failed_gates={','.join(failed)}")
        if os.environ.get("QUANT_ACCEPTANCE_GATE_REQUIRED") == "1" or cfg.acceptance_gate.required:
            raise SystemExit(f"ACCEPTANCE GATE REJECT: {acceptance_path} failed_gates={','.join(failed)}")
    modeling_artifacts = modeling_context.get("modeling_artifacts", {})
    print(f"[CLI] Running {'HMM-aware ' if hmm else ''}walkforward")
    print(f"[CLI] {'HMM ' if hmm else ''}walkforward result: {result.height:,} rows")
    print(f"[CLI] wrote {result_path}")
    write_run_manifest(
        run_id=Path(out_dir).name,
        config=cfg,
        files=[Path(args.data)],
        audit_paths={
            "leakage": str(leakage_path),
            "execution_trace": str(out_dir / "execution_trace_report.json"),
            "metrics": str(metrics_path),
            "failure_attribution": str(diag_prefix.with_suffix(".json")),
            "stress": str(stress_path) if stress_path else "",
            "acceptance": str(acceptance_path),
            "output": str(result_path),
            "oos_predictions": str(out_dir / "oos_predictions.parquet"),
            "stage_oos_predictions": str(out_dir / stage_oos_name),
            "selector": str(modeling_artifacts.get("selector_path", "")),
            "scaler": str(modeling_artifacts.get("scaler_path", "")),
            "cli_command": command,
        },
        splits=[
            {
                "symbol": symbol,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "test_start": args.start,
                "test_end": args.end,
                "backtest_results": str(result_path),
                "oos_predictions": str(out_dir / "oos_predictions.parquet"),
                "stage_oos_predictions": str(out_dir / stage_oos_name),
                "baseline_oos_predictions": str(out_dir / "baseline_oos_predictions.parquet") if modeling_mode != "full_research" else "",
                "final_oos_predictions": str(out_dir / "final_oos_predictions.parquet") if modeling_mode == "full_research" else "",
                "leakage_report": str(leakage_path),
                "leakage_status": leakage.get("status"),
                "execution_trace_report": str(out_dir / "execution_trace_report.json"),
                "execution_trace_status": exec_report.get("status"),
                "metrics_report": str(metrics_path),
                "failure_attribution_report": str(diag_prefix.with_suffix(".json")),
                "dominant_failure": diag_report.get("diagnostic", {}).get("dominant_failure"),
                "metrics_status": "PASS",
                "stress_report": str(stress_path) if stress_path else "",
                "stress_status": stress.get("status") if stress else "MISSING",
                "acceptance_report": str(acceptance_path),
                "acceptance_status": acceptance.get("status"),
                "selector_artifact": str(modeling_artifacts.get("selector_path", "")),
                "scaler_artifact": str(modeling_artifacts.get("scaler_path", "")),
            }
        ],
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    cfg = _load_cfg()
    root = Path(args.artifacts)
    paths = sorted(root.rglob("backtest_results*.parquet"))
    if not paths:
        raise SystemExit(f"AGGREGATE FAIL: no backtest_results parquet under {root}")
    df = pl.concat([pl.read_parquet(p) for p in paths], how="diagonal")
    metrics = write_metrics_report(df, getattr(flat_config, "ACTIVE_PROFILE", "profile"))
    run_deployment_readiness(cfg)
    print(json.dumps({"status": "PASS", "files": len(paths), "metrics": metrics}, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pipeline.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--data", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--start")
    d.add_argument("--end")
    r = sub.add_parser("run")
    r.add_argument("--data")
    r.add_argument("--manifest")
    r.add_argument("--out", required=True)
    r.add_argument("--from-stage")
    r.add_argument("--data-root")
    r.add_argument("--train-start")
    r.add_argument("--train-end")
    r.add_argument("--start")
    r.add_argument("--end")
    rh = sub.add_parser("run-hmm")
    rh.add_argument("--data")
    rh.add_argument("--manifest")
    rh.add_argument("--out", required=True)
    rh.add_argument("--from-stage")
    rh.add_argument("--data-root")
    rh.add_argument("--train-start")
    rh.add_argument("--train-end")
    rh.add_argument("--start")
    rh.add_argument("--end")
    a = sub.add_parser("aggregate")
    a.add_argument("--artifacts", default="output")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "discover":
        cmd_discover(args)
    elif args.cmd == "run":
        cmd_run(args, hmm=False)
    elif args.cmd == "run-hmm":
        cmd_run(args, hmm=True)
    elif args.cmd == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
