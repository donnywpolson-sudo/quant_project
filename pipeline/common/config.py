"""
config_manager.py — Single-source-of-truth configuration.

Pydantic-validated hierarchical config with backward-compatible
SimpleNamespace for all quant modules.

Architecture (profile-based, primary):
  - ``load_config()``: reads configs/alpha_tiered.yaml, deep-merges
    ``base`` + ``profiles[active_profile]``, validates with Pydantic,
    resolves ${ENV_VAR} placeholders, populates the SimpleNamespace,
    and returns the Pydantic RootConfig.

  - ``CONFIG_ENV`` env var overrides ``active_profile`` at runtime.

  - Falls back to old flat YAML tier loading (alpha_0.yaml + tier YAML)
    if alpha_tiered.yaml is absent or not profile-based.

Usage:
    # Structured (Pydantic) — run.py
    from pipeline.common.config import load_config, RootConfig
    cfg: RootConfig = load_config()
    print(cfg.discovery.bootstrap_folds)

    # Flat (SimpleNamespace) — all quant modules
    from pipeline.common.config import config
    print(config.BOOTSTRAP_FOLDS)
    print(config.ACTIVE_PROFILE)

    # Idempotent — safe to call multiple times
    from pipeline.common.config import load_config
    load_config()  # no-op after first call
"""

import copy
import logging
import os
import re
from datetime import time
from pathlib import Path
from types import SimpleNamespace

DEFAULT_MARKETS = [
    "6B", "6E", "6J",
    "CL", "ES", "GC", "HE", "HG", "LE", "NG", "NQ", "RTY", "SI", "SR3",
    "YM", "ZB", "ZC", "ZN", "ZS", "ZW",
]
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ============================================================================
# Thread-limiting environment variables — default to full multi-threading
# for data ingestion and feature engineering.  Call clamp_to_single_threaded()
# before model fitting (ExtraTrees, GaussianHMM, etc.) where deterministic
# reproducibility is required.  Executes at module import time so that
# anything importing config_manager gets the default (multi-threaded).
# ============================================================================
_THREAD_VARS = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
}


def _enable_multi_threading() -> None:
    """Enable full CPU utilisation by clearing thread-limiting env vars.
    Called at module import time so data ingestion and feature engineering
    benefit from multi-threading by default."""
    for var in _THREAD_VARS:
        os.environ.pop(var, None)
    logger.debug(
        "Multi-threading enabled (thread-limiting vars cleared): %s",
        list(_THREAD_VARS),
    )


def clamp_to_single_threaded() -> None:
    """Force numeric libraries to single-threaded mode for model fitting
    where deterministic reproducibility is required.

    Call this before fitting ExtraTreesRegressor, GaussianHMM, or any
    other estimator where thread-level nondeterminism matters."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    logger.debug(
        "Thread-limiting env vars clamped to single-threaded: %s",
        list(_THREAD_VARS),
    )


_enable_multi_threading()

# ============================================================================
# Module-level config namespace (SimpleNamespace with flat attributes)
# This is what ALL quant modules import and use.
# Populated by load_config().
# ============================================================================
config = SimpleNamespace()

_LOADED = False

# ============================================================================
# Pydantic Models — complete schema covering every parameter from the old
# config_loader.py defaults (418 lines) plus the alpha tier YAML files.
# ============================================================================


class SessionConfig(BaseModel):
    timezone: str = "America/New_York"
    session_start_local: str = "18:00"
    session_end_local: str = "16:00"
    session_break_start_local: str = "17:00"
    session_break_end_local: str = "18:00"


class FeaturesConfig(BaseModel):
    resample_frequencies: list[str] = Field(default_factory=lambda: ["1m"])
    drop_incomplete_rows: bool = True
    roll_windows: list[int] = Field(default_factory=lambda: [5, 10, 20, 50])
    roll_windows_1h: list[int] = Field(default_factory=lambda: [2, 4, 6, 12])
    roll_windows_daily: list[int] = Field(default_factory=lambda: [5, 10, 20])
    roll_window_min_rows: int = 20
    feature_transforms: list[str] = Field(
        default_factory=lambda: [
            "lags",
            "ratios",
            "z_scores",
            "pairwise_products_limited",
            "cross_timeframe_ratios",
        ]
    )
    max_pairwise_interactions: int = 500
    max_cross_timeframe_interactions: int = 200
    htf_trend_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    htf_volatility_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    htf_alignment_filter: bool = True
    htf_trend_threshold: float = 0.1
    vol_median_window: int = 20
    vol_smooth_window: int = 5
    regime_high_thresh: float = 0.6
    regime_low_thresh: float = 0.4
    regime_missing_default: float = 0.0


class TargetConfig(BaseModel):
    target_15m_horizon: int = 15
    target_scale_factor: float = 100.0


class DiscoveryConfig(BaseModel):
    discovery_window_days: int = 60
    bootstrap_folds: int = 30
    extra_trees_params: dict = Field(
        default_factory=lambda: {
            "random_state": 42,
            "n_jobs": 1,
            "n_estimators": 100,
            "max_depth": 8,
            "max_features": 0.3,
            "bootstrap": False,
        }
    )
    selection_freq_threshold: float = 0.75
    sign_consistency_threshold: float = 0.8
    cumulative_importance_threshold: float = 0.95
    min_selected_features: int = 10
    max_selected_features: int = 1000


class WalkforwardConfig(BaseModel):
    wf_train_days: int = 60
    wf_test_days: int = 1
    wf_step_days: int = 1
    ridge_params: dict = Field(
        default_factory=lambda: {
            "alpha": 1.0,
            "solver": "cholesky",
            "fit_intercept": True,
            "random_state": 42,
        }
    )
    model_type: str = "Ridge"
    probability_smoothing_alpha: float = 0.1
    corr_threshold: float = 0.95
    wf_parallel_folds: int = 1
    burn_in_bars: int = 500
    enable_meta_labeling: bool = False
    meta_threshold: float = 0.5
    mode: str = ""  # "" = inner bar-fold walkforward, "outer_split" = single train→test pass
    discovery_target: str = "target_15m_ret"
    walkforward_target: str = "target_15m_ret"
    embargo_bars: int = 0
    purge_target_overlap: bool = True


class ExecutionConfig(BaseModel):
    execute_at: str = "open[t+1]"
    entry_lag_bars: int = 1
    slippage_k: float = 0.001
    slippage_ticks: float = 1.0
    spread_ticks: float = 1.0
    vol_penalty: float = 0.005
    commission_per_trade: float = 2e-05
    tx_cost_per_roundturn: float = 0.00015
    commission_per_contract: float = 1.50
    exchange_fees_per_contract: float = 0.0
    participation_rate_limit: float = 0.05
    max_contracts: int = 1
    target_vol: float = 0.01
    max_leverage: float = 3.0
    max_pos_change_per_min: float = 0.1
    flat_before_close_minutes: int = 5
    latency_bars: int = 0
    reject_same_bar_fill: bool = True
    prediction_entry_threshold: float = 0.0
    min_position_hold_bars: int = 0
    write_execution_trace: bool = True
    execution_trace_rows: int = 200
    htf_trend_alignment: bool = True
    htf_vol_scaling: bool = True
    htf_vol_window: int = 10
    max_position_size: float | None = None
    daily_loss_limit: str | None = None
    z_score_entry_threshold: float = 1.5
    target_risk_per_trade: float = 0.01
    equity: float = 100000.0
    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.01
    gap_slippage_pct: float = 0.002


class PreprocessingConfig(BaseModel):
    clip_min: float = -10.0
    clip_max: float = 10.0
    eps: float = 1e-09
    replace_inf_nan_with: float = 0.0
    remove_prediction_bias: bool = False
    seed: int = 42


class IOConfig(BaseModel):
    row_group_size: int = 65536
    max_files: int = 20
    skip_completed: bool = True
    cache_requires_config_hash: bool = True


class PipelineConfig(BaseModel):
    enable_discovery: bool = True
    enable_expansion: bool = True
    modeling_mode: str = "minimal_compatible"
    start_stage: str = "raw"
    allow_checkpoint_start: bool = True
    checkpoint_stage: str | None = None
    checkpoint_root: str | None = None
    auto_adopt_checkpoint: bool = False


class DataSectionConfig(BaseModel):
    root: str = "data/validated"
    raw_root: str = "data/raw"
    validated_root: str = "data/validated"
    session_normalized_root: str = "data/session_normalized"
    causally_gated_root: str = "data/causally_gated_normalized"
    require_validated_files: bool = True
    forbid_raw_fallback_after_validation: bool = True
    manifest_required: bool = True
    allow_manifest_rebuild: bool = False
    data_glob: str = "data/futures/*.parquet"
    manifest_path: str = "output/manifest.json"
    baseline_features_file: str = "configs/baseline_features.yaml"
    baseline_features_persist_path: str = "output/baseline_feature_matrix.parquet"
    trades_out: str = "output/trades.csv"
    log_dir: str = "logs/"


class MemoryConfig(BaseModel):
    ram_cap_bytes: int = 14 * 1024**3  # 14 GB
    rss_stop_bytes: int = int(13.5 * 1024**3)  # 13.5 GB
    rows_per_chunk_max: int = 5_000_000
    memory_safety_margin: float = 0.95
    memory_log_enabled: bool = True


class RollPolicyConfig(BaseModel):
    method: str = "volume_or_days_before_expiry"
    days_before_expiry: int = 5
    volume_ratio_threshold: float = 1.0
    open_interest_ratio_threshold: float | None = None
    adjustment: str = "back_adjusted"
    trade_actual_contracts: bool = False
    allow_missing_roll_metadata: bool = False


class PointInTimeConfig(BaseModel):
    enabled: bool = True
    timestamp_col: str = "ts_event"
    prediction_time_col: str = "prediction_time"
    availability_time_suffix: str = "_available_at"
    fail_on_missing_availability_for: list[str] = Field(
        default_factory=lambda: ["settlement", "open_interest", "economic_release", "roll_metadata"]
    )
    allow_same_bar_execution: bool = False
    minimum_execution_lag_bars: int = 1


class LeakageAuditConfig(BaseModel):
    enabled: bool = True
    fail_on_error: bool = True
    report_dir: str = "reports/leakage"
    forbidden_feature_prefixes: list[str] = Field(default_factory=lambda: ["target_", "future_", "label_"])
    forbidden_model_metadata_prefixes: list[str] = Field(
        default_factory=lambda: ["continuous_", "roll_", "front_contract", "back_contract"]
    )
    max_allowed_feature_target_abs_corr: float = 0.999


class StressTestsConfig(BaseModel):
    enabled: bool = True
    report_dir: str = "reports/stress"
    cost_multipliers: list[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0])
    delayed_entry_bars: list[int] = Field(default_factory=lambda: [0, 1])
    adverse_fill_ticks: list[int] = Field(default_factory=lambda: [0, 1])
    remove_top_trade_percentiles: list[float] = Field(default_factory=lambda: [0.0, 0.05])
    market_ablation: bool = True
    year_ablation: bool = True


class AcceptanceGateConfig(BaseModel):
    enabled: bool = True
    report_dir: str = "reports/acceptance"
    min_oos_sharpe: float = 0.25
    min_trades: int = 30
    max_drawdown_pct: float = -0.20
    min_profit_factor: float = 1.05
    max_turnover_per_bar: float = 10.0
    require_positive_after_2x_costs: bool = True
    require_positive_after_1bar_delay: bool = True
    max_single_market_pnl_concentration: float = 0.60
    max_single_year_pnl_concentration: float = 0.60
    fail_on_leakage: bool = True
    fail_on_execution_trace_error: bool = True
    required: bool = False


class DeploymentConfig(BaseModel):
    enabled: bool = False
    mode: str = "research_only"
    paper_trading_required: bool = True
    live_shadow_required: bool = True
    require_kill_switch: bool = True
    require_post_trade_reconciliation: bool = True
    max_daily_loss: float = 1000.0
    max_drawdown_pct: float = -0.10


class RootConfig(BaseModel):
    """Master config — every parameter the system needs, with defaults."""

    # -- top-level -----------------------------------------------------------
    symbols: list[str] = Field(default_factory=lambda: DEFAULT_MARKETS.copy())
    time_zone: str = "America/New_York"
    log_level: str = "INFO"
    data_years: int = 1
    folds: int = 1
    start_year: int = 2024
    end_year: int = 2024

    # -- sections ------------------------------------------------------------
    session: SessionConfig = Field(default_factory=SessionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    walkforward: WalkforwardConfig = Field(default_factory=WalkforwardConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    data: DataSectionConfig = Field(default_factory=DataSectionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    roll_policy: RollPolicyConfig = Field(default_factory=RollPolicyConfig)
    point_in_time: PointInTimeConfig = Field(default_factory=PointInTimeConfig)
    leakage_audit: LeakageAuditConfig = Field(default_factory=LeakageAuditConfig)
    stress_tests: StressTestsConfig = Field(default_factory=StressTestsConfig)
    acceptance_gate: AcceptanceGateConfig = Field(default_factory=AcceptanceGateConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    # -- legacy flat keys retained for module compatibility ------------------
    markets: list[str] = Field(default_factory=lambda: DEFAULT_MARKETS.copy())
    market_configs: dict = Field(
        default_factory=lambda: {m: "configs/market_specs.yaml" for m in DEFAULT_MARKETS}
    )
    use_correlation_filter: bool = False
    correlation_threshold: float = 0.75
    enable_discovery: bool = True
    enable_expansion: bool = True
    rolling_wf: bool = True
    data_start_year: int = 2010
    data_end_year: int = 2026
    wf_train_days_yearly: int = 1
    wf_test_days_yearly: int = 1
    training_years: int | None = None
    walkforward_years: int | None = None
    rolling: bool | None = None
    max_markets: int | None = None


# ============================================================================
# Deep merge — nested dicts merged recursively, lists/scalars overwritten
# ============================================================================
def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ============================================================================
# env-var interpolation — replaces ${VAR} with os.environ[VAR]
# ============================================================================
_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        m = _ENV_RE.fullmatch(obj)
        if m:
            # Single placeholder — resolve or return None
            return os.environ.get(m.group(1))
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


# ============================================================================
# Time parsing helper
# ============================================================================
def _parse_time(raw: str) -> time | None:
    """Parse 'HH:MM' → datetime.time, or None on failure."""
    try:
        h, m = map(int, str(raw).split(":"))
        return time(h, m)
    except (ValueError, TypeError):
        logger.warning("Could not parse time string %r", raw)
        return None


# ============================================================================
# Populate SimpleNamespace from Pydantic RootConfig
# ============================================================================
def _populate_simple_namespace(cfg: RootConfig, active_profile: str = "", config_source: str = "") -> None:
    """Convert a validated Pydantic RootConfig into flat UPPER_SNAKE_CASE
    attributes on the module-level ``config`` SimpleNamespace.

    This is the bridge between the structured Pydantic world and the
    flat-attribute world that all quant/* modules consume.
    """
    c = cfg  # shorthand

    # -- data paths ----------------------------------------------------------
    config.DATA_ROOT = c.data.root
    config.RAW_ROOT = c.data.raw_root
    config.VALIDATED_ROOT = c.data.validated_root
    config.SESSION_NORMALIZED_ROOT = c.data.session_normalized_root
    config.CAUSALLY_GATED_ROOT = c.data.causally_gated_root
    config.REQUIRE_VALIDATED_FILES = c.data.require_validated_files
    config.FORBID_RAW_FALLBACK_AFTER_VALIDATION = c.data.forbid_raw_fallback_after_validation
    config.MANIFEST_REQUIRED = c.data.manifest_required
    config.ALLOW_MANIFEST_REBUILD = c.data.allow_manifest_rebuild
    config.DATA_GLOB = c.data.data_glob
    config.MANIFEST_PATH = c.data.manifest_path
    config.BASELINE_FEATURES_FILE = c.data.baseline_features_file
    config.BASELINE_FEATURES_PERSIST_PATH = c.data.baseline_features_persist_path
    config.TRADES_OUT = c.data.trades_out
    config.LOG_DIR = c.data.log_dir

    # -- memory --------------------------------------------------------------
    config.RAM_CAP_BYTES = c.memory.ram_cap_bytes
    config.RSS_STOP_BYTES = c.memory.rss_stop_bytes
    config.ROWS_PER_CHUNK_MAX = c.memory.rows_per_chunk_max
    config.MEMORY_SAFETY_MARGIN = c.memory.memory_safety_margin
    config.MEMORY_LOG_ENABLED = c.memory.memory_log_enabled

    # -- professional safeguards --------------------------------------------
    config.ROLL_POLICY = c.roll_policy.model_dump()
    config.ROLL_POLICY_METHOD = c.roll_policy.method
    config.ROLL_POLICY_DAYS_BEFORE_EXPIRY = c.roll_policy.days_before_expiry
    config.ROLL_POLICY_VOLUME_RATIO_THRESHOLD = c.roll_policy.volume_ratio_threshold
    config.ROLL_POLICY_OPEN_INTEREST_RATIO_THRESHOLD = c.roll_policy.open_interest_ratio_threshold
    config.ROLL_POLICY_ADJUSTMENT = c.roll_policy.adjustment
    config.ROLL_POLICY_TRADE_ACTUAL_CONTRACTS = c.roll_policy.trade_actual_contracts
    config.ROLL_POLICY_ALLOW_MISSING_ROLL_METADATA = c.roll_policy.allow_missing_roll_metadata

    config.POINT_IN_TIME = c.point_in_time.model_dump()
    config.POINT_IN_TIME_ENABLED = c.point_in_time.enabled
    config.POINT_IN_TIME_TIMESTAMP_COL = c.point_in_time.timestamp_col
    config.POINT_IN_TIME_PREDICTION_TIME_COL = c.point_in_time.prediction_time_col
    config.POINT_IN_TIME_AVAILABILITY_TIME_SUFFIX = c.point_in_time.availability_time_suffix
    config.POINT_IN_TIME_FAIL_ON_MISSING_AVAILABILITY_FOR = list(c.point_in_time.fail_on_missing_availability_for)
    config.POINT_IN_TIME_ALLOW_SAME_BAR_EXECUTION = c.point_in_time.allow_same_bar_execution
    config.POINT_IN_TIME_MINIMUM_EXECUTION_LAG_BARS = c.point_in_time.minimum_execution_lag_bars

    config.LEAKAGE_AUDIT = c.leakage_audit.model_dump()
    config.LEAKAGE_AUDIT_ENABLED = c.leakage_audit.enabled
    config.LEAKAGE_AUDIT_FAIL_ON_ERROR = c.leakage_audit.fail_on_error
    config.LEAKAGE_AUDIT_REPORT_DIR = c.leakage_audit.report_dir
    config.LEAKAGE_FORBIDDEN_FEATURE_PREFIXES = list(c.leakage_audit.forbidden_feature_prefixes)
    config.LEAKAGE_FORBIDDEN_MODEL_METADATA_PREFIXES = list(c.leakage_audit.forbidden_model_metadata_prefixes)
    config.LEAKAGE_MAX_ALLOWED_FEATURE_TARGET_ABS_CORR = c.leakage_audit.max_allowed_feature_target_abs_corr

    config.STRESS_TESTS = c.stress_tests.model_dump()
    config.STRESS_TESTS_ENABLED = c.stress_tests.enabled
    config.STRESS_TESTS_REPORT_DIR = c.stress_tests.report_dir
    config.STRESS_COST_MULTIPLIERS = list(c.stress_tests.cost_multipliers)
    config.STRESS_DELAYED_ENTRY_BARS = list(c.stress_tests.delayed_entry_bars)
    config.STRESS_ADVERSE_FILL_TICKS = list(c.stress_tests.adverse_fill_ticks)
    config.STRESS_REMOVE_TOP_TRADE_PERCENTILES = list(c.stress_tests.remove_top_trade_percentiles)

    config.ACCEPTANCE_GATE = c.acceptance_gate.model_dump()
    config.ACCEPTANCE_GATE_ENABLED = c.acceptance_gate.enabled
    config.ACCEPTANCE_GATE_REPORT_DIR = c.acceptance_gate.report_dir
    config.ACCEPTANCE_MIN_OOS_SHARPE = c.acceptance_gate.min_oos_sharpe
    config.ACCEPTANCE_MIN_TRADES = c.acceptance_gate.min_trades
    config.ACCEPTANCE_MAX_DRAWDOWN_PCT = c.acceptance_gate.max_drawdown_pct
    config.ACCEPTANCE_MIN_PROFIT_FACTOR = c.acceptance_gate.min_profit_factor
    config.ACCEPTANCE_MAX_TURNOVER_PER_BAR = c.acceptance_gate.max_turnover_per_bar
    config.ACCEPTANCE_GATE_REQUIRED = c.acceptance_gate.required

    config.DEPLOYMENT = c.deployment.model_dump()
    config.DEPLOYMENT_ENABLED = c.deployment.enabled
    config.DEPLOYMENT_MODE = c.deployment.mode
    config.DEPLOYMENT_MAX_DAILY_LOSS = c.deployment.max_daily_loss
    config.DEPLOYMENT_MAX_DRAWDOWN_PCT = c.deployment.max_drawdown_pct

    # -- session -------------------------------------------------------------
    config.TIMEZONE = c.session.timezone
    config.SESSION_START_LOCAL = _parse_time(c.session.session_start_local) or time(18, 0)
    config.SESSION_END_LOCAL = _parse_time(c.session.session_end_local) or time(16, 0)
    config.SESSION_BREAK_START_LOCAL = _parse_time(c.session.session_break_start_local) or time(17, 0)
    config.SESSION_BREAK_END_LOCAL = _parse_time(c.session.session_break_end_local) or time(18, 0)

    # -- features ------------------------------------------------------------
    config.RESAMPLE_FREQUENCIES = list(c.features.resample_frequencies)
    config.DROP_INCOMPLETE_ROWS = c.features.drop_incomplete_rows
    config.ROLL_WINDOWS = list(c.features.roll_windows)
    config.ROLL_WINDOWS_1H = list(c.features.roll_windows_1h)
    config.ROLL_WINDOWS_DAILY = list(c.features.roll_windows_daily)
    config.ROLL_WINDOW_MIN_ROWS = c.features.roll_window_min_rows
    config.FEATURE_TRANSFORMS = list(c.features.feature_transforms)
    config.MAX_PAIRWISE_INTERACTIONS = c.features.max_pairwise_interactions
    config.MAX_CROSS_TIMEFRAME_INTERACTIONS = c.features.max_cross_timeframe_interactions
    config.HTF_TREND_WINDOWS = list(c.features.htf_trend_windows)
    config.HTF_VOLATILITY_WINDOWS = list(c.features.htf_volatility_windows)
    config.HTF_ALIGNMENT_FILTER = c.features.htf_alignment_filter
    config.HTF_TREND_THRESHOLD = c.features.htf_trend_threshold
    config.VOL_MEDIAN_WINDOW = c.features.vol_median_window
    config.VOL_SMOOTH_WINDOW = c.features.vol_smooth_window
    config.REGIME_HIGH_THRESH = c.features.regime_high_thresh
    config.REGIME_LOW_THRESH = c.features.regime_low_thresh
    config.REGIME_MISSING_DEFAULT = c.features.regime_missing_default

    # -- target --------------------------------------------------------------
    config.TARGET_15M_HORIZON = c.target.target_15m_horizon
    config.TARGET_SCALE_FACTOR = c.target.target_scale_factor

    # -- discovery -----------------------------------------------------------
    config.DISCOVERY_WINDOW_DAYS = c.discovery.discovery_window_days
    config.BOOTSTRAP_FOLDS = c.discovery.bootstrap_folds
    config.EXTRA_TREES_PARAMS = dict(c.discovery.extra_trees_params)
    config.SELECTION_FREQ_THRESHOLD = c.discovery.selection_freq_threshold
    config.SIGN_CONSISTENCY_THRESHOLD = c.discovery.sign_consistency_threshold
    config.CUMULATIVE_IMPORTANCE_THRESHOLD = c.discovery.cumulative_importance_threshold
    config.MIN_SELECTED_FEATURES = c.discovery.min_selected_features
    config.MAX_SELECTED_FEATURES = c.discovery.max_selected_features

    # -- walkforward ---------------------------------------------------------
    config.WF_TRAIN_DAYS = c.walkforward.wf_train_days
    config.WF_TEST_DAYS = c.walkforward.wf_test_days
    config.WF_STEP_DAYS = c.walkforward.wf_step_days
    config.RIDGE_PARAMS = dict(c.walkforward.ridge_params)
    config.MODEL_TYPE = c.walkforward.model_type
    config.PROBABILITY_SMOOTHING_ALPHA = c.walkforward.probability_smoothing_alpha
    config.CORR_THRESHOLD = c.walkforward.corr_threshold
    config.WF_PARALLEL_FOLDS = c.walkforward.wf_parallel_folds
    config.BURN_IN_BARS = c.walkforward.burn_in_bars
    config.ENABLE_META_LABELING = c.walkforward.enable_meta_labeling
    config.META_THRESHOLD = c.walkforward.meta_threshold
    config.WF_MODE = c.walkforward.mode
    config.DISCOVERY_TARGET = getattr(c.walkforward, 'discovery_target', 'target_15m_ret')
    config.WALKFORWARD_TARGET = getattr(c.walkforward, 'walkforward_target', c.walkforward.discovery_target)
    config.WF_EMBARGO_BARS = c.walkforward.embargo_bars
    config.WF_PURGE_TARGET_OVERLAP = c.walkforward.purge_target_overlap

    # -- execution -----------------------------------------------------------
    config.EXECUTE_AT = c.execution.execute_at
    config.SLIPPAGE_K = c.execution.slippage_k
    config.VOL_PENALTY = c.execution.vol_penalty
    config.COMMISSION_PER_TRADE = c.execution.commission_per_trade
    config.TX_COST_PER_ROUNDTURN = c.execution.tx_cost_per_roundturn
    config.COMMISSION_PER_CONTRACT = c.execution.commission_per_contract
    config.EXCHANGE_FEES_PER_CONTRACT = c.execution.exchange_fees_per_contract
    config.EXECUTION_ENTRY_LAG_BARS = c.execution.entry_lag_bars
    config.SLIPPAGE_TICKS = c.execution.slippage_ticks
    config.SPREAD_TICKS = c.execution.spread_ticks
    config.PARTICIPATION_RATE_LIMIT = c.execution.participation_rate_limit
    config.EXECUTION_MAX_CONTRACTS = c.execution.max_contracts
    config.EXECUTION_LATENCY_BARS = c.execution.latency_bars
    config.REJECT_SAME_BAR_FILL = c.execution.reject_same_bar_fill
    config.PREDICTION_ENTRY_THRESHOLD = c.execution.prediction_entry_threshold
    config.MIN_POSITION_HOLD_BARS = c.execution.min_position_hold_bars
    config.WRITE_EXECUTION_TRACE = c.execution.write_execution_trace
    config.EXECUTION_TRACE_ROWS = c.execution.execution_trace_rows
    config.TARGET_VOL = c.execution.target_vol
    config.MAX_LEVERAGE = c.execution.max_leverage
    config.MAX_POS_CHANGE_PER_MIN = c.execution.max_pos_change_per_min
    config.FLAT_BEFORE_CLOSE_MINUTES = c.execution.flat_before_close_minutes
    config.HTF_TREND_ALIGNMENT = c.execution.htf_trend_alignment
    config.HTF_VOL_SCALING = c.execution.htf_vol_scaling
    config.HTF_VOL_WINDOW = c.execution.htf_vol_window
    config.Z_SCORE_ENTRY_THRESHOLD = c.execution.z_score_entry_threshold
    config.TARGET_RISK_PER_TRADE = c.execution.target_risk_per_trade
    config.EQUITY = c.execution.equity
    config.STOP_LOSS_PCT = c.execution.stop_loss_pct
    config.TAKE_PROFIT_PCT = c.execution.take_profit_pct
    config.GAP_SLIPPAGE_PCT = c.execution.gap_slippage_pct
    config.MAX_POSITION_SIZE = (
        float(c.execution.max_position_size)
        if c.execution.max_position_size is not None
        else float('inf')
    )

    # -- preprocessing -------------------------------------------------------
    config.CLIP_MIN = c.preprocessing.clip_min
    config.CLIP_MAX = c.preprocessing.clip_max
    config.EPS = c.preprocessing.eps
    config.REPLACE_INF_NAN_WITH = c.preprocessing.replace_inf_nan_with
    config.REMOVE_PREDICTION_BIAS = c.preprocessing.remove_prediction_bias
    config.SEED = c.preprocessing.seed

    # -- io ------------------------------------------------------------------
    config.ROW_GROUP_SIZE = c.io.row_group_size
    config.MAX_FILES = c.io.max_files
    config.SKIP_COMPLETED = c.io.skip_completed
    config.CACHE_REQUIRES_CONFIG_HASH = c.io.cache_requires_config_hash

    # -- pipeline ------------------------------------------------------------
    config.ENABLE_DISCOVERY = c.pipeline.enable_discovery
    config.ENABLE_EXPANSION = c.pipeline.enable_expansion
    config.MODELING_MODE = c.pipeline.modeling_mode
    config.START_STAGE = c.pipeline.start_stage
    config.ALLOW_CHECKPOINT_START = c.pipeline.allow_checkpoint_start
    config.CHECKPOINT_STAGE = c.pipeline.checkpoint_stage
    config.CHECKPOINT_ROOT = c.pipeline.checkpoint_root
    config.AUTO_ADOPT_CHECKPOINT = c.pipeline.auto_adopt_checkpoint

    # -- legacy flat keys ----------------------------------------------------
    config.MARKETS = list(c.markets)
    config.MARKET_CONFIGS = dict(c.market_configs)
    config.USE_CORRELATION_FILTER = c.use_correlation_filter
    config.CORRELATION_THRESHOLD = c.correlation_threshold
    config.ROLLING_WF = c.rolling_wf
    config.DATA_START_YEAR = c.data_start_year
    config.DATA_END_YEAR = c.data_end_year
    config.START_YEAR = c.start_year
    config.END_YEAR = c.end_year
    config.WF_TRAIN_DAYS_YEARLY = c.wf_train_days_yearly
    config.WF_TEST_DAYS_YEARLY = c.wf_test_days_yearly
    config.MAX_MARKETS = c.max_markets

    # -- config identity (set by loader) ----------------------------------
    config.ACTIVE_PROFILE = active_profile
    config.CONFIG_SOURCE = config_source


# ============================================================================
# Config resolution — locations for YAML files
# ============================================================================
_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# ============================================================================
# Public API
# ============================================================================
def load_config(env: str | None = None, configs_dir: Path | None = None) -> RootConfig:
    """
    Load hierarchical config, validate with Pydantic, populate SimpleNamespace.

    Primary path (profile-based):
      1. Read configs/alpha_tiered.yaml
      2. Identify active_profile from CONFIG_ENV env var or alpha_tiered.yaml default
      3. Deep-merge ``base`` + ``profiles[active_profile]``
      4. Resolve ${ENV_VAR} placeholders
      5. Validate with Pydantic RootConfig model
      6. Populate module-level ``config`` SimpleNamespace

    Fallback path (legacy flat YAML):
      Reads alpha_0.yaml, deep-merges tier YAML as before.

    Args:
        env: Override profile name (overrides default, overridden by CONFIG_ENV).
        configs_dir: Override configs directory (default: project-root/configs/)

    Returns:
        Validated RootConfig (Pydantic model) for structured access.
    """
    global _LOADED
    if _LOADED:
        return None

    base_dir = configs_dir or _CONFIGS_DIR
    profile_env = os.environ.get("CONFIG_ENV") or os.environ.get("QUANT_ENV")
    profile_override = env or profile_env

    # ---- Primary: profile-based alpha_tiered.yaml --------------------------
    alpha_path = base_dir / "alpha_tiered.yaml"
    if alpha_path.exists():
        with open(alpha_path) as f:
            raw = yaml.safe_load(f) or {}

        if isinstance(raw, dict) and "active_profile" in raw and "base" in raw and "profiles" in raw:
            active = profile_override or raw["active_profile"]
            profiles_section = raw.get("profiles", {})
            if active not in profiles_section:
                available = list(profiles_section.keys())
                raise ValueError(
                    f"active_profile '{active}' not found in alpha_tiered.yaml profiles. "
                    f"Available: {available}"
                )

            merged = copy.deepcopy(raw["base"])
            _deep_merge(merged, profiles_section[active])
            merged = _resolve_env_vars(merged)

            if "symbols" in merged and isinstance(merged["symbols"], list):
                if "market_configs" not in merged:
                    merged["market_configs"] = {}
                for m in merged["symbols"]:
                    merged["market_configs"][m] = "configs/market_specs.yaml"

            try:
                root_cfg = RootConfig(**merged)
            except ValidationError as e:
                raise ValueError(
                    f"Config validation failed (alpha_tiered.yaml profile={active}): {e}"
                ) from e

            config_source = f"{alpha_path.name}::{active}"
            _populate_simple_namespace(root_cfg, active_profile=active, config_source=config_source)

            _LOADED = True
            logger.info("Configuration loaded from alpha_tiered.yaml (profile=%s)", active)
            return root_cfg

    # ---- Fallback: legacy flat YAML tier loading ----------------------------
    if profile_override is None:
        profile_override = "alpha_1"

    base_path = base_dir / "alpha_0.yaml"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Config not found: alpha_tiered.yaml or alpha_0.yaml in {base_dir}"
        )

    with open(base_path) as f:
        merged: dict = yaml.safe_load(f) or {}

    tier_name = "alpha_4" if profile_override == "production" else profile_override
    tier_path = base_dir / f"{tier_name}.yaml"
    if tier_path.exists():
        with open(tier_path) as f:
            tier_cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, tier_cfg)

    merged = _resolve_env_vars(merged)

    if "symbols" in merged and isinstance(merged["symbols"], list):
        if "market_configs" not in merged:
            merged["market_configs"] = {}
        for m in merged["symbols"]:
            merged["market_configs"][m] = "configs/market_specs.yaml"

    try:
        root_cfg = RootConfig(**merged)
    except ValidationError as e:
        raise ValueError(
            f"Config validation failed for env '{profile_override}': {e}"
        ) from e

    config_source = f"flat::{profile_override}"
    _populate_simple_namespace(root_cfg, active_profile=profile_override, config_source=config_source)

    _LOADED = True
    logger.info("Configuration loaded (env=%s)", profile_override)
    return root_cfg


def load_env_config() -> RootConfig:
    """Convenience: reads tier from CONFIG_ENV or QUANT_ENV environment variable."""
    return load_config(os.environ.get("CONFIG_ENV") or os.environ.get("QUANT_ENV"))
