python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Active config
$env:CONFIG_ENV="tier_0_smoke_pipeline"
# config file: configs/alpha_tiered.yaml

# Smoke/safety validation mode
# pipeline.modeling_mode: minimal_compatible
# Validates pipeline wiring and safety gates only. Do not interpret as alpha.
# Deployment readiness remains NOT_READY and no strategy can be production-ready.
python run.py

# Full research mode
# pipeline.modeling_mode: full_research
# Runs train-window feature selection, train-only scaling, OOS prediction,
# execution/cost modeling, and the same leakage/trace/metrics/stress/
# acceptance/deployment gates. Missing full-research modules fail fast.
python run.py

# Phase 1 data/report workflow
# Raw data contract: external ingest writes data/raw/{market}/{year}.parquet
# with ts_event, open, high, low, close, volume; contract/roll metadata optional.
# Session normalization now reads the merged per-market session/validation
# config from configs/raw_data_validation.yaml by default.
python scripts/validate_databento_continuous.py --audit-only
python scripts/validate_databento_continuous.py --write-validated --clean-policy drop-invalid
python scripts/build_data_manifests.py --stages raw validated session_normalized causally_gated_normalized labeled
python scripts/session_normalize.py
python scripts/causal_gate_normalized.py
python -m pipeline.audit.data_quality --root data/validated --out reports/validation/data_quality_report.json
python -m pipeline.audit.session_roll
python -m pipeline.audit.pipeline_coverage --config configs/alpha_tiered.yaml --strict

# Research run uses the configured pipeline.modeling_mode.
python run.py

# CLI help
python -m pipeline.cli --help
python -m pipeline.cli discover --help
python -m pipeline.cli run --help
python -m pipeline.cli aggregate --help

# Hard gate
Downstream research refuses to run from data/validated if parquet files are
missing or manifests are missing. Remediation:
python scripts/validate_databento_continuous.py --write-validated --clean-policy drop-invalid

# Review
reports/leakage/
reports/metrics/
reports/stress/
reports/acceptance/
artifacts/run_manifests/

# Mode semantics
# minimal_compatible: smoke/safety validation only; not alpha evidence.
# full_research: actual alpha research path with train-only selection/scaling.
# deployment.mode=research_only keeps deployment readiness NOT_READY.
# Bad strategies should be rejected by acceptance gates, not crash the pipeline.

# Starting from existing causally gated normalized data
# 1. Adopt existing checkpoint folder into canonical layout.
python -m pipeline.data.adopt_checkpoint --stage causally_gated_normalized --source path/to/my_existing_causal_data --target data/causally_gated_normalized --copy

# 2. Validate checkpoint.
python -m pipeline.data_gate.checkpoint --stage causally_gated_normalized --root data/causally_gated_normalized

# 3. Run downstream from stage 9.
python run.py --from-stage causally_gated_normalized --data-root data/causally_gated_normalized

# Equivalent environment form:
$env:QUANT_START_STAGE="causally_gated_normalized"
$env:QUANT_DATA_ROOT="data/causally_gated_normalized"
python run.py

# Checkpoint semantics:
# - stages 1-8 are skipped as SKIPPED_CHECKPOINT and recorded in run manifest.
# - causally_gated_normalized must contain prediction_time and earliest_execution_time.
# - if prediction_time is missing, start from session_normalized instead:
python run.py --from-stage session_normalized --data-root data/session_normalized
# - if only validated data exists:
python run.py --from-stage validated --data-root data/validated

# Using existing normalized or causally gated data
# Auto-detect the safest valid stage:
python -m pipeline.data.classify_checkpoint --source path/to/existing_folder

# Auto-adopt into the correct canonical data/<stage> root:
python -m pipeline.data.adopt_checkpoint --stage auto --source path/to/existing_folder --target-root data --copy

# Auto-run from the inferred stage:
python run.py --from-stage auto --data-root path/to/existing_folder

# Manual routing from classifier output:
# If classifier says validated:
python run.py --from-stage validated --data-root data/validated
# If classifier says session_normalized:
python run.py --from-stage session_normalized --data-root data/session_normalized
# If classifier says causally_gated_normalized:
python run.py --from-stage causally_gated_normalized --data-root data/causally_gated_normalized

# Safety notes:
# - missing session_id means the data is not session-normalized.
# - missing prediction_time means the data is not causally gated.
# - the pipeline will not force data into a later stage; false checkpoint confidence is rejected.

# Artifact layout
data/raw -> data/validated -> data/session_normalized -> data/causally_gated_normalized
reports/validation -> reports/session_normalization -> reports/causal_gating -> reports/wfa -> reports/metrics
artifacts/models -> artifacts/scalers -> artifacts/selectors -> artifacts/run_manifests -> artifacts/backtests

# Config ownership
# configs/raw_data_validation.yaml: merged raw validation + per-market session calendar policy.
# configs/market_specs.yaml: contract specs, tick value/size, multiplier, risk defaults.
# configs/alpha_tiered.yaml: active research/profile/runtime config.
# archive/deprecated_configs/market_sessions.yaml: old simple fallback example, not active.
