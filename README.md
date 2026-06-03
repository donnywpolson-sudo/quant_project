python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Active config
$env:CONFIG_ENV="tier_0_smoke_pipeline"
# config file: configs/alpha_tiered.yaml

# Phase 1 data/report workflow
python scripts/validate_databento_continuous.py --audit-only
python scripts/validate_databento_continuous.py --write-validated --clean-policy drop-invalid
python scripts/session_normalize.py
python scripts/causal_gate_normalized.py
python scripts/build_data_manifests.py --stages raw validated session_normalized causally_gated_normalized

# Research run
python run.py

# Artifact layout
data/raw -> data/validated -> data/session_normalized -> data/causally_gated_normalized
reports/validation -> reports/session_normalization -> reports/causal_gating -> reports/wfa -> reports/metrics
artifacts/models -> artifacts/scalers -> artifacts/selectors -> artifacts/run_manifests -> artifacts/backtests
