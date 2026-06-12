# Quant Project Layout

## How to read this document

This file is the project contract. Use it to check what each pipeline phase
must read, write, validate, and refuse to trust.

Fast map:

- Part 1: current operating model, active configs, artifact flow, and research policy.
- Part 2: phase-by-phase build requirements and acceptance checks.
- End sections: provenance, known limitations, required tests, build priority, and Codex implementation prompt.

Stable contracts to preserve:

- file paths
- config keys
- output column names
- phase names
- gate meanings
- train/test/final-holdout separation rules

---

## Part 1 — Pipeline Structure Overview

### Project goal

Build an intraday futures quant trading model for an individual day trader using Databento continuous-contract 1-minute OHLCV parquet data.

The system is designed for:

- intraday trading only;
- no overnight holds;
- OHLCV-only baseline and expanded features;
- realistic target/execution alignment;
- market-specific tick, point-value, commission, and slippage accounting;
- walk-forward out-of-sample testing;
- train-only preprocessing, feature ranking, feature selection, and policy selection;
- final holdout evaluation that is not used for feature or policy decisions;
- prop-firm-style account and rule simulation;
- explicit accept/reject gates that separate pipeline integrity from economic viability.

The pipeline should first prove that the data, labels, features, walk-forward splits, execution, costs, and gates are structurally correct. Economic profitability is a separate gate and may fail even when the pipeline passes.

---

### Raw ingest contract

Phase 1 is a two-step raw-ingest workflow:

- Phase 1A downloads and archives immutable Databento DBN/DBN.ZST chunks.
- Phase 1B validates OHLCV plus definition DBN chunks and converts market-year OHLCV chunks into immutable 1-minute OHLCV parquet.
- Phase 2 validates, session-normalizes, roll-audits, synthetic-marks, and causally gates the parquet.

DBN archive path pattern:

```text
data/raw/{market}/{year}.dbn.zst
data/raw/definition/{market}/{year}.dbn.zst
```

Raw parquet path pattern:

```text
data/raw/{market}/{year}.parquet
```

Required raw schema:

```text
source_dataset = GLBX.MDP3
source_schema = ohlcv-1m
ts_event
open
high
low
close
volume
rtype
publisher_id
instrument_id
symbol
data_quality_status
data_quality_degraded
raw_symbol
tick_size
source_file
source_sha256
```

Production and research profiles fail when strict raw fields are missing. The
`metadata_optional_test` profile is relaxed for test fixtures only; missing
strict raw fields are FAIL, not WARN.

Timestamp policy:

```text
Phase 1 preserves ts_event.
Phase 2 converts ts_event -> ts.
```

Databento metadata columns must be preserved for audit but excluded from model features by default:

```text
rtype
publisher_id
instrument_id
symbol
```

Important realism note:

```text
Continuous contracts are research series, not directly tradeable instruments.
```

Raw ingest must preserve `rtype`, `publisher_id`, `instrument_id`, and `symbol`.
Phase 1 must not rename `ts_event`, must not fill missing 1-minute bars, and
must fail if required schema or metadata is missing or fake-filled. Missing-bar
repair is Phase 2-only and must be marked synthetic. The pipeline must report
symbol and instrument changes, flag roll boundaries, invalidate suspicious
roll-window labels, and eventually support contract-specific execution mapping
before live deployment.

---

### Active config policy

Use the active tiered config:

```text
configs/alpha_tiered.yaml
```

Market/session/cost details live in:

```text
configs/markets/{market}.yaml
configs/market_sessions.yaml
configs/costs.yaml
configs/prop_rules.yaml
```

Current implementation organization:

```text
scripts/phase1A_download/        DBN archive download
scripts/phase1B_convert/         DBN-to-raw-parquet conversion
scripts/phase2_causal_base/      validation, session normalization, causal gating
scripts/phase3_labels/           target and label generation
scripts/phase4_features/         baseline features; will be created later
scripts/utilities/               repo safety utilities
```

Operational profile model:

```text
tier_0 = ES smoke test
tier_1 = CL/ES/ZN recent core
tier_2 = CL/ES/ZN long core
tier_3 = real full universe
all_raw = inventory only
metadata_optional_test = tests only
```

`tier_1` is for frequent CL/ES/ZN iteration. `tier_2` is the same core over
long history. `tier_3` is the actual 27-market GLBX-only research universe. `tier_1` results do not prove `tier_3`
performance. Missing Tier-3 data must fail stage validation clearly; the
pipeline must not shrink the Tier-3 universe to whatever data exists. `all_raw`
is inventory only and must not feed labels, WFA, gates, or research decisions.

Each market config should include:

```text
market
tick_size
tick_value
point_value
session_template
intraday_flatten_time
roll_exclusion_bars
```

Tier-1 machinery proof universe:

```text
CL
ES
ZN
```

Tier-3 real research universe:

```text
ES NQ RTY YM VX
CL NG RB HO
GC SI HG
SR3 ZN ZB
6A 6B 6C 6E 6J 6M 6N 6S
ZC ZS ZW LE HE
```

Year/profile policy:

```text
downloaded_years = 2010-2026
recent_research = 2023-2025
long_research = 2010-2025
forward_years = 2026
research_years = 2023-2024
final_holdout_years = 2025
entry_lag_bars = 1
target_horizon_bars = 15
trend_horizon_bars = 30
purge_bars = auto
resolved_purge_bars = entry_lag_bars + target_horizon_bars
default resolved_purge_bars = 16
```

Features use completed bar `t`, entry occurs on bar `t+1`, and the 15-minute
target exits at `t+1+15`, so the target touches through `t+16`. Purge must cover
that full dependency window; hardcoded 15-bar purge is not valid for this
alignment.

Profiles:

```text
tier_0
tier_1
tier_2
tier_3
metadata_optional_test test-only
all_raw inventory-only
```

2026 is forward/incomplete-year validation, not normal training history.

Do not add new markets, alternative data, order book data, or hyperparameter tuning until the baseline pipeline is structurally correct and the strategy gates are honest.

---

### Pipeline structure overview

This is the realistic operational pipeline. The old 27-stage checklist is preserved conceptually, but implementation is consolidated into fewer runnable stages with stronger reports and gates.

| Phase | Name | Main artifact | Purpose |
|---:|---|---|---|
| 1A | DBN Archive | `data/raw/{market}/{year}.dbn.zst` plus `data/raw/definition/{market}/{year}.dbn.zst` | Download and archive immutable Databento OHLCV and definition DBN/DBN.ZST market-year chunks. |
| 1B | Raw Parquet Stitch | `data/raw/{market}/{year}.parquet` | Validate OHLCV plus definition DBN chunks and convert market-year OHLCV chunks into immutable parquet. |
| 2 | Causal Base Builder | `data/causally_gated_normalized/{market}/{year}.parquet` | Validate, session-normalize, roll-flag, synthetic-mark, and causally gate raw bars. |
| 3 | Target / Label Generation | `data/labeled/{market}/{year}.parquet` | Build next-bar-entry 15-minute labels with cost-aware and intraday validity flags. |
| 4 | Baseline + L0 Regime Feature Matrix | `data/feature_matrices/baseline/{market}/{year}.parquet` | Build OHLCV-only baseline and L0 regime features plus metadata, target, and initial registry columns. |
| 5 | Column Registry | `feature_cols.json`, `target_cols.json`, `metadata_cols.json` | Audit, freeze, and promote feature/target/metadata separation for WFA. |
| 6 | WFA Split Plan | `reports/wfa/split_plan.json` | Build deterministic train/test folds with purge and final-holdout awareness. |
| 7A | Linear Control WFA | `reports/wfa/baseline_wfa_report.json` | Train Ridge/logistic control models using train-only preprocessing and test-only prediction. |
| 7B | Sklearn Nonlinear Challenger WFA | `reports/wfa/baseline_wfa_report.json` | Compare HistGradientBoosting challengers without changing split discipline. |
| 7C | Optional Boosted-Tree Challenger WFA | `reports/wfa/baseline_wfa_report.json` | Optionally compare CPU-first LightGBM/XGBoost challengers without requiring those dependencies. |
| 8 | Baseline OOS Predictions | `data/predictions/baseline/oos_predictions.parquet` | Store multi-model OOS predictions with raw and calibrated scores plus execution-ready prices. |
| 8A | Signal Calibration + Model Comparison | `reports/model_selection/` | Compare models and calibration using train-fold-only fitting; exclude final holdout from selection. |
| 9 | Baseline Execution + Cost Model | `data/executions/baseline/executions.parquet` | Convert calibrated model scores into positions, apply costs, enforce no-overnight rules. |
| 10 | Baseline Metrics + Diagnostics | `reports/metrics/baseline_metrics.json` | Evaluate prediction behavior, turnover, cost drag, and net economics. |
| 11 | Baseline Accept / Reject Gate | `reports/gates/baseline_gate.json` | Separate structural pass/fail from economic pass/fail. |
| 12 | Feature Expansion | `data/feature_matrices/expanded/{market}/{year}.parquet` | Add broader OHLCV-only candidates. |
| 13 | Feature Discovery | `reports/feature_discovery/` | Analyze nulls, stability, redundancy, leakage, and train-only correlations. |
| 14 | Train-Only Feature Ranking / Selection | `reports/feature_selection/` | Select features without using final holdout or test-fold information. |
| 15 | Frozen Feature + Model + Calibration + Policy Set | `data/frozen_features/phase5_v1/`, `data/frozen_models/phase5_v1/` | Freeze features, model config, calibration config, and execution policy before final evaluation. |
| 16 | Final Holdout Split Plan | `reports/final_wfa/final_split_plan.json` | Ensure final test windows are entirely inside final holdout. |
| 17 | Final WFA With Frozen Features | `reports/final_wfa/final_wfa_report.json` | Evaluate frozen model setup on final holdout. |
| 18 | Final OOS Predictions | `data/predictions/final/oos_predictions.parquet` | Store final OOS predictions with execution-ready prices. |
| 19 | Final Execution + Cost Model | `data/executions/final/executions.parquet` | Apply frozen execution and cost rules to final predictions. |
| 20 | Final Metrics + Diagnostics | `reports/final_metrics/` | Compare final against baseline, placebo, stress, and simple-rule baselines. |
| 21 | Prop-Firm Account Simulation | `reports/prop_sim/` | Test daily loss, trailing drawdown, max contracts, forced flat, and payout viability. |
| 22 | Strategy Accept / Reject Gate | `reports/gates/strategy_gate.json` | Final decision gate. Must reject negative-net or rule-violating strategies. |

---

### Legacy stage mapping

```text
Old stages 1-8   -> Phase 1A-2: DBN archive, raw parquet stitch, and causal base builder
Old stages 9-13  -> Phase 3-5: labels, baseline feature matrix, registry
Old stages 14-19 -> Phase 6-11: WFA, predictions, calibration/model comparison, execution, metrics, baseline gate
Old stages 20-23 -> Phase 12-15: feature expansion, discovery, selection, frozen set
Old stages 24-27 -> Phase 16-22: final holdout, final WFA, final predictions, final execution, final metrics, prop simulation, strategy gate
```

---

### Core artifact flow

```text
data/raw/{market}/{year}.dbn.zst
data/raw/definition/{market}/{year}.dbn.zst
-> data/raw/{market}/{year}.parquet
-> data/causally_gated_normalized/{market}/{year}.parquet
-> data/labeled/{market}/{year}.parquet
-> data/feature_matrices/baseline/{market}/{year}.parquet
-> reports/wfa/split_plan.json
-> data/predictions/baseline/oos_predictions.parquet
-> data/executions/baseline/executions.parquet
-> reports/metrics/baseline_metrics.json
-> reports/gates/baseline_gate.json
-> data/feature_matrices/expanded/{market}/{year}.parquet
-> reports/feature_discovery/
-> reports/feature_selection/
-> data/frozen_features/phase5_v1/feature_cols.json
-> data/frozen_models/phase5_v1/model_config.yaml
-> data/frozen_features/phase5_v1/policy_config.json
-> reports/final_wfa/final_split_plan.json
-> data/predictions/final/oos_predictions.parquet
-> data/executions/final/executions.parquet
-> reports/final_metrics/final_metrics.json
-> reports/prop_sim/final_prop_simulation.json
-> reports/gates/strategy_gate.json
```

---

### Downstream ML model policy

The pipeline treats model choice as a staged research comparison, not a single
hardcoded model.

Required staged order:

- Phase 7A linear controls: Ridge/logistic benchmark models.
- Phase 7B HistGradientBoosting challengers: first sklearn nonlinear comparison.
- Phase 7C optional LightGBM/XGBoost challengers: optional CPU-first boosted-tree comparison.
- Phase 8A calibration/model comparison: train-fold-only calibration and model reporting.
- Phase 15: frozen feature + model + calibration + policy set.

Do not use neural nets, transformers, or reinforcement learning until simpler
tabular models survive WFA, costs, turnover, final holdout, and prop-firm
simulation.

Approved initial model families:

```text
ridge_return
logistic_direction
logistic_fade_success
logistic_trend_danger
hist_gradient_boosting_direction
hist_gradient_boosting_fade_success
hist_gradient_boosting_trend_danger
lightgbm_direction_optional
xgboost_direction_optional
```

All models must use:

- train-fold-only fitting
- train-only imputation
- train-only scaling where applicable
- test-fold-only prediction
- no random train/test split
- no final-holdout tuning
- `model_id` recorded in all prediction and metric reports
- model config hash recorded in all prediction and metric reports
- feature config hash recorded in all prediction and metric reports

Ridge/logistic models are controls. HistGradientBoosting is the first nonlinear
challenger. LightGBM/XGBoost are optional downstream challengers, disabled by
default, and their external dependencies must not be required for baseline tests
to pass. The first priority classifier is the trend-danger / do-not-fade classifier
because fade strategies are vulnerable to real trend days.

---

### Research, validation, and forward policy

A walk-forward split alone is not enough if feature selection, policy selection, and human iteration keep using the same period.

Current year policy:

```text
downloaded_years = 2010-2026
recent_research = 2023-2025
long_research = 2010-2025
forward_years = 2026
```

Rules:

- Feature discovery and feature selection may use only research-period train folds.
- Execution policy selection may use only research-period train/validation folds.
- Forward years may be evaluated only after features and policy are frozen.
- Do not change features, thresholds, policy rules, or costs after inspecting forward-year results.
- 2026 is forward/incomplete-year validation, not normal training history.
- If forward validation fails, mark the strategy rejected; do not tune on forward results.

When more history is available, prefer:

```text
research:   older years
validation: middle years
final:      latest untouched year
paper/live: future forward test
```

---

### Non-negotiable design rules

#### Data rules

- Raw data is immutable.
- `ts_event` becomes `ts`.
- Files are processed with year-boundary bleed so sessions, rolling windows, and target horizons do not break at calendar-year boundaries.
- Continuous-contract roll changes are reported using `symbol` and `instrument_id` changes.
- Suspicious roll-window labels are invalidated or separately reported.
- Synthetic rows may be inserted only inside valid session segments.
- Synthetic rows are never trainable.
- No forward-fill across `session_segment_id`.
- Labels crossing synthetic rows, session boundaries, cutoff boundaries, or suspicious roll boundaries must be invalid.

#### Target and execution rules

- Features use completed bar `t`.
- Prediction is made after bar `t` is complete.
- Entry occurs on bar `t+1`.
- Exit for the default 15-minute target occurs at `t+1+15`.
- Label price convention must match the execution model.
- Targets must include both directional and cost-aware fields.
- Tiny moves that cannot overcome estimated costs must not be treated as tradeable opportunities.

#### Intraday trading rules

- No overnight holds.
- Position must be flat before the configured intraday cutoff.
- Labels are invalid if the full entry-to-exit horizon cannot complete before the cutoff.
- For a 15-minute target with next-bar execution, require enough time for entry and exit.

#### Feature rules

- Baseline and expanded features are OHLCV-only.
- Databento metadata IDs are excluded from features.
- Features must be grouped by `session_segment_id` where rolling logic is used.
- Feature families should be diverse, but uncorrelated status must be verified by train-fold-only Pearson and Spearman correlation clustering.
- Do not keep redundant indicators just because they have different names.
- Do not use final holdout results to choose features.

#### Modeling rules

- Walk-forward analysis only.
- No random train/test splits.
- No full-sample fitting.
- Imputer fit on train only.
- Scaler fit on train only.
- Model fit on train fold only.
- Predictions are generated on test fold only.
- Feature ranking, feature selection, and policy selection are train-only or research-period-only.
- Final WFA uses frozen features, frozen model config, frozen calibration config,
  and frozen policy only.

#### Execution and cost rules

- Raw predictions are not positions.
- A deterministic position policy converts predictions into signals and positions.
- Policy comparison is allowed, but the final policy must be selected without final-holdout information.
- Costs are charged on absolute position change.
- Long-to-short and short-to-long flips cost `2.0` units of position change.
- PnL must be reported in return units, ticks, and dollars.
- Market-specific tick value, point value, commission, and slippage assumptions must be used.
- Strategy gates must fail or warn if costs dominate gross edge.

#### Prop-firm realism rules

- Simulate starting balance, profit target, daily loss limit, trailing drawdown, max contracts, minimum trading days, and forced flatten time.
- A strategy that violates prop rules is not viable even if aggregate net return is positive.
- Report days-to-payout, rule violations, worst intraday drawdown, and probability of drawdown-before-target under scenario tests.

#### Gate rules

- Pipeline integrity can pass while economic viability fails.
- A negative-net strategy must not be labeled as accepted.
- A strategy that fails prop-rule simulation must not be labeled as accepted for the user's actual use case.
- Cost drag, turnover, trade count, drawdown, and rule violations must be explicit gate inputs.
- Metrics must be reported overall, by market, by year, by fold, by market-year, by policy, and by cost-stress scenario.

---

### Baseline OHLCV feature families

The baseline feature set should start with diverse OHLCV-only families. The goal is not to assume these are uncorrelated; the goal is to generate low-redundancy candidates and let train-only correlation clustering select survivors.

Initial candidate families:

```text
returns / momentum
range / volatility
bar shape / candle structure
volume / participation
VWAP / fair-value distance
position in recent range
EMA / trend state
time of day / session state
opening range
session state so far
compression / expansion
liquidity / friction proxies
```

Recommended initial baseline candidates:

```text
ret_1
ret_5
ret_30
realized_vol_15
volatility_ratio_15_60
range_z_60
close_position_in_bar
body_to_range
wick_imbalance
volume_z_60
volume_ratio_15_60
volume_rel_to_tod
dist_session_vwap
dist_rolling_vwap_15
dist_rolling_vwap_60
pos_in_15m_range
pos_in_60m_range
dist_ema_15
slope_ema_60
ema_stack_15_60
minute_of_session_sin
minute_of_session_cos
session_progress
minutes_until_session_close
opening_range_width
dist_from_opening_range_high
dist_from_opening_range_low
session_return_from_open
position_in_session_range
range_ratio_5_60
inside_bar_flag
outside_bar_flag
range_per_volume
abs_return_per_volume
```

After train-only correlation clustering, expect roughly 15-25 features to survive.

---

### Baseline target and execution alignment

Do not train on a target that assumes execution at the same bar close used to compute features.

Preferred alignment:

```text
features use completed bar t
prediction is made after bar t is complete
entry execution occurs on bar t+1
15-minute exit is based on t+1+15
```

Acceptable label implementation:

```text
target_entry_price = open[t+1] or conservative close[t+1]
target_exit_price  = open[t+16] or conservative close[t+16]
target_ret_15m     = target_exit_price / target_entry_price - 1
```

The exact price convention must match the execution-cost model.

Required target fields:

```text
target_invalid_reason
target_ret_15m
target_ret_ticks_15m
target_gross_dollars_15m
target_estimated_cost_ticks
target_estimated_cost_dollars
target_net_ticks_after_est_cost
target_net_dollars_after_est_cost
target_sign_15m
target_sign_with_deadzone
target_tradeable_after_cost
target_valid
mae_ticks_15m
mfe_ticks_15m
fade_long_success_15m
fade_short_success_15m
trend_danger_up_30m
trend_danger_down_30m
revert_to_vwap_30m
revert_to_session_mid_30m
label_semantics
cost_source
cost_provisional
```

Net-cost arithmetic:

```text
net_magnitude = max(abs(target_ret_ticks_15m) - target_estimated_cost_ticks, 0)
target_net_ticks_after_est_cost = sign(target_ret_ticks_15m) * net_magnitude
```

`target_tradeable_after_cost` means the absolute forward move exceeds estimated
cost; it is not guaranteed profitability.

`target_valid` must be false if:

```text
current row is not causal_valid
entry row is missing
exit row is missing
entry or exit touches a synthetic row
path touches valid_ohlcv false
path touches boundary_session_flag true
path touches roll_window_flag true
entry/exit price invalid
future path crosses session_segment_id
future path crosses configured intraday cutoff
future path crosses suspicious roll boundary
time remaining is insufficient for entry and exit
```

---

### Standard commands

Causal base:

```bash
python -m scripts.phase2_causal_base.build_causal_base_data --profile tier_1
```

Labels:

```bash
python -m scripts.phase3_labels.build_labels --profile tier_1
```

Baseline + L0 regime features:

```bash
python -m scripts.phase4_features.build_baseline_features --profile tier_1
```

Tests:

```bash
python -m pytest -q
```

Git check:

```bash
git status --short
```

---

## Part 2 — Phase-by-Phase Build Requirements

# Phase 1 — DBN Archive and Raw Parquet Stitch

## Purpose

Archive immutable Databento OHLCV and point-in-time definition DBN/DBN.ZST
chunks, then validate and convert/stitch OHLCV chunks into immutable
continuous-contract 1-minute OHLCV parquet files.

## Input

Databento `GLBX.MDP3` DBN or DBN.ZST chunks, one OHLCV and one definition file
per market/year.

## Required paths

DBN archive:

```text
data/raw/{market}/{year}.dbn.zst
data/raw/definition/{market}/{year}.dbn.zst
```

Raw parquet output:

```text
data/raw/{market}/{year}.parquet
```

## Required raw schema

```text
ts_event
open
high
low
close
volume
rtype
publisher_id
instrument_id
symbol
data_quality_status
data_quality_degraded
```

Production and research profiles require every field above. The
`metadata_optional_test` schema variant is relaxed for test fixtures only.

## Build requirements

- Support one DBN chunk per market/year.
- Request and manifest both `schema="ohlcv-1m"` and `schema="definition"`.
- Hash every DBN chunk before conversion and verify the sidecar manifest.
- Convert all chunks, concatenate, sort by `ts_event`, and check duplicates.
- Do not mutate raw files.
- Do not overwrite DBN archives or raw parquet files during pipeline runs.
- Do not track raw files in git.
- Ensure market and year can be inferred from the path.
- Preserve `rtype`, `publisher_id`, `instrument_id`, and `symbol` for audit.
- Require definition coverage for every OHLCV `instrument_id`.
- Require `raw_symbol` mapping and positive tick-size metadata from definition rows.
- Do not rename `ts_event` to `ts` in Phase 1.
- Do not fill missing 1-minute bars in Phase 1.
- Fail if required schema or metadata is missing or fake-filled.
- Report `price_scale_policy`, `data_quality_source`,
  `vendor_quality_available`, `decoded_symbols`, `input_hashes`,
  `output_hash`, row counts, `first_ts`, and `last_ts`.

## Breakpoints

- Missing required columns in production or research profiles.
- Bad timestamp type.
- Duplicate `ts_event` rows.
- Missing or fake-filled Databento metadata.
- Missing data-quality source.
- Mixed or unexpected symbols without roll reporting.
- Calendar-year files that split Globex sessions.
- Continuous-contract adjustment artifacts treated as alpha.

## Acceptance checks

- File exists for each configured market/year.
- File is non-empty.
- Required columns exist; missing strict raw fields are FAIL, not WARN.
- Each DBN input chunk is hashed and recorded.
- Every OHLCV instrument has definition coverage.
- Output hash, row counts, first timestamp, and last timestamp are reported.
- Path market/year matches configured profile.

---

# Phase 2 — Causal Base Builder

## Purpose

Validate raw OHLCV data, normalize sessions, handle missing session minutes, mark synthetic rows, flag roll transitions, add session metadata, and produce the only approved modeling base table.

This phase replaces separate materialized stages for raw manifest, validation, validated data, session normalization, session-normalized data, causal gating, and causal output.

## Script

```bash
python -m scripts.phase2_causal_base.build_causal_base_data --profile tier_1
```

## Input

```text
data/raw/{market}/{year}.parquet
```

## Output

```text
data/causally_gated_normalized/{market}/{year}.parquet
```

## Reports

```text
reports/causal_base/causal_base_manifest.json
reports/causal_base/causal_base_validation.csv
reports/causal_base/causal_base_validation.json
```

## Required output columns

```text
ts
market
year
symbol
instrument_id
publisher_id
rtype
open
high
low
close
volume
valid_ohlcv
causal_invalid_reason
raw_row_present
is_synthetic
synthetic_gap_id
synthetic_gap_size_minutes
synthetic_gap_reason
data_quality_status
data_quality_degraded
session_data_quality_degraded
trainable_data_quality
causal_valid
session_id
session_date
session_segment_id
inside_session
boundary_session_flag
session_calendar_status
holiday_calendar_available
early_close_calendar_available
calendar_coverage_status
is_session_open
is_session_close
minutes_since_session_open
minutes_until_session_close
minute_of_day
day_of_week
roll_boundary_flag
bars_since_roll
bars_until_roll
roll_window_flag
metadata_available
roll_detection_available
roll_detection_source
roll_policy_status
source_path
source_file_hash
source_row_number
raw_schema_variant
raw_schema_policy
required_raw_schema_cols
raw_schema_missing_cols
missing_required_raw_cols
timestamp_source
```

## Required logic

```text
load raw parquet
rename ts_event -> ts
infer market/year from path
load previous-year tail and next-year head for boundary bleed
sort deterministically
validate schema
validate timestamp parseability
reject duplicate timestamps by default
validate OHLC consistency
validate non-negative integer-like volume
validate tick grid using market config
normalize to CME Globex sessions
insert missing 1-minute session bars where appropriate
mark synthetic rows
assign session_id
assign session_segment_id
compute session metadata
compute calendar coverage status
flag roll boundaries using symbol/instrument_id changes
flag roll exclusion windows
compute causal_valid
write current-year partition only
write validation reports
```

## Synthetic row policy

```text
raw_row_present = false
is_synthetic = true
causal_valid = false
volume = 0
open/high/low/close = previous known close within the same session_segment_id
```

Never forward-fill across `session_segment_id`.

## Causal validity policy

```text
causal_valid =
raw_row_present
and not is_synthetic
and valid_ohlcv
and inside_session
and trainable_data_quality
and not roll_window_flag
and not boundary_session_flag
```

## Calendar coverage policy

```text
calendar_coverage_status values:
regular_session_only
config_backed
hardcoded_regular_session
```

Empty `holidays`, `closed_dates`, or `early_closes` configuration is WARN, not
FAIL. `regular_session_only` is structurally usable but is not full
holiday/early-close proof.

## Raw schema policy

```text
production/research profiles require strict raw schema fields
metadata_optional_test is relaxed/test-only
missing strict raw fields are FAIL, not WARN
```

## Hard failures

```text
missing required columns in production/research profiles
unparseable ts_event
null ts_event
empty file
invalid OHLC structure
negative volume
unknown market config
duplicate timestamps unless explicitly handled
```

## Warnings

```text
missing session minutes
synthetic rows inserted
tick-grid anomalies
large gaps
stale close runs
zero volume bars
unexpected rtype values
multiple publisher_id values
symbol changes
instrument_id changes
roll boundary rows
roll exclusion rows
low session coverage
empty holidays/closed_dates/early_closes calendar config
regular_session_only calendar coverage
```

## Acceptance checks

- Output exists for each configured market/year.
- Output row count is non-zero.
- Required output columns exist.
- `ts` is sorted and unique per market/year.
- Synthetic rows are marked and not trainable.
- Session metadata is populated.
- Roll boundary and roll-window counts are reported.
- Reports include row counts, hash metadata, warning counts, and failure counts.

---

# Phase 3 — Target / Label Generation

## Purpose

Create forward-looking labels while preserving realistic intraday execution alignment and net-cost awareness.

## Script

```bash
python -m scripts.phase3_labels.build_labels --profile tier_1
```

## Input

```text
data/causally_gated_normalized/{market}/{year}.parquet
```

## Output

```text
data/labeled/{market}/{year}.parquet
reports/labels/label_manifest.json
reports/labels/label_report.json
```

## Target policy

```text
execution style = intraday only
prediction time = after completed bar t
default entry = next bar t+1
default horizon = 15 minutes after entry
no overnight labels
cost-aware target fields required
```

## Required target columns

```text
target_entry_ts
target_exit_ts
target_entry_price
target_exit_price
target_invalid_reason
target_ret_15m
target_ret_ticks_15m
target_gross_dollars_15m
target_estimated_cost_ticks
target_estimated_cost_dollars
target_net_ticks_after_est_cost
target_net_dollars_after_est_cost
target_sign_15m
target_sign_with_deadzone
target_tradeable_after_cost
target_valid
target_horizon_bars
mae_ticks_15m
mfe_ticks_15m
fade_long_success_15m
fade_short_success_15m
trend_danger_up_30m
trend_danger_down_30m
revert_to_vwap_30m
revert_to_session_mid_30m
label_semantics
cost_source
cost_provisional
```

## Downstream target groups

```yaml
return_target:
  - target_ret_15m
  - target_ret_ticks_15m
  - target_net_ticks_after_est_cost
  - target_net_dollars_after_est_cost

direction_target:
  - target_sign_15m
  - target_sign_with_deadzone
  - target_tradeable_after_cost

fade_success_target:
  - target_fade_long_success_15m
  - target_fade_short_success_15m
  - target_fade_success_15m

trend_danger_target:
  - target_trend_danger_long_30m
  - target_trend_danger_short_30m
  - target_trend_danger_30m
```

Fade-success and trend-danger labels are targets only. They may be built using
future information inside target construction, must never be used as features,
and must be excluded from feature matrices.

## Required validity logic

`target_valid` must be false if:

```text
current row causal_valid is false
entry row is unavailable
exit row is unavailable
future path touches synthetic rows
future path touches `valid_ohlcv == false`
future path touches `boundary_session_flag == true`
future path crosses session_segment_id
future path crosses roll_boundary_flag or roll_window_flag
future path crosses intraday flatten cutoff
entry/exit price is invalid
minutes_until_session_close is insufficient
```

## Cost-aware target logic

- Directional labels alone are not enough.
- A tiny positive move that cannot beat costs should not be treated as a tradeable long opportunity.
- `target_sign_with_deadzone` should be neutral when absolute target move is below estimated round-trip cost plus configured buffer.
- Corrected net cost arithmetic:

```text
net_magnitude = max(abs(target_ret_ticks_15m) - target_estimated_cost_ticks, 0)
target_net_ticks_after_est_cost = sign(target_ret_ticks_15m) * net_magnitude
```

- `target_tradeable_after_cost` means the absolute forward move exceeds estimated cost, not guaranteed profitability.

## Build requirements

- Labels may use future prices.
- Features may never use future prices.
- Label and execution conventions must match.
- Label invalid counts must be reported by reason.
- Target values must be available in returns, ticks, and dollars.

## Breakpoints

- Using `close[t]` as both feature and executable entry price.
- Allowing labels to cross session boundaries.
- Allowing labels to cross synthetic rows.
- Allowing labels to cross roll transitions.
- Creating target rows late in the session that cannot be exited intraday.
- Treating sub-cost moves as tradeable alpha.

## Acceptance checks

- `target_valid` exists.
- Cost-aware target columns exist.
- Invalid target counts are reported by reason.
- No valid target crosses session segment, synthetic rows, cutoff, or roll boundary.
- Target horizon equals configured horizon.

---

# Phase 4 — Baseline + L0 Regime Feature Matrix

## Purpose

Create the initial OHLCV-only modeling matrix with baseline features, L0 regime context, metadata columns, target columns, and validity columns.

## Script

```bash
python -m scripts.phase4_features.build_baseline_features --profile tier_1
```

## Planned paths

```text
scripts/phase4_features/build_baseline_features.py
tests/phase4_features/test_build_baseline_features.py
```

## Input

```text
data/labeled/{market}/{year}.parquet
```

## Output

```text
data/feature_matrices/baseline/{market}/{year}.parquet
reports/features_baseline/baseline_feature_manifest.json
reports/features_baseline/baseline_feature_report.json
reports/features_baseline/feature_registry.json
reports/features_baseline/feature_correlation_report.csv
data/feature_matrices/baseline/feature_cols.json
data/feature_matrices/baseline/target_cols.json
data/feature_matrices/baseline/metadata_cols.json
data/feature_matrices/baseline/excluded_cols.json
```

Phase 4 writes the initial `feature_cols`, `target_cols`, `metadata_cols`, and
`excluded_cols` registries. Phase 5 audits, freezes, and promotes them for WFA.

## Feature goal

The first useful goal is not predicting every 15-minute direction. The first useful goal is identifying when fading is unsafe.

## Feature families

```text
trend-danger / path shape
failed breakout / rejection
range / chop state
session structure
volatility regime
volume behavior
higher-timeframe context from 1m bars
time-of-day regime
intermarket L0 context
```

## Representative features

```text
feature_efficiency_ratio_15
feature_efficiency_ratio_30
feature_efficiency_ratio_60
feature_failed_breakout_above_20
feature_failed_breakout_below_20
feature_session_high_dist
feature_session_low_dist
feature_session_mid_dist
feature_opening_range_30_high_dist
feature_opening_range_30_low_dist
feature_vol_expansion_15_vs_60
feature_large_bar_count_30
feature_bars_since_shock
feature_volume_surge_with_range
feature_volume_surge_without_progress
feature_prior_session_high_dist
feature_prior_session_low_dist
feature_prior_session_close_dist
feature_overnight_gap_ticks
feature_rel_ret_vs_ES_15
feature_corr_vs_ES_60
feature_es_zn_divergence_30
feature_cl_es_divergence_30
```

## Required logic

- Rolling features are grouped by `session_segment_id`.
- Rolling, lag, and count features must not compute through
  `feature_input_valid=false` rows.
- If any required lookback row is invalid, the derived feature value must be
  `NaN`, not `false`, `0`, or a forward-filled value.
- Session cumulative features reset by `session_segment_id` or valid session boundary.
- Opening range features are unavailable until the opening range window is complete.
- Volume relative-to-time-of-day features must use train-only references inside WFA or be implemented as fold-aware features.
- Synthetic rows are not trainable.
- Do not compute returns through synthetic rows.
- Missingness and low-liquidity behavior may be represented as features, but synthetic prices must not create false volatility compression.

## Breakpoints

- Rolling windows cross sessions.
- Opening range values leak before the opening range completes.
- Time-of-day volume normalization uses full-sample averages.
- Databento metadata becomes model features.
- Synthetic rows contaminate volatility, return, or VWAP features.

## Acceptance checks

- Output matrix exists.
- Feature columns are numeric and finite or intentionally nullable with later train-only imputation.
- Required metadata and target columns are preserved.
- No known forbidden metadata columns are included as feature columns.

---

# Phase 5 — Feature / Target / Metadata Column Registry

## Purpose

Audit, freeze, and promote the initial Phase 4 column registry for WFA so
modeling cannot consume obvious leakage columns.

## Script

```bash
python -m scripts.build_column_registry --profile tier_1 --matrix baseline
```

## Output

```text
data/feature_matrices/baseline/feature_cols.json
data/feature_matrices/baseline/target_cols.json
data/feature_matrices/baseline/metadata_cols.json
data/feature_matrices/baseline/excluded_cols.json
```

## Forbidden feature columns

```text
rtype
publisher_id
instrument_id
symbol
ts
market
year
session_id
session_date
session_segment_id
raw_row_present
is_synthetic
causal_valid
roll_boundary_flag
bars_since_roll
bars_until_roll
roll_window_flag
valid_ohlcv
causal_invalid_reason
data_quality_status
data_quality_degraded
session_data_quality_degraded
trainable_data_quality
inside_session
boundary_session_flag
session_calendar_status
calendar_coverage_status
metadata_available
raw_schema_variant
raw_schema_policy
target_entry_ts
target_exit_ts
target_entry_price
target_exit_price
target_invalid_reason
target_ret_15m
target_ret_ticks_15m
target_gross_dollars_15m
target_estimated_cost_ticks
target_estimated_cost_dollars
target_net_ticks_after_est_cost
target_net_dollars_after_est_cost
target_sign_15m
target_sign_with_deadzone
target_tradeable_after_cost
target_valid
target_horizon_bars
mae_ticks_15m
mfe_ticks_15m
fade_long_success_15m
fade_short_success_15m
trend_danger_up_30m
trend_danger_down_30m
revert_to_vwap_30m
revert_to_session_mid_30m
label_semantics
cost_source
cost_provisional
```

## Build requirements

- Start from the initial registry written by Phase 4.
- Audit and freeze `feature_cols`, `target_cols`, `metadata_cols`, and
  `excluded_cols` before WFA.
- Use an allowlist-first feature registry where possible.
- Fail hard if any target column appears in `feature_cols.json`.
- Fail hard if any Databento metadata identifier appears in `feature_cols.json`.
- Fail hard if timestamp, session ID, raw validity, or target validity columns appear in features.
- Report total feature count.
- Add semantic leakage tests for features whose computation could use future data.

## Acceptance checks

- All registry JSON files exist.
- Feature columns exist in the matrix.
- Forbidden columns are absent from feature columns.
- Target columns are present in target registry.

---

# Phase 6 — WFA Split Plan

## Purpose

Create deterministic walk-forward train/test folds with research/final-holdout separation.

## Script

```bash
python -m scripts.build_wfa_splits --profile tier_1
```

## Output

```text
reports/wfa/split_plan.csv
reports/wfa/split_plan.json
```

## Default research policy

```text
recent_research = 2023-2025
long_research = 2010-2025
forward_years = 2026
train_days = 365
test_days = 30
step_days = 30
purge_bars = auto
resolved_purge_bars = entry_lag_bars + target_horizon_bars
default resolved_purge_bars = 16
```

## Required fold fields

```text
market
fold_id
split_group
train_start
train_end
purged_train_end
test_start
test_end
train_rows_before_purge
train_rows_after_purge
test_rows
purge_bars
resolved_purge_bars
is_final_holdout
```

## Build requirements

- Splits are by timestamp, not random.
- Splits are per market unless explicitly configured otherwise.
- No train/test overlap.
- Purge removes train rows whose labels could overlap the test start.
- Empty folds are rejected.
- Research splits and final-holdout splits are explicitly tagged.

## Acceptance checks

- Split plan exists.
- Each fold has positive train and test rows.
- `purged_train_end < test_start`.
- Fold count is reported by market.
- Final-holdout test windows are not used for feature or policy selection.

---

# Phase 7A-7C — Staged Baseline WFA Train / Test

## Purpose

Train staged baseline and challenger models on each train fold and generate
out-of-sample predictions for each test fold.

## Script

```bash
python -m scripts.run_wfa --profile tier_1 --matrix baseline --run baseline
```

## Input

```text
data/feature_matrices/baseline/
data/feature_matrices/baseline/feature_cols.json
reports/wfa/split_plan.json
```

## Output

```text
reports/wfa/baseline_wfa_report.json
```

## Model policy

```yaml
Phase 7A linear controls:
  ridge_return_v1:
    model_family: ridge_regression
    task: regression
    target: target_ret_15m

  logistic_direction_v1:
    model_family: logistic_regression
    task: classification
    target: target_sign_with_deadzone

  logistic_fade_success_v1:
    model_family: logistic_regression
    task: classification
    target: target_fade_success_15m

  logistic_trend_danger_v1:
    model_family: logistic_regression
    task: classification
    target: target_trend_danger_30m

Phase 7B nonlinear challengers:
  histgb_direction_v1:
    model_family: hist_gradient_boosting
    task: classification
    target: target_sign_with_deadzone

  histgb_fade_success_v1:
    model_family: hist_gradient_boosting
    task: classification
    target: target_fade_success_15m

  histgb_trend_danger_v1:
    model_family: hist_gradient_boosting
    task: classification
    target: target_trend_danger_30m

Phase 7C optional serious nonlinear challengers:
  lightgbm_*:
    model_family: lightgbm
    enabled_by_default: false
    cpu_first: true

  xgboost_*:
    model_family: xgboost
    enabled_by_default: false
    cpu_first: true
```

Ridge/logistic are controls. HistGradientBoosting is the first nonlinear
challenger. LightGBM/XGBoost are optional downstream challengers. Optional
external dependencies must not be required for baseline tests to pass.

Common rules:

```text
imputer = train-only
scaler = train-only where applicable
fit = train fold only
predict = test fold only
random train/test split = forbidden
final-holdout tuning = forbidden
hyperparameter tuning = disabled initially
model_id = required in prediction and metric reports
model_config_hash = required in prediction and metric reports
feature_config_hash = required in prediction and metric reports
```

## Build requirements

- Filter training rows to `causal_valid == true` and `target_valid == true`.
- Fit imputer only on train fold.
- Fit scaler only on train fold.
- Fit each model only on train fold.
- Predict only test rows.
- Record `model_id`, `model_family`, target name, model config hash, and feature
  config hash for every prediction and metric row.
- Save fold-level diagnostics.
- Report prediction distribution and target distribution by fold.

## Breakpoints

- Imputer or scaler fit on full matrix.
- Model fit on test rows.
- Fold leakage through feature selection or preprocessing.
- Training on invalid synthetic or invalid target rows.
- Model collapses to near-constant class-prior predictions.

## Acceptance checks

- Fold report exists.
- Every fold reports train/test rows.
- No skipped rows are unexplained.
- No fitting object uses test data.

---

# Phase 8 — Baseline OOS Predictions

## Purpose

Write out-of-sample predictions with enough model, target, price, and metadata
columns for execution without unsafe joins.

## Output

```text
data/predictions/baseline/oos_predictions.parquet
reports/wfa/baseline_predictions_manifest.json
```

## Required columns

```text
market
year
fold_id
timestamp
session_id
session_segment_id
split_group
model_id
model_family
target_name
prediction_type
y_true
y_pred_raw
y_pred_calibrated
p_long
p_short
p_flat
p_fade_success
p_trend_danger
calibration_id
model_config_hash
feature_config_hash
execution_open
execution_close
target_valid
causal_valid
close
target_entry_ts
target_exit_ts
minutes_until_session_close
```

`session_id` should be present when available; `session_segment_id` is required
when session IDs are not available at prediction time.

## Build requirements

- Predictions are unique by `market`, `timestamp`, `fold_id`, `model_id`, and
  `target_name`.
- Predictions are OOS only.
- Execution price and exit price are included or derivable without rejoining raw data.
- Invalid target rows are excluded from model scoring unless explicitly reported as skipped.
- Regression models may leave probability columns null.
- Classification models should populate relevant probability columns when
  available.
- Raw predictions and calibrated predictions must be preserved separately.
- Position policy must consume calibrated or model-score fields, not blindly
  trade raw predictions.

## Acceptance checks

- OOS prediction parquet exists.
- No duplicate prediction rows.
- Prediction rows are test-fold rows only.
- Required execution columns exist.

---

# Phase 8A — Signal Calibration and Model Comparison

## Purpose

Fit train-only calibration where configured, compare model candidates, and
select research-period candidates without using final holdout.

## Output

```text
reports/model_selection/model_comparison.csv
reports/model_selection/model_selection_report.json
reports/model_selection/calibration_report.json
```

## Calibration rules

- Calibration is fit on train fold or a train-internal calibration split only.
- Calibration cannot be fit on the test fold.
- Calibration cannot use final holdout.
- Calibration outputs must have `calibration_id`.
- Raw model score and calibrated score must both be preserved.
- Calibration can be skipped for a model, but the skip must be explicit in
  reports.

Allowed calibration approaches:

```text
none
logistic/Platt style
isotonic only if enough data and train-only fitting is enforced
```

## Model comparison grouping

Reports must be grouped by:

```text
model_id
model_family
target_name
market
fold
train/test window
config hash
```

## Model comparison metrics

Include where available:

```text
gross return
net return
gross Sharpe
net Sharpe
max drawdown
turnover/bar
trade count
cost drag
per-market metrics
per-fold stability
trend-day behavior
fade-allowed vs fade-blocked behavior
final-holdout excluded from selection
```

## Acceptance checks

- Calibration report exists or each model has an explicit no-calibration marker.
- Model comparison excludes final-holdout rows from selection.
- `calibration_id`, `model_config_hash`, and `feature_config_hash` are reported.
- Frozen model selection occurs before final holdout.

---

# Phase 9 — Baseline Execution + Cost Model

## Purpose

Convert calibrated model scores into positions, compare deterministic position
policies, charge realistic market-specific costs, and compute net returns in
returns, ticks, and dollars.

## Script

```bash
python -m scripts.run_execution_costs --profile tier_1 --run baseline
```

## Input

```text
data/predictions/baseline/oos_predictions.parquet
configs/costs.yaml
configs/markets/{market}.yaml
```

## Output

```text
data/executions/baseline/executions.parquet
reports/execution/baseline_cost_report.json
```

## Required execution logic

```text
expected_return
p_long
p_short
p_flat
p_fade_success
p_trend_danger
-> deterministic signal threshold / no-trade band
-> optional hysteresis
-> optional minimum hold rule
-> intraday flatten rule
-> position state machine
-> position_change
-> slippage cost
-> commission cost
-> gross_return
-> gross_ticks
-> gross_dollars
-> net_return
-> net_ticks
-> net_dollars
```

`p_trend_danger` can block fade trades. `p_fade_success` can allow or disallow
fade trades. Raw return prediction alone should not directly become trades. A
deterministic policy converts model scores into flat/long/short/size/add/no-add
decisions, and all policy choices must be replayable from saved OOS predictions
and config.

## Cost rule

```text
flat -> long  = position_change 1.0
flat -> short = position_change 1.0
long -> flat  = position_change 1.0
short -> flat = position_change 1.0
long -> short = position_change 2.0
short -> long = position_change 2.0
```

## Required policy comparisons

At minimum, compare deterministic policies:

```text
raw_sign
threshold_small
threshold_medium
no_trade_band
hysteresis
min_hold_5m
```

For baseline diagnostics, report all policies. Do not select a final policy using final-holdout information.

## Required cost scenarios

```text
cost_1x
cost_2x
cost_3x
```

## Required output columns

```text
market
timestamp
fold_id
split_group
model_id
model_family
target_name
policy_name
cost_scenario
y_pred_raw
y_pred_calibrated
p_long
p_short
p_flat
p_fade_success
p_trend_danger
signal
position
prev_position
position_change
gross_return
gross_ticks
gross_dollars
slippage_cost_ticks
slippage_cost_dollars
commission_cost_dollars
total_cost_dollars
net_return
net_ticks
net_dollars
forced_flat_flag
```

## Acceptance checks

- Costs are non-negative.
- Long-short flips cost twice a flat-to-position transition.
- No position is held after the configured intraday cutoff.
- Turnover and cost drag are reported by policy and cost scenario.
- Dollar PnL reconciles to tick value and position size.

---

# Phase 10 — Baseline Metrics + Diagnostics

## Purpose

Evaluate baseline predictions, execution behavior, costs, cost stress, and net economics.

## Script

```bash
python -m scripts.build_metrics --profile tier_1 --run baseline
```

## Output

```text
reports/metrics/baseline_metrics.json
reports/metrics/baseline_metrics.csv
reports/metrics/cost_breakdown.csv
reports/metrics/turnover_diagnostics.csv
reports/metrics/signal_diagnostics.csv
reports/metrics/prediction_bucket_diagnostics.csv
reports/metrics/cost_stress_baseline.csv
```

## Required metrics

```text
gross_return
net_return
gross_ticks
net_ticks
gross_dollars
net_dollars
gross_sharpe
net_sharpe
max_drawdown
trade_events
turnover_per_bar
cost_drag
slippage_total
commission_total
prediction_distribution
position_distribution
long_return
short_return
flat_percentage
```

## Required breakdowns

```text
overall
by market
by year
by fold
by market-year
by policy_name
by cost_scenario
```

## Required prediction diagnostics

- Bin `y_pred` into deciles or quantiles.
- Report realized gross and net return by prediction bucket.
- Identify whether the model has monotonic predictive value or just noise around zero.

## Acceptance checks

- Metrics exist for every execution policy and cost scenario.
- Cost breakdown reconciles to total cost.
- Aggregate metrics do not hide per-market failures.
- Prediction bucket report exists.

---

# Phase 11 — Baseline Accept / Reject Gate

## Purpose

Decide whether the baseline pipeline is valid and whether the baseline strategy is economically viable.

## Script

```bash
python -m scripts.run_gate --profile tier_1 --run baseline
```

## Output

```text
reports/gates/baseline_gate.json
```

## Gate categories

```text
pipeline_integrity
causal_integrity
wfa_integrity
prediction_integrity
cost_integrity
turnover_control
economic_viability
```

## Gate logic

The gate may pass pipeline integrity and fail economic viability.

Fail or warn if:

```text
net_return < 0
net_sharpe <= 0
cost_drag > abs(gross_return)
turnover_per_bar above configured threshold
trade_events excessive
long-short flips dominate turnover
performance exists only under cost_1x but fails cost_2x or cost_3x
```

## Acceptance checks

- Gate JSON exists.
- Gate reports pass/warn/fail by category.
- Reasons are explicit.
- Negative net economics are not hidden behind an overall pass.

---

# Phase 12 — Feature Expansion

## Purpose

Generate a broader OHLCV-only candidate matrix for feature discovery and selection.

## Script

```bash
python -m scripts.build_expanded_features --profile tier_1
```

## Input

```text
data/labeled/{market}/{year}.parquet
```

## Output

```text
data/feature_matrices/expanded/{market}/{year}.parquet
reports/features/expanded_feature_report.json
```

## Candidate families

```text
additional return horizons
additional realized volatility windows
additional range and candle-shape features
additional volume participation features
rolling VWAP variants
session VWAP variants
range-position variants
EMA distance and slope variants
opening-range variants
session-state variants
compression and expansion flags
liquidity and friction proxies
```

## Build requirements

- OHLCV-only.
- Causal.
- Session-aware.
- Hypothesis-tagged where possible.
- No alternative data.
- No order book data.
- No metadata identifiers as features.
- Do not expand features to fix a broken cost model or broken target.

## Acceptance checks

- Expanded matrix exists.
- Expanded feature report includes added, rejected, and invalid features.
- Rolling features do not cross session boundaries.

---

# Phase 13 — Feature Discovery

## Purpose

Analyze feature behavior, redundancy, stability, leakage risk, and economic usefulness before selection.

## Script

```bash
python -m scripts.run_feature_discovery --profile tier_1
```

## Output

```text
reports/feature_discovery/feature_summary.csv
reports/feature_discovery/correlation_report.csv
reports/feature_discovery/correlation_clusters.csv
reports/feature_discovery/leakage_report.json
reports/feature_discovery/stability_report.csv
reports/feature_discovery/incremental_net_impact.csv
```

## Required checks

```text
null rate
infinite value count
near-zero variance
Pearson correlation
Spearman correlation
correlation clusters at abs(corr) >= 0.70
market stability
fold stability
forbidden-column leakage
target overlap leakage
incremental train-fold net impact
```

## Build requirements

- Discovery that informs selection must use research-period train folds only.
- Full-sample discovery may be allowed only for descriptive diagnostics and must not drive final selection.
- Correlation clustering should identify redundant feature groups.
- Feature usefulness should be measured economically where possible, not only statistically.

## Acceptance checks

- Reports exist.
- Leakage report has zero forbidden feature overlap.
- Highly correlated clusters are reported.
- Incremental net impact report exists or is explicitly marked unsupported.

---

# Phase 14 — Train-Only Feature Ranking / Selection

## Purpose

Select features without using OOS test information or final-holdout information.

## Script

```bash
python -m scripts.run_feature_selection --profile tier_1
```

## Output

```text
reports/feature_selection/feature_ranking.csv
reports/feature_selection/selected_features.csv
reports/feature_selection/rejected_features.csv
reports/feature_selection/selection_manifest.json
```

## Selection policy

```text
filter invalid features
filter high-null features
filter near-zero variance features
cluster abs(Pearson or Spearman correlation) >= 0.70
keep one representative per cluster
prefer features stable across markets and folds
prefer features with positive train-fold economic contribution
rank using research-period train-fold-only information
```

## Build requirements

- No final-holdout target statistics.
- No final OOS metrics used for selection.
- Rejected features must include rejection reason.
- Selected features must be reproducible.
- Every selection run must write a manifest to limit human-in-the-loop overfitting.

## Acceptance checks

- Selection report exists.
- Selected features are all present in expanded matrix.
- No forbidden columns selected.
- Selection manifest includes config hash and input hash.

---

# Phase 15 — Frozen Feature + Model + Calibration + Policy Set

## Purpose

Freeze selected features, selected model config, calibration config, and the
selected execution policy before final evaluation.

## Script

```bash
python -m scripts.freeze_features --profile tier_1
python -m scripts.freeze_model --profile tier_1
python -m scripts.freeze_calibration --profile tier_1
python -m scripts.freeze_policy --profile tier_1
```

## Output

```text
data/frozen_features/phase5_v1/feature_cols.json
data/frozen_features/phase5_v1/selected_features.csv
data/frozen_features/phase5_v1/rejected_features.csv
data/frozen_features/phase5_v1/policy_config.json
data/frozen_features/phase5_v1/manifest.json

data/frozen_models/phase5_v1/model_config.yaml
data/frozen_models/phase5_v1/model_selection_report.json
data/frozen_models/phase5_v1/calibration_config.yaml
data/frozen_models/phase5_v1/manifest.json
```

## Build requirements

- Frozen feature list is immutable during final WFA.
- Frozen model config is immutable during final WFA.
- Frozen calibration config is immutable during final WFA.
- Frozen policy is immutable during final execution.
- Manifests include feature-selection, model-selection, calibration, policy, and
  config hashes.
- Final WFA must consume only frozen features, frozen model config, and frozen
  calibration config.
- Final execution must consume only frozen policy unless running diagnostic comparison that is not used for the gate.
- Final holdout cannot choose a model.
- Final holdout cannot tune thresholds.
- Final holdout cannot change calibration.
- Final holdout cannot change features.

## Acceptance checks

- Frozen feature files exist.
- Frozen model files exist.
- Frozen calibration files exist.
- Frozen policy file exists.
- Feature count is reported.
- Frozen features match selected features.
- Frozen model, calibration, and policy were selected without final-holdout data.
- Frozen artifacts include config hashes.

---

# Phase 16 — Final Holdout Split Plan

## Purpose

Create the final evaluation split plan while ensuring final test windows are entirely inside the untouched holdout period.

## Script

```bash
python -m scripts.build_final_splits --profile tier_1
```

## Output

```text
reports/final_wfa/final_split_plan.csv
reports/final_wfa/final_split_plan.json
```

## Build requirements

- Test windows must be inside `final_holdout_years`.
- Training windows may use only data available before the test window.
- Feature list, model config, calibration config, and policy must already be frozen.
- The split report must clearly mark all final-holdout folds.

## Acceptance checks

- Final split plan exists.
- No final test fold begins before the configured holdout start.
- No feature or policy selection stage consumes final test rows.

---

# Phase 17 — Final WFA With Frozen Features

## Purpose

Evaluate the frozen feature set through the same causal WFA process on final holdout folds.

## Script

```bash
python -m scripts.run_wfa --profile tier_1 --matrix expanded --features data/frozen_features/phase5_v1/feature_cols.json --model-config data/frozen_models/phase5_v1/model_config.yaml --calibration-config data/frozen_models/phase5_v1/calibration_config.yaml --split-plan reports/final_wfa/final_split_plan.json --run final
```

## Input

```text
data/feature_matrices/expanded/
data/frozen_features/phase5_v1/feature_cols.json
data/frozen_models/phase5_v1/model_config.yaml
data/frozen_models/phase5_v1/calibration_config.yaml
reports/final_wfa/final_split_plan.json
```

## Output

```text
reports/final_wfa/final_wfa_report.json
```

## Build requirements

- Final split plan only.
- Same purge rules.
- Train-only imputer.
- Train-only scaler.
- Frozen features only.
- Frozen model config only.
- Frozen calibration config only.
- Test-fold-only predictions.
- No feature, model, calibration, or policy changes after seeing final results.

## Acceptance checks

- Final WFA report exists.
- Frozen feature list was used exactly.
- Fold diagnostics are complete.
- Final folds are holdout-tagged.

---

# Phase 18 — Final OOS Predictions

## Purpose

Write final out-of-sample predictions with the same multi-model schema and
execution-ready columns as baseline OOS predictions.

## Output

```text
data/predictions/final/oos_predictions.parquet
reports/final_wfa/final_predictions_manifest.json
```

## Required columns

```text
market
year
fold_id
timestamp
session_id
session_segment_id
split_group
model_id
model_family
target_name
prediction_type
y_true
y_pred_raw
y_pred_calibrated
p_long
p_short
p_flat
p_fade_success
p_trend_danger
calibration_id
model_config_hash
feature_config_hash
execution_open
execution_close
target_valid
causal_valid
close
target_entry_ts
target_exit_ts
minutes_until_session_close
```

## Acceptance checks

- Predictions are OOS only.
- Predictions are final-holdout test rows only.
- No duplicate prediction rows.
- Execution-ready prices exist.
- Frozen model config hash and frozen feature config hash are present.
- Calibration is frozen or explicitly marked as skipped.

---

# Phase 19 — Final Execution + Cost Model

## Purpose

Apply the frozen execution and cost policy to final predictions.

## Script

```bash
python -m scripts.run_execution_costs --profile tier_1 --run final --policy data/frozen_features/phase5_v1/policy_config.json
```

## Input

```text
data/predictions/final/oos_predictions.parquet
data/frozen_features/phase5_v1/policy_config.json
data/frozen_models/phase5_v1/model_config.yaml
data/frozen_models/phase5_v1/calibration_config.yaml
configs/costs.yaml
```

## Output

```text
data/executions/final/executions.parquet
reports/final_execution/final_cost_report.json
```

## Build requirements

- Same cost model as baseline unless versioned.
- Frozen execution policy only for gate metrics.
- Frozen model and calibration outputs only.
- Same intraday flatten rule.
- No overnight positions.
- Report cost stress scenarios, but do not choose a new policy from final stress results.

## Acceptance checks

- Final execution artifact exists.
- Costs reconcile.
- No positions remain open after cutoff.
- Frozen model, calibration, and policy were used for final gate metrics.

---

# Phase 20 — Final Metrics + Diagnostics

## Purpose

Evaluate final model economics and compare against baseline, placebo, simple-rule baselines, and cost stress scenarios.

## Script

```bash
python -m scripts.build_metrics --profile tier_1 --run final
python -m scripts.run_placebo_baselines --profile tier_1 --run final
```

## Output

```text
reports/final_metrics/final_metrics.json
reports/final_metrics/final_metrics.csv
reports/final_metrics/cost_breakdown.csv
reports/final_metrics/turnover_diagnostics.csv
reports/final_metrics/signal_diagnostics.csv
reports/final_metrics/baseline_vs_final.csv
reports/final_metrics/placebo_baselines.csv
reports/final_metrics/cost_stress_final.csv
```

## Required comparisons

```text
baseline vs final gross_return
baseline vs final net_return
baseline vs final gross_sharpe
baseline vs final net_sharpe
baseline vs final turnover_per_bar
baseline vs final trade_events
baseline vs final cost_drag
baseline vs final policy results
final vs flat/no-trade baseline
final vs seeded random-signal baseline
final vs lagged-signal placebo
final vs shuffled-target placebo
final vs simple VWAP rule
final vs simple opening-range rule
cost_1x vs cost_2x vs cost_3x
```

## Required breakdowns

```text
overall
by market
by year
by fold
by market-year
by policy_name
by cost_scenario
```

## Acceptance checks

- Final metrics exist.
- Baseline-vs-final comparison exists.
- Placebo baseline comparison exists.
- Cost stress report exists.
- Costs and net returns reconcile.

---

# Phase 21 — Prop-Firm Account Simulation

## Purpose

Evaluate whether the strategy is usable for the user's real constraint set: intraday futures trading with prop-firm-style drawdown and payout rules.

## Script

```bash
python -m scripts.run_prop_simulation --profile tier_1 --run final
```

## Input

```text
data/executions/final/executions.parquet
configs/prop_rules.yaml
```

## Output

```text
reports/prop_sim/final_prop_simulation.json
reports/prop_sim/daily_pnl.csv
reports/prop_sim/rule_violations.csv
reports/prop_sim/drawdown_path.csv
reports/prop_sim/payout_scenarios.csv
```

## Required simulated constraints

```text
starting_balance
profit_target
max_daily_loss
trailing_drawdown
max_contracts
minimum_trading_days
forced_flatten_time
consistency_rule_if_applicable
```

## Required metrics

```text
daily_pnl
cumulative_pnl
daily_max_drawdown
worst_intraday_drawdown
max_trailing_drawdown
rule_violation_count
days_to_profit_target
drawdown_before_target_flag
payout_target_hit_flag
probability_drawdown_before_target_under_scenarios
```

## Build requirements

- Simulate fixed position size before scaling.
- Add configurable contract sizing only after fixed-size viability is known.
- Forced flat must match the execution model.
- Any daily loss or trailing drawdown violation must be visible in the strategy gate.

## Acceptance checks

- Prop simulation outputs exist.
- Daily PnL reconciles to execution-level dollar PnL.
- Rule violations are explicit.
- Strategy gate can consume prop simulation result.

---

# Phase 22 — Strategy Accept / Reject Gate

## Purpose

Make the final strategy decision.

## Script

```bash
python -m scripts.run_gate --profile tier_1 --run final
```

## Output

```text
reports/gates/strategy_gate.json
```

## Gate categories

```text
pipeline_integrity
causal_integrity
wfa_integrity
holdout_integrity
feature_integrity
policy_integrity
prediction_integrity
cost_integrity
turnover_control
economic_viability
cost_stress_viability
baseline_comparison
placebo_comparison
prop_firm_viability
```

## Accept criteria

```text
pipeline_integrity = pass
causal_integrity = pass
wfa_integrity = pass
holdout_integrity = pass
feature_integrity = pass
policy_integrity = pass
prediction_integrity = pass
cost_integrity = pass
turnover_control = pass
economic_viability = pass
cost_stress_viability = pass
prop_firm_viability = pass
```

## Reject or warn if

```text
net_return < 0
net_sharpe <= 0
cost_drag > abs(gross_return)
turnover_per_bar above threshold
trade_events excessive
long-short flipping dominates
performance concentrated in one market or fold
final model does not improve baseline after costs
final model fails cost_2x or cost_3x stress
final model fails placebo/simple-rule comparison
prop simulation hits daily loss or trailing drawdown
payout target is not reached before drawdown in realistic scenarios
```

## Acceptance checks

- Strategy gate exists.
- Gate gives explicit pass/warn/fail reasons.
- Gate does not accept a negative-net strategy.
- Gate does not accept a final-holdout-contaminated strategy.
- Gate does not accept a prop-rule-violating strategy for the user's stated use case.
- Gate separates pipeline validity from economic viability.

---

## Artifact Provenance Requirement

Every phase report must include:

```text
generated_at
git_commit
script_path
script_hash
config_hash
input_file_hashes
output_file_hashes
profile
markets
years
warning_count
failure_count
failures
```

---

## Known Limitations

- Continuous contracts are research series, not directly live-tradable contracts.
- Calendar coverage is config-backed through 2026, but exchange schedules must be refreshed before live use.
- L0 OHLCV cannot model queue position, spread, order-book imbalance, or true fill probability.
- Provisional costs are structural only, not economic evidence.
- ZN synthetic-density warning must remain visible.
- Prop-firm viability requires dollar path simulation, not just Sharpe/net return.

---

## Required Tests

Add or maintain tests for:

```text
ts_event -> ts conversion
raw schema validation
duplicate timestamp rejection
OHLC validation
volume validation
tick-grid validation
session boundary assignment
year-boundary bleed
synthetic row handling
no forward-fill across session boundaries
roll boundary flagging
roll-window target invalidation
target does not cross session_segment_id
target does not use synthetic future rows
target does not cross roll boundary
target respects intraday cutoff
target entry/exit alignment uses t+1 to t+16 convention
cost-aware target deadzone
feature rolling windows grouped by session_segment_id
opening range does not leak before completion
volume relative-to-time-of-day does not use final-holdout full-sample averages
forbidden columns excluded from feature_cols
WFA purge enforcement
model registry validation
purge auto-resolution
target group exclusion from features
multi-model prediction schema
calibration train-only discipline
model selection excludes final holdout
frozen model immutability
project layout downstream ML consistency
research/final-holdout separation
train-only imputer/scaler
train-only feature selection
train-only/frozen policy selection
OOS prediction uniqueness
execution price alignment
position_change cost math
long-to-short flip cost = 2.0
market-specific tick and dollar PnL conversion
forced flat before cutoff
cost stress scenario generation
placebo baseline generation
prop-firm daily loss violation detection
prop-firm trailing drawdown violation detection
gate fails negative net strategy
gate fails final-holdout-contaminated strategy
gate fails prop-rule-violating strategy
```

---

## Implementation Priority

Build and validate in this order:

```text
1. Causal base builder reports and required columns
2. Label execution alignment and cost-aware target_valid rules
3. Baseline feature matrix with OHLCV feature families
4. Column registry hard gate
5. WFA split plan with research/final-holdout separation
6. Train-only preprocessing tests
7. OOS predictions with execution-ready columns
8. Execution cost model with tick/dollar PnL and policy comparison
9. Metrics and gates that separate pipeline validity from economic viability
10. Cost stress and prediction bucket diagnostics
11. Expanded features and correlation clustering
12. Train-only feature selection and frozen feature set
13. Frozen policy selection
14. Final holdout split plan
15. Final WFA, final predictions, final execution, final metrics
16. Placebo/simple-rule baselines
17. Prop-firm account simulation
18. Final strategy gate
```

---

## Codex Implementation Prompt

Update the repo to match this `project_layout.md`.

Primary changes to implement now:

```text
1. Add or update the Pipeline Structure Overview section in project_layout.md.
2. Keep Phase 1 as DBN archive plus DBN-to-parquet stitch, ending at data/raw/{market}/{year}.parquet with the strict Databento OHLCV schema.
3. Keep `scripts.phase2_causal_base.build_causal_base_data` as the single validation/session-normalization/causal-gating module.
4. Add roll-window flags and roll-window label invalidation.
5. Add cost-aware target columns in ticks and dollars.
6. Add research/final-holdout separation: 2023-2024 research, 2025 final holdout.
7. Add train-only feature selection and frozen feature list.
8. Add frozen execution policy selection before final holdout.
9. Add final execution + cost model before final metrics.
10. Add cost stress reports: cost_1x, cost_2x, cost_3x.
11. Add placebo/simple-rule baseline comparisons.
12. Add prop-firm account simulation and rule-violation reports.
13. Update strategy gate so it can fail economic viability, cost-stress viability, and prop-firm viability separately from pipeline integrity.
14. Add tests listed in the Required Tests section.
```

Do not add new markets, alternative data, order book data, or hyperparameter tuning yet.

