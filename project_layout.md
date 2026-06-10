# Quant Project Layout

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

### Raw data contract

Raw input files are Databento continuous-contract 1-minute OHLCV parquet files.

Path pattern:

```text
data/raw/{market}/{year}.parquet
```

Required raw schema:

```text
rtype
publisher_id
instrument_id
open
high
low
close
volume
symbol
ts_event
```

Timestamp policy:

```text
ts_event -> ts
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

The pipeline must report symbol and instrument changes, flag roll boundaries, invalidate suspicious roll-window labels, and eventually support contract-specific execution mapping before live deployment.

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

Initial research universe:

```text
CL
ES
ZN
```

Initial years:

```text
2023
2024
2025
```

Recommended split with only these years:

```text
2023-2024 = research / feature discovery / policy selection
2025      = final holdout evaluation
```

Do not add new markets, alternative data, order book data, or hyperparameter tuning until the baseline pipeline is structurally correct and the strategy gates are honest.

---

### Pipeline structure overview

This is the realistic operational pipeline. The old 27-stage checklist is preserved conceptually, but implementation is consolidated into fewer runnable stages with stronger reports and gates.

| Phase | Name | Main artifact | Purpose |
|---:|---|---|---|
| 1 | Raw Data | `data/raw/{market}/{year}.parquet` | Immutable Databento OHLCV input. |
| 2 | Causal Base Builder | `data/causally_gated_normalized/{market}/{year}.parquet` | Validate, session-normalize, roll-flag, synthetic-mark, and causally gate raw bars. |
| 3 | Target / Label Generation | `data/labeled/{market}/{year}.parquet` | Build next-bar-entry 15-minute labels with cost-aware and intraday validity flags. |
| 4 | Baseline Feature Matrix | `data/feature_matrices/baseline/{market}/{year}.parquet` | Build OHLCV-only baseline features plus metadata and target columns. |
| 5 | Column Registry | `feature_cols.json`, `target_cols.json`, `metadata_cols.json` | Hard-gate feature/target/metadata separation. |
| 6 | WFA Split Plan | `reports/wfa/split_plan.json` | Build deterministic train/test folds with purge and final-holdout awareness. |
| 7 | Baseline WFA Train/Test | `reports/wfa/baseline_wfa_report.json` | Train baseline model using train-only preprocessing and test-only prediction. |
| 8 | Baseline OOS Predictions | `data/predictions/baseline/oos_predictions.parquet` | Store OOS predictions with execution-ready prices. |
| 9 | Baseline Execution + Cost Model | `data/executions/baseline/executions.parquet` | Convert predictions to positions, apply costs, enforce no-overnight rules. |
| 10 | Baseline Metrics + Diagnostics | `reports/metrics/baseline_metrics.json` | Evaluate prediction behavior, turnover, cost drag, and net economics. |
| 11 | Baseline Accept / Reject Gate | `reports/gates/baseline_gate.json` | Separate structural pass/fail from economic pass/fail. |
| 12 | Feature Expansion | `data/feature_matrices/expanded/{market}/{year}.parquet` | Add broader OHLCV-only candidates. |
| 13 | Feature Discovery | `reports/feature_discovery/` | Analyze nulls, stability, redundancy, leakage, and train-only correlations. |
| 14 | Train-Only Feature Ranking / Selection | `reports/feature_selection/` | Select features without using final holdout or test-fold information. |
| 15 | Frozen Feature + Policy Set | `data/frozen_features/phase5_v1/` | Freeze features and execution policy before final evaluation. |
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
Old stages 1-8   -> Phase 1-2: raw data plus causal base builder
Old stages 9-13  -> Phase 3-5: labels, baseline feature matrix, registry
Old stages 14-19 -> Phase 6-11: WFA, predictions, execution, metrics, baseline gate
Old stages 20-23 -> Phase 12-15: feature expansion, discovery, selection, frozen set
Old stages 24-27 -> Phase 16-22: final holdout, final WFA, final predictions, final execution, final metrics, prop simulation, strategy gate
```

---

### Core artifact flow

```text
data/raw/{market}/{year}.parquet
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
-> data/frozen_features/phase5_v1/policy_config.json
-> reports/final_wfa/final_split_plan.json
-> data/predictions/final/oos_predictions.parquet
-> data/executions/final/executions.parquet
-> reports/final_metrics/final_metrics.json
-> reports/prop_sim/final_prop_simulation.json
-> reports/gates/strategy_gate.json
```

---

### Research, validation, and final holdout policy

A walk-forward split alone is not enough if feature selection, policy selection, and human iteration keep using the same period.

Minimum policy for current data:

```text
research period: 2023-2024
final holdout:   2025
```

Rules:

- Feature discovery and feature selection may use only research-period train folds.
- Execution policy selection may use only research-period train/validation folds.
- Final holdout may be evaluated only after features and policy are frozen.
- Do not change features, thresholds, policy rules, or costs after inspecting final holdout results.
- If final holdout fails, mark the strategy rejected; do not tune on the final holdout.

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
- Final WFA uses frozen features and frozen policy only.

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
```

`target_valid` must be false if:

```text
current row is not causal_valid
entry row is missing
exit row is missing
entry or exit touches a synthetic row
future path crosses session_segment_id
future path crosses configured intraday cutoff
future path crosses suspicious roll boundary
time remaining is insufficient for entry and exit
```

---

### Standard commands

Causal base:

```bash
python scripts/build_causal_base_data.py --profile tier_1_CL_ES_ZN
```

Baseline pipeline:

```bash
python scripts/run_pipeline.py --profile tier_1_CL_ES_ZN --from-stage labels --to-stage baseline_gate
```

Feature expansion and final pipeline:

```bash
python scripts/run_pipeline.py --profile tier_1_CL_ES_ZN --from-stage feature_expansion --to-stage strategy_gate
```

Tests:

```bash
pytest -q
```

Git check:

```bash
git status --short
```

---

## Part 2 — Phase-by-Phase Build Requirements

# Phase 1 — Raw Data

## Purpose

Store immutable Databento continuous-contract 1-minute OHLCV parquet files.

## Input

External Databento parquet exports.

## Required path

```text
data/raw/{market}/{year}.parquet
```

## Required raw schema

```text
rtype
publisher_id
instrument_id
open
high
low
close
volume
symbol
ts_event
```

## Build requirements

- Do not mutate raw files.
- Do not overwrite raw files during pipeline runs.
- Do not track raw files in git.
- Ensure market and year can be inferred from the path.
- Preserve Databento metadata for audit.
- Report file hash, first timestamp, last timestamp, and row count.

## Breakpoints

- Missing required columns.
- Bad timestamp type.
- Duplicate `ts_event` rows.
- Mixed or unexpected symbols without roll reporting.
- Calendar-year files that split Globex sessions.
- Continuous-contract adjustment artifacts treated as alpha.

## Acceptance checks

- File exists for each configured market/year.
- File is non-empty.
- Required columns exist.
- Path market/year matches configured profile.

---

# Phase 2 — Causal Base Builder

## Purpose

Validate raw OHLCV data, normalize sessions, handle missing session minutes, mark synthetic rows, flag roll transitions, add session metadata, and produce the only approved modeling base table.

This phase replaces separate materialized stages for raw manifest, validation, validated data, session normalization, session-normalized data, causal gating, and causal output.

## Script

```bash
python scripts/build_causal_base_data.py --profile tier_1_CL_ES_ZN
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
raw_row_present
is_synthetic
causal_valid
session_id
session_date
session_segment_id
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
causal_valid = raw_row_present and not is_synthetic and valid_ohlcv and inside_valid_session
```

## Hard failures

```text
missing required columns
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
python scripts/build_labels.py --profile tier_1_CL_ES_ZN
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
```

## Required validity logic

`target_valid` must be false if:

```text
current row causal_valid is false
entry row is unavailable
exit row is unavailable
future path touches synthetic rows
future path crosses session_segment_id
future path crosses roll_boundary_flag or roll_window_flag
future path crosses intraday flatten cutoff
minutes_until_session_close is insufficient
```

## Cost-aware target logic

- Directional labels alone are not enough.
- A tiny positive move that cannot beat costs should not be treated as a tradeable long opportunity.
- `target_sign_with_deadzone` should be neutral when absolute target move is below estimated round-trip cost plus configured buffer.
- `target_tradeable_after_cost` should be true only when the forward move exceeds estimated costs.

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

# Phase 4 — Baseline Feature Matrix

## Purpose

Create the initial OHLCV-only modeling matrix with baseline features, metadata columns, target columns, and validity columns.

## Script

```bash
python scripts/build_baseline_features.py --profile tier_1_CL_ES_ZN
```

## Input

```text
data/labeled/{market}/{year}.parquet
```

## Output

```text
data/feature_matrices/baseline/{market}/{year}.parquet
reports/features/baseline_feature_report.json
```

## Baseline feature families

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

## Recommended candidate columns

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

## Required logic

- Rolling features are grouped by `session_segment_id`.
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

Freeze column roles for modeling and prevent obvious leakage.

## Script

```bash
python scripts/build_column_registry.py --profile tier_1_CL_ES_ZN --matrix baseline
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
target_entry_ts
target_exit_ts
target_entry_price
target_exit_price
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
```

## Build requirements

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
python scripts/build_wfa_splits.py --profile tier_1_CL_ES_ZN
```

## Output

```text
reports/wfa/split_plan.csv
reports/wfa/split_plan.json
```

## Default research policy

```text
research_years = 2023-2024
final_holdout_years = 2025
train_days = 365
test_days = 30
step_days = 30
purge_bars = 15
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

# Phase 7 — Baseline WFA Train / Test

## Purpose

Train the baseline model on each train fold and generate out-of-sample predictions for each test fold.

## Script

```bash
python scripts/run_wfa.py --profile tier_1_CL_ES_ZN --matrix baseline --run baseline
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

```text
model = Ridge baseline
imputer = train-only
scaler = train-only
fit = train fold only
predict = test fold only
hyperparameter tuning = disabled
```

## Build requirements

- Filter training rows to `causal_valid == true` and `target_valid == true`.
- Fit imputer only on train fold.
- Fit scaler only on train fold.
- Fit model only on train fold.
- Predict only test rows.
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

Write out-of-sample predictions with enough price and metadata columns for execution without unsafe joins.

## Output

```text
data/predictions/baseline/oos_predictions.parquet
reports/wfa/baseline_predictions_manifest.json
```

## Required columns

```text
market
year
ts
fold_id
split_group
y_pred
y_true
target_ret_15m
target_ret_ticks_15m
target_sign_15m
target_sign_with_deadzone
target_tradeable_after_cost
target_valid
causal_valid
close
target_entry_ts
target_exit_ts
target_entry_price
target_exit_price
execution_price
exit_price
session_segment_id
minutes_until_session_close
```

## Build requirements

- Predictions are unique by `market`, `ts`, and `fold_id`.
- Predictions are OOS only.
- Execution price and exit price are included or derivable without rejoining raw data.
- Invalid target rows are excluded from model scoring unless explicitly reported as skipped.

## Acceptance checks

- OOS prediction parquet exists.
- No duplicate prediction rows.
- Prediction rows are test-fold rows only.
- Required execution columns exist.

---

# Phase 9 — Baseline Execution + Cost Model

## Purpose

Convert predictions into positions, compare deterministic position policies, charge realistic market-specific costs, and compute net returns in returns, ticks, and dollars.

## Script

```bash
python scripts/run_execution_costs.py --profile tier_1_CL_ES_ZN --run baseline
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
raw prediction
-> signal threshold / no-trade band
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
ts
fold_id
split_group
policy_name
cost_scenario
y_pred
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
python scripts/build_metrics.py --profile tier_1_CL_ES_ZN --run baseline
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
python scripts/run_gate.py --profile tier_1_CL_ES_ZN --run baseline
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
python scripts/build_expanded_features.py --profile tier_1_CL_ES_ZN
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
python scripts/run_feature_discovery.py --profile tier_1_CL_ES_ZN
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
python scripts/run_feature_selection.py --profile tier_1_CL_ES_ZN
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

# Phase 15 — Frozen Feature + Policy Set

## Purpose

Freeze selected features and the selected execution policy before final evaluation.

## Script

```bash
python scripts/freeze_features.py --profile tier_1_CL_ES_ZN
python scripts/freeze_policy.py --profile tier_1_CL_ES_ZN
```

## Output

```text
data/frozen_features/phase5_v1/feature_cols.json
data/frozen_features/phase5_v1/selected_features.csv
data/frozen_features/phase5_v1/rejected_features.csv
data/frozen_features/phase5_v1/policy_config.json
data/frozen_features/phase5_v1/manifest.json
```

## Build requirements

- Frozen feature list is immutable during final WFA.
- Frozen policy is immutable during final execution.
- Manifest includes feature-selection report hash and policy-selection report hash.
- Final WFA must consume only frozen features.
- Final execution must consume only frozen policy unless running diagnostic comparison that is not used for the gate.

## Acceptance checks

- Frozen feature files exist.
- Frozen policy file exists.
- Feature count is reported.
- Frozen features match selected features.
- Frozen policy was selected without final-holdout data.

---

# Phase 16 — Final Holdout Split Plan

## Purpose

Create the final evaluation split plan while ensuring final test windows are entirely inside the untouched holdout period.

## Script

```bash
python scripts/build_final_splits.py --profile tier_1_CL_ES_ZN
```

## Output

```text
reports/final_wfa/final_split_plan.csv
reports/final_wfa/final_split_plan.json
```

## Build requirements

- Test windows must be inside `final_holdout_years`.
- Training windows may use only data available before the test window.
- Feature list and policy must already be frozen.
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
python scripts/run_wfa.py --profile tier_1_CL_ES_ZN --matrix expanded --features data/frozen_features/phase5_v1/feature_cols.json --split-plan reports/final_wfa/final_split_plan.json --run final
```

## Input

```text
data/feature_matrices/expanded/
data/frozen_features/phase5_v1/feature_cols.json
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
- Test-fold-only predictions.
- No feature or policy changes after seeing final results.

## Acceptance checks

- Final WFA report exists.
- Frozen feature list was used exactly.
- Fold diagnostics are complete.
- Final folds are holdout-tagged.

---

# Phase 18 — Final OOS Predictions

## Purpose

Write final out-of-sample predictions with execution-ready columns.

## Output

```text
data/predictions/final/oos_predictions.parquet
reports/final_wfa/final_predictions_manifest.json
```

## Required columns

```text
market
year
ts
fold_id
split_group
y_pred
y_true
target_ret_15m
target_ret_ticks_15m
target_sign_15m
target_sign_with_deadzone
target_tradeable_after_cost
target_valid
causal_valid
close
target_entry_ts
target_exit_ts
target_entry_price
target_exit_price
execution_price
exit_price
session_segment_id
minutes_until_session_close
```

## Acceptance checks

- Predictions are OOS only.
- Predictions are final-holdout test rows only.
- No duplicate prediction rows.
- Execution-ready prices exist.

---

# Phase 19 — Final Execution + Cost Model

## Purpose

Apply the frozen execution and cost policy to final predictions.

## Script

```bash
python scripts/run_execution_costs.py --profile tier_1_CL_ES_ZN --run final --policy data/frozen_features/phase5_v1/policy_config.json
```

## Input

```text
data/predictions/final/oos_predictions.parquet
data/frozen_features/phase5_v1/policy_config.json
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
- Same intraday flatten rule.
- No overnight positions.
- Report cost stress scenarios, but do not choose a new policy from final stress results.

## Acceptance checks

- Final execution artifact exists.
- Costs reconcile.
- No positions remain open after cutoff.
- Frozen policy was used for final gate metrics.

---

# Phase 20 — Final Metrics + Diagnostics

## Purpose

Evaluate final model economics and compare against baseline, placebo, simple-rule baselines, and cost stress scenarios.

## Script

```bash
python scripts/build_metrics.py --profile tier_1_CL_ES_ZN --run final
python scripts/run_placebo_baselines.py --profile tier_1_CL_ES_ZN --run final
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
python scripts/run_prop_simulation.py --profile tier_1_CL_ES_ZN --run final
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
python scripts/run_gate.py --profile tier_1_CL_ES_ZN --run final
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
2. Keep raw input as data/raw/{market}/{year}.parquet with Databento OHLCV schema.
3. Keep build_causal_base_data.py as the single validation/session-normalization/causal-gating script.
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
