# Downstream ML Model Plan

Status: Phases 1-4 are complete. Do not jump to advanced ML until the pipeline can benchmark models honestly with walk-forward splits, purge, costs, turnover, and per-market stability.

## Core principle

The best model is unknown until tested. The downstream plan is not "use XGBoost." The plan is:

1. Start with simple linear control models.
2. Add one nonlinear challenger.
3. Compare everything under the exact same WFA/cost/turnover framework.
4. Only keep models that improve net performance and reduce failure modes.

## Primary targets

Use models against these targets:

| Target | Purpose |
|---|---|
| `target_return_15m` | Predict expected 15-minute forward return |
| `target_direction_15m` | Predict long / short / flat direction |
| `target_fade_success` | Predict whether fading a move is likely to work |
| `target_trend_danger` | Predict when not to fade / trend-day danger |

The most important target for my trading style may be `target_trend_danger`, not raw return prediction.

## Phase 5A: Linear baseline models

These are the control models.

| Target | Model |
|---|---|
| `target_return_15m` | Ridge Regression |
| `target_direction_15m` | Logistic Regression |
| `target_fade_success` | Logistic Regression |
| `target_trend_danger` | Logistic Regression |

Purpose:

- Fast
- Interpretable
- Harder to overfit
- Baseline for all future models
- Confirms whether features have any signal at all

Do not skip this phase.

## Phase 5B: First nonlinear challenger

Use:

| Model | Purpose |
|---|---|
| `HistGradientBoostingClassifier` | First nonlinear classifier |
| `HistGradientBoostingRegressor` | First nonlinear return model |

Purpose:

- Tests whether nonlinear feature interactions matter
- Avoids external dependency complexity at first
- Good bridge before LightGBM/XGBoost

## Phase 5C: Serious nonlinear challenger

Use one of:

| Model | Priority |
|---|---|
| LightGBM | First choice |
| XGBoost | Second choice |
| CatBoost | Later only if categorical regime/symbol/session labels are added |

Use CPU first. GPU is optional later.

Main uses:

| Target | Nonlinear model |
|---|---|
| `target_direction_15m` | LightGBM / XGBoost classifier |
| `target_fade_success` | LightGBM / XGBoost classifier |
| `target_trend_danger` | LightGBM / XGBoost classifier |
| `target_return_15m` | LightGBM / XGBoost regressor |

## Phase 6: Signal calibration

Raw model scores should not directly become trades.

Convert predictions into calibrated probabilities/scores:

| Output | Meaning |
|---|---|
| `p_long` | Probability long is favorable |
| `p_short` | Probability short is favorable |
| `p_fade_success` | Probability fade setup works |
| `p_trend_danger` | Probability fading is dangerous |
| `expected_return` | Predicted forward return |
| `expected_cost_adjusted_return` | Predicted return after estimated costs |

Then apply thresholds and no-trade zones.

## Phase 7: Position policy

The model predicts. The position policy trades.

Position policy decides:

- flat / long / short
- trade or no trade
- fade allowed or blocked
- position size
- whether adds are allowed
- whether trend danger forces flat

Important rule:

`target_trend_danger` should be used to block bad fade trades.

## Phase 8: Ensemble candidate

Only after individual models are tested.

Possible blend:

| Component | Role |
|---|---|
| Logistic Regression | Stable linear baseline |
| LightGBM / XGBoost | Nonlinear interaction model |
| Trend-danger classifier | Risk filter |
| Fade-success classifier | Style-specific filter |

The final model should probably be a small ensemble, not one magic model.

## Explicitly defer

Do not use these yet:

| Model type | Reason |
|---|---|
| Neural nets / MLP | Too easy to overfit early |
| LSTM / GRU | Sequence complexity and leakage risk |
| Transformers | Overkill for current stage |
| Reinforcement learning | Requires realistic simulator and mature execution model |
| Massive hyperparameter tuning | Wasteful before labels/costs/position policy are proven |

## Hardware constraint

Current machine:

- Ryzen 5 2600, 6 cores / 12 threads
- 32 GB RAM
- GTX 1070 Ti, 8 GB VRAM

Recommended approach:

- CPU-first tabular ML
- Use `float32`
- Train fold-by-fold
- Avoid loading unnecessary markets/years at once
- Try GPU only after CPU LightGBM/XGBoost works

## Model acceptance criteria

A model is useful only if it improves:

| Metric | Requirement |
|---|---|
| Net return | Better than linear baseline |
| Net Sharpe | Better than linear baseline |
| Cost drag | Not destroying gross edge |
| Turnover/bar | Controlled |
| Per-market stability | Not one-market fake alpha |
| Per-fold stability | Not one lucky period |
| Trend-day behavior | Reduces fade disasters |
| Fade filter | Blocks bad fades without killing all trades |

## Best downstream order

1. Ridge return baseline
2. Logistic direction baseline
3. Logistic fade-success baseline
4. Logistic trend-danger baseline
5. HistGradientBoosting challenger
6. LightGBM or XGBoost challenger
7. Probability calibration
8. Position policy
9. Linear + boosted-tree blend
10. Advanced regime/deep learning only much later