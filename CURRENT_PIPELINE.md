# Current Pipeline

This project has a real research pipeline scaffold, but it is not yet a
complete alpha-generating or production backtesting system.

## Current Status

- `tier_1_research` has complete raw, causal, label, and baseline feature coverage.
- `tier_2_research` has complete raw, causal, and label coverage, but incomplete baseline feature coverage.
- `tier_3_research` has complete available raw, causal, and label coverage, but incomplete baseline feature coverage.
- Current baseline WFA evidence is partial and should not be treated as promoted alpha.
- Phase 8 policy diagnostics can evaluate saved predictions with costs, but they are not a live fill simulator or full execution backtester.

## Promotion Gates

Before calling the system research-alpha ready:

- Phase 4 baseline features must exist for the intended profile scope.
- Phase 7 must produce non-stale out-of-sample predictions across the intended WFA folds.
- Phase 8 must pass the costed promotion gate.
- Model promotion must remain blocked when net PnL, net Sharpe-like metrics, cost drag, or per-market/per-fold stability fail.

Before live or paper-live use:

- Contract-specific execution mapping must exist.
- Exchange calendar and early-close data must be refreshed.
- Fixed research slippage assumptions must be replaced by a live/paper fill model.
- A real execution layer must handle order lifecycle, fills, rejects, position state, risk limits, and audit logs.

## Phases In Simple Terms

- Phase 1A: download Databento DBN archives.
- Phase 1B: convert DBN archives into raw yearly parquet files.
- Phase 2: clean and normalize bars into causal session-aware data.
- Phase 3: create future-looking labels and cost-aware targets.
- Phase 4: build model features while excluding target and leakage columns.
- Phase 5: build walk-forward train/test splits with purge and embargo.
- Phase 6: no separate implemented phase in this repo.
- Phase 7: train baseline models and save out-of-sample predictions.
- Phase 8: score predictions with a deterministic research policy, costs, and promotion gates.

## Useful Checks

```powershell
python -m scripts.validation.check_tier_2_coverage --profile tier_1 --stage all
python -m scripts.validation.check_tier_2_coverage --profile tier_3 --stage features
python -m scripts.phase8_model_selection.evaluate_predictions --run baseline --require-promotion-ready
```
