# Futures Quant Workflow

Active config: `configs/alpha_tiered.yaml`

## 27-stage pipeline

1. RAW DATA
2. RAW DATA MANIFEST
3. RAW DATA VALIDATION
4. VALIDATED DATA
5. SESSION NORMALIZATION
6. SESSION-NORMALIZED DATA
7. CAUSAL GATING
8. CAUSALLY GATED NORMALIZED DATA
9. TARGET / LABEL GENERATION
10. LABELED DATA
11. BASELINE FEATURE GENERATION
12. BASELINE FEATURE MATRIX
13. FEATURE / TARGET / METADATA COLUMN REGISTRY
14. WFA SPLIT PLAN
15. BASELINE WFA TRAIN / TEST
16. OOS PREDICTIONS
17. EXECUTION + COST MODEL
18. METRICS + DIAGNOSTICS
19. BASELINE ACCEPT / REJECT GATE
20. FEATURE EXPANSION
21. FEATURE DISCOVERY
22. TRAIN-ONLY FEATURE RANKING / SELECTION
23. FROZEN FEATURE SET
24. FINAL WFA WITH FROZEN FEATURES
25. FINAL OOS PREDICTIONS
26. FINAL METRICS + DIAGNOSTICS
27. STRATEGY ACCEPT / REJECT GATE

## Folder layout

```text
data/
  raw/
    manifest.json
    _manifest.csv
    {market}/{year}.parquet
  validated/
    manifest.json
    _manifest.csv
    {market}/{year}.parquet
  session_normalized/
    manifest.json
    _manifest.csv
    {market}/{year}.parquet
  causally_gated_normalized/
    manifest.json
    _manifest.csv
    {market}/{year}.parquet
  labeled/
  features_baseline/
  feature_matrices/
    baseline/
    expanded/
  frozen_features/

reports/
  validation/
  session_normalization/
  causal_gating/
  wfa/
  metrics/

artifacts/
  models/
  scalers/
  selectors/
  run_manifests/
  backtests/

configs/
  alpha_tiered.yaml
  raw_data_validation.yaml
  market_specs.yaml
```

`data/validated/` currently has manifests only unless raw validation is run with
`--write-validated --clean-policy drop-invalid`.
