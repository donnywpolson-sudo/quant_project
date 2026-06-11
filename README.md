# Quant Project

Intraday futures research pipeline using Databento continuous-contract 1-minute OHLCV data.

## Environment

Use Python 3.11.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Databento API Key

Put the Databento API key in `databento.env` at the project root:

```powershell
Set-Content -Path .\databento.env -Value 'DATABENTO_API_KEY="YOUR_KEY"' -Encoding utf8
```

The raw downloader only reads `DATABENTO_API_KEY` from that file. `databento.env` is git-ignored.
It also accepts a raw key as the only non-comment line in `databento.env`.

## Raw Data Download

Canonical implementation: `scripts/raw_ingest/download_databento_raw.py`.

Raw files are written as:

```text
data/raw/{market}/{year}.parquet
```

Smoke test:

```powershell
python -m scripts.raw_ingest.download_databento_raw --symbols ES --start-year 2026 --end-year 2026 --end-date 2026-01-03 --out data\raw_api_test --overwrite
```

Full L0/OHLCV archive:

```powershell
python -m scripts.raw_ingest.download_databento_raw --universe extended_cme_vix --start-year 2010 --end-year 2026 --end-date 2026-06-10
```

The downloader does not replace existing files unless `--overwrite` is passed.

## Project Profiles

Operational profiles live in `configs/alpha_tiered.yaml`.

```text
tier_0 = smoke test
tier_1 = CL/ES/ZN machinery proof set
tier_2 = exact 28-market real universe
all_raw = inventory only
metadata_optional_test = unit-test only
```

Default profile: `tier_1_core_recent`.

Use `tier_1_core` for current Phase 1-4 debugging. `tier_1` results do not
prove `tier_2` performance; `tier_2` is the actual research universe. Missing
tier-2 data should fail stage validation clearly, not silently shrink the
universe.

## Causal Base

Canonical implementation: `scripts/phase2_causal_base/build_causal_base_data.py`.

Build the normalized causal base for the tier-1 machinery proof set:

```powershell
python -m scripts.phase2_causal_base.build_causal_base_data --profile tier_1_core
```

Output:

```text
data/causally_gated_normalized/{market}/{year}.parquet
reports/causal_base/
```

## Labels

Canonical implementation: `scripts/phase3_labels/build_labels.py`.

Build labels for the tier-1 machinery proof set:

```powershell
python -m scripts.phase3_labels.build_labels --profile tier_1_core
```

Output:

```text
data/labeled/{market}/{year}.parquet
reports/labels/
```

## Baseline Feature Matrix

Planned Phase 4 implementation:

```text
scripts/phase4_features/build_baseline_features.py
tests/phase4_features/test_build_baseline_features.py
```

Canonical command once implemented:

```powershell
python -m scripts.phase4_features.build_baseline_features --profile tier_1_core
```

## Tests

```powershell
python -m pytest -q
```

## Simple GitHub Sync

Stage, commit, rebase, and push all non-risky local changes from this computer to GitHub:

```powershell
python push_github.py
```

Pull GitHub changes onto this computer before working:

```powershell
python pull_github.py
```

`push_github.py` prints changed files, blocks risky data/secret/output paths, runs tests, creates backup branches, stages with `git add -A`, commits, pulls with `--rebase`, and pushes. Raw data and generated reports stay out of GitHub.
