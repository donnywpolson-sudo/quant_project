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

Put the Databento API key in `secrets/databento.env`:

```powershell
Set-Content -Path .\secrets\databento.env -Value 'DATABENTO_API_KEY="YOUR_KEY"' -Encoding utf8
```

The raw downloader reads `secrets/databento.env` first, then `databento.env` at
the project root. Both forms are git-ignored and may contain either
`DATABENTO_API_KEY=...` or the raw key as the only non-comment line.

## Raw Data Ingest

Phase 1A implementation: `scripts/phase1A_download/download_databento_raw.py`.
Phase 1B implementation: `scripts/phase1B_convert/convert_databento_raw.py`.

Phase 1A archives both required Databento DBN/DBN.ZST schemas by default:

```text
data/dbn/ohlcv_1m/{market}/{year}/{chunk_start}_{chunk_end}.dbn.zst
data/dbn/definition/{market}/{year}/{chunk_start}_{chunk_end}.dbn.zst
```

Phase 1B converts and stitches DBN chunks into immutable raw parquet:

```text
data/raw/{market}/{year}.parquet
```

```powershell
python -m scripts.phase1B_convert.convert_databento_raw --dbn-root data\dbn\ohlcv_1m --raw-root data\raw
```

Smoke test:

```powershell
python -m scripts.phase1A_download.download_databento_raw --symbols ES --start 2026-01-01 --end 2026-01-03 --dry-run
```

Full Phase 1A archive:

```powershell
python -m scripts.phase1A_download.download_databento_raw --universe extended_cme --start-year 2010 --end-year 2026 --end-date 2026-06-10
```

Project-wide excluded markets: `E7`, `J7`, `QI`, `QO`, `ZQ`, `PA`.

The downloader does not replace existing files unless `--overwrite` is passed.

## Project Profiles

Operational profiles live in `configs/alpha_tiered.yaml`.

```text
tier_0 = ES smoke test
tier_1 = CL/ES/ZN recent core
tier_2 = CL/ES/ZN long core
tier_3 = exact 31-market GLBX-only long universe
all_raw = inventory only
metadata_optional_test = unit-test only
```

Default profile: `tier_1`.

Use `tier_1` for current Phase 1-4 debugging. `tier_1` results do not
prove `tier_3` performance; `tier_3` is the actual research universe. Missing
Tier-3 data must fail stage validation clearly, not silently shrink the
universe.

## Causal Base

Canonical implementation: `scripts/phase2_causal_base/build_causal_base_data.py`.

Build the normalized causal base for the tier-1 machinery proof set:

```powershell
python -m scripts.phase2_causal_base.build_causal_base_data --profile tier_1
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
python -m scripts.phase3_labels.build_labels --profile tier_1
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
python -m scripts.phase4_features.build_baseline_features --profile tier_1
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
