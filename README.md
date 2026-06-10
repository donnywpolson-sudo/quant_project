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

Set the key in your shell only. Do not write it to repo files.

```powershell
$env:DATABENTO_API_KEY="YOUR_KEY"
```

## Raw Data Download

Raw files are written as:

```text
data/raw/{market}/{year}.parquet
```

Smoke test:

```powershell
python scripts\download_databento_raw.py --symbols ES --start-year 2026 --end-year 2026 --end-date 2026-01-03 --out data\raw_api_test --execute --overwrite
```

Full L0/OHLCV archive:

```powershell
python scripts\download_databento_raw.py --universe extended_cme_vix --start-year 2010 --end-year 2026 --end-date 2026-06-10 --execute
```

The downloader does not replace existing files unless `--overwrite` is passed.

## Causal Base

Build the normalized causal base from every raw market/year file:

```powershell
python scripts\build_causal_base_data.py --profile all_raw
```

Output:

```text
data/causally_gated_normalized/{market}/{year}.parquet
reports/causal_base/
```

## Tests

```powershell
python -m pytest -q
```

## Simple GitHub Sync

Push this computer's code changes to GitHub:

```powershell
python push_github.py
```

Pull GitHub changes onto this computer before working:

```powershell
python pull_github.py
```

These scripts are for code/config/docs/tests only. Raw data and generated reports stay out of GitHub.
