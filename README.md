# CS439 Final Project Submission Folder

This directory is a standalone, submission-focused copy of the research pipeline from the larger trading workspace. It is scoped to offline data science work only: feature engineering, model training, walk-forward validation, backtesting, ablation experiments, and analysis.

It does not require the rest of the `freqtrade` repository or any live Hummingbot services to run the core project workflow.

## Included

- Local market data snapshot for the main project pair universe:
  - `data/binanceus/ETH_USDT-5m.feather`
  - `data/binanceus/BTC_USDT-5m.feather`
  - `data/binanceus/SOL_USDT-5m.feather`
  - `data/coinbase/ETH_USD-5m.feather`
- Cached microstructure features:
  - `data/binanceus/orderbook/*`
  - `data/execution/*`
- Current promoted model artifacts:
  - `models/latest/*`
- Existing result files:
  - `walk_forward_results.json`
  - `backtest_results.json`
  - `backtest_ab_results.json`
  - `as_tuning_results.json`
- Research scripts:
  - `train.py`
  - `backtest.py`
  - `experiments.py`
  - `update_data.py`
  - `setup_doctor.py`
- Baseline strategy:
  - `baselines/EthDryRun.py`

## Project Layout

```text
cs439_final_project/
  backtest.py
  train.py
  experiments.py
  features.py
  ...
  shared/
  baselines/
  config/
  data/
  models/
  logs/
  notebooks/
  scripts/
```

## Quick Start

If you want to keep using the existing parent environment from `C:\freqtrade\.venv`, the PowerShell scripts in `scripts/` will pick it up automatically.

The wrappers are still portable when this folder is moved or extracted elsewhere. If neither a local `.venv` nor a parent `.venv` exists, they fall back to `python` on `PATH`.

### 1. Sanity check the standalone folder

```powershell
pwsh ./scripts/run_setup_doctor.ps1
```

### 2. Run a backtest with the included latest model snapshot

```powershell
pwsh ./scripts/run_backtest.ps1
```

### 3. Run the A/B experiment harness

```powershell
pwsh ./scripts/run_backtest_ab.ps1
```

### 4. Retrain models from the included local data

```powershell
pwsh ./scripts/run_train.ps1
```

### 5. Run feature/model ablations

```powershell
pwsh ./scripts/run_experiments.ps1
```

## Direct Python Commands

If you prefer not to use the PowerShell wrappers:

```powershell
python setup_doctor.py --quick
python backtest.py
python backtest.py --ab
python train.py
python experiments.py
```

## Refreshing Data

The copied data is enough for immediate offline work, but you can refresh it locally:

```powershell
python update_data.py --exchange binanceus --days 60 --pairs ETH/USDT BTC/USDT SOL/USDT
python update_data.py --exchange coinbase --days 60 --pairs ETH/USD
```

## Notes For Submission

- This folder is intended to be the clean GitHub repo root for the class project.
- It is intentionally offline-only and reproducible.
- The live trading and dashboard infrastructure from the original workspace has been excluded.
- `config/ml_mm.yml` and `config/conf_fee_overrides.yml` are minimal local config files used only by the offline backtest/training utilities.

## Editable Paper

The main editable paper draft is:

- `paper/final_paper.md`

Supporting paper artifacts are:

- `paper/results_snapshot.md`
- `paper/rerun_notes.md`
- `paper/figures/*`

Export an editable Word document with:

```powershell
python ./paper/export_docx.py
```

Regenerate figures and the snapshot with:

```powershell
python ./paper/generate_paper_assets.py
```

## Zip Portability

This repository was verified in this session by zipping it, extracting it to a different path with spaces, and running the extracted copy successfully.

Verified commands from the extracted copy:

- `scripts/run_setup_doctor.ps1`
- `scripts/run_backtest.ps1`
- `paper/generate_paper_assets.py`

For full details and caveats, see `ZIP_PORTABILITY.md`.

## Recommended Paper Framing

Use this folder to support a project framed around:

`Do multi-horizon, regime-aware ML signals improve short-horizon crypto direction prediction and simulated market-making performance over simpler baselines?`

That framing matches the included code and current result artifacts.
