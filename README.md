

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

### 1. Error check the standalone folder

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




## Zip Portability

This repository was verified in this session by zipping it, extracting it to a different path with spaces, and running the extracted copy successfully.

Verified commands from the extracted copy:

- `scripts/run_setup_doctor.ps1`
- `scripts/run_backtest.ps1`
- `paper/generate_paper_assets.py`

For full details and caveats, see `ZIP_PORTABILITY.md`.

