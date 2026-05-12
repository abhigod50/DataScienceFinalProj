# Zip Portability Verification

## Verified Result

This repository was verified in this session by:

1. creating a real zip archive of the project folder
2. extracting it to a different path with spaces in the directory name
3. running the extracted copy from that new location

Verified extracted path:

`%TEMP%\CS439 Portable Check\Extracted Repo`

## What Was Run Successfully From The Extracted Copy

- `scripts/run_setup_doctor.ps1`
- `scripts/run_backtest.ps1`
- `paper/generate_paper_assets.py`

Observed result:

- setup doctor passed
- baseline backtest completed and wrote a fresh `backtest_results.json` inside the extracted folder
- paper figures and `paper/results_snapshot.md` were regenerated inside the extracted folder

## What This Proves

- The folder can be zipped, extracted elsewhere, and run without depending on the original `C:\freqtrade\cs439_final_project` path.
- The project does not depend on `models/latest` being a symlink; it is a normal directory.
- The PowerShell wrappers work even when the extracted folder has no parent `.venv`, as long as `python` is available on `PATH`.

## What This Does Not Prove

It does not prove literal compatibility with every possible PC regardless of environment. A target machine still needs:

- Python 3.11+
- the packages in `requirements.txt`
- Windows PowerShell if you want to use the wrapper scripts
- enough disk space and memory for the included data and models

If a PC does not have PowerShell wrappers available or preferred, the direct Python commands in [README.md](README.md) are the fallback.

## Practical Recommendation

You can zip this folder and share it as a standalone project submission. For the receiving PC, the safest setup is:

1. install Python 3.11 or newer
2. install dependencies with `python -m pip install -r requirements.txt`
3. run `pwsh ./scripts/run_setup_doctor.ps1`
4. run the needed script or paper-generation command

## Current Repository Size

The repository measured approximately `77.6 MB` before zipping, so it is small enough to share as a normal class-project archive.
