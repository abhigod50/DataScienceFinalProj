# Teammate Handoff Guide

This is the easiest way to share the project and paper so another person can work on both without fighting file versions.

## Recommended Setup

Use two tools:

1. GitHub for the project code, data files, scripts, figures, and results.
2. Google Docs and Google Sheets for the paper text and editable charts.

That split is the easiest because code and result files version well in GitHub, while paper writing and chart editing are easier in Google tools.

## Best Workflow

### 1. Upload the project to a private GitHub repo

The easiest version is:

1. Create a new private GitHub repository.
2. Upload the contents of this folder as the repo root.
3. Add your teammate as a collaborator.

If you want the easiest desktop workflow, use GitHub Desktop instead of the command line.

### 2. Have your teammate clone the repo

Your teammate should clone the repo locally, then run:

```powershell
python -m pip install -r requirements.txt
pwsh ./scripts/run_setup_doctor.ps1
```

That gives them a local working copy for code, backtests, figures, and regenerated outputs.

### 3. Share the paper as a Google Doc

For the paper itself, upload this file to Google Drive:

- `paper/final_paper.docx`

Then open it with Google Docs and share that Google Doc with your teammate.

This is easier than asking both people to edit the Markdown file directly.

### 4. Share editable charts through Google Sheets

If your teammate needs to edit graphs, upload the CSVs from:

- `paper/figure_data/`

Open each CSV in Google Sheets, build or edit the chart there, and insert the linked chart into Google Docs.

Useful CSV files are:

- `paper/figure_data/walk_forward_summary.csv`
- `paper/figure_data/backtest_comparison.csv`
- `paper/figure_data/ab_experiments.csv`
- `paper/figure_data/ab_experiments_full_snapshot.csv`
- `paper/figure_data/as_tuning.csv`
- `paper/figure_data/ablation_summary.csv`

## What To Send Your Teammate

Send them these three things:

1. the private GitHub repo link
2. the Google Docs link for `paper/final_paper.docx`
3. the Google Drive or Sheets links for any chart CSVs they need to edit

## What Each Person Should Edit

Use this split to avoid conflicts:

- Edit code, JSON results, scripts, and generated figures through the GitHub repo.
- Edit paper wording in Google Docs.
- Edit charts in Google Sheets.

When major figure numbers change, regenerate them locally with:

```powershell
python ./paper/generate_paper_assets.py
python ./paper/export_docx.py
```

Then re-upload the updated DOCX and any changed CSVs to Drive.

## Easiest Non-Git Fallback

If you want the absolute simplest sharing method and only one person edits at a time:

1. zip the whole project folder
2. upload the zip to Google Drive or OneDrive
3. have your teammate extract it locally
4. share the paper separately as Google Docs

This works because the repository was already verified as portable after zip and extract.

The downside is that this is worse for collaboration because it is easy to overwrite each other's changes.

## My Recommendation

The easiest setup for a class teammate is:

1. GitHub private repo for the project
2. Google Docs for the paper text
3. Google Sheets for figure edits

That gives you the least friction and the lowest chance of losing work.
