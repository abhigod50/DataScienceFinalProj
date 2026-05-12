# Google Docs Graph Workflow

## Best Workflow

Google Docs is good for writing, but editable charts are better handled in Google Sheets.

Use this workflow:

1. Upload `paper/final_paper.docx` to Google Drive and open it in Google Docs.
2. Upload one of the CSV files from `paper/figure_data/` to Google Drive.
3. Open that CSV in Google Sheets.
4. In Google Sheets, select the table and use `Insert -> Chart`.
5. Edit chart type, colors, labels, and legends in the Sheets chart editor.
6. In Google Docs, insert the chart with `Insert -> Chart -> From Sheets`.
7. Keep the chart linked so later Sheets edits can update the chart in Docs.

## Which CSV To Use

- Walk-forward figure: `paper/figure_data/walk_forward_summary.csv`
- Main backtest figure: `paper/figure_data/backtest_comparison.csv`
- A/B figure: `paper/figure_data/ab_experiments.csv`
- Named-variants-only A/B view: `paper/figure_data/ab_experiments_full_snapshot.csv`
- Adverse-selection figure: `paper/figure_data/as_tuning.csv`
- Ablation figure: `paper/figure_data/ablation_summary.csv`

## How To Show The Data Source

The cleanest approach is to add a source line directly below each figure in the paper.

Recommended format:

`Source: Author-generated figure from walk_forward_results.json via paper/figure_data/walk_forward_summary.csv.`

Use `paper/figure_sources.md` to map each figure to:

- the editable CSV
- the original JSON artifact
- the fields used to build the plot

## Suggested Figure Caption Pattern

Caption line:

`Figure X. Purged walk-forward validation across six folds.`

Source line below it:

`Source: Author-generated from walk_forward_results.json via paper/figure_data/walk_forward_summary.csv.`

## Important A/B Note

The current `ab_experiments.png` and `paper/figure_data/ab_experiments.csv` are intended to reflect the latest full `backtest_ab_results.json` output.

That main A/B figure includes:

- the named strategy variants from `experiments[*]`
- the best execution-grid candidate from `strategy2_exec_grid.best_candidate`
- the best signal-search candidate from `strategy1_signal_search.best_candidate`

The companion named-variants-only view is preserved separately in:

- `paper/figures/ab_experiments_full_snapshot.png`
- `paper/figure_data/ab_experiments_full_snapshot.csv`

Both A/B files should be regenerated after each new full `python .\backtest.py --ab` run by rerunning `python .\paper\generate_paper_assets.py`.
