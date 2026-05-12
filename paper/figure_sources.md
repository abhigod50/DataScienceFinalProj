# Figure Sources

Use this file when you need to cite a figure's data source directly in the paper or in Google Docs.

## walk_forward_summary.png

- Editable chart data: `paper/figure_data/walk_forward_summary.csv`
- Underlying source file: `walk_forward_results.json`
- Source fields: `folds[*].auc`, `folds[*].accuracy`, `folds[*].sharpe_ratio`, `folds[*].total_return_pct`, `folds[*].total_trades`

## backtest_comparison.png

- Editable chart data: `paper/figure_data/backtest_comparison.csv`
- Underlying source file: `backtest_results.json`
- Source fields: `ml_metrics` and `fixed_metrics` for return, Sharpe, drawdown, and trade count

## ab_experiments.png

- Editable chart data: `paper/figure_data/ab_experiments.csv`
- Underlying source file: `backtest_ab_results.json`
- Source fields: `experiments[*].ml_metrics`, `strategy2_exec_grid.best_candidate.ml_metrics`, `strategy1_signal_search.best_candidate.ml_metrics`
- Scope: current full A/B rerun with named variants plus the strongest search winners.

## ab_experiments_full_snapshot.png

- Editable chart data: `paper/figure_data/ab_experiments_full_snapshot.csv`
- Underlying source file: `backtest_ab_results.json`
- Source fields: `experiments[*].ml_metrics`
- Scope: current full-rerun named variants only.

## as_tuning.png

- Editable chart data: `paper/figure_data/as_tuning.csv`
- Underlying source file: `as_tuning_results.json`
- Source fields: `baseline_metrics` and `top5_by_composite[*].metrics`

## ablation_summary.png

- Editable chart data: `paper/figure_data/ablation_summary.csv`
- Underlying source file: `experiments_results.json`
- Source fields: `baseline.direction_auc`, `baseline.volatility_r2`, and selected rows from `comparisons`
