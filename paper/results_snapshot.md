# Results Snapshot

Generated at: 2026-05-10T21:51:27.168446+00:00

## Walk-Forward

- Folds: 6
- Mean AUC: 0.785
- Mean Sharpe: -1.839
- Positive last 6 folds: 2

## Main Backtest

- ML return: -0.007% vs fixed -0.029%
- ML Sharpe: -0.616 vs fixed -0.529
- ML max drawdown: -0.054% vs fixed -0.276%
- ML trades: 79 vs fixed 57

## A/B Variants

- Best named variant: inventory_target_tuned
- Best overall strategy: execution_grid_search
- Best return: 0.024% vs control -0.000%
- Best Sharpe: 2.558 vs control -0.020
- Note: Main A/B figure includes named variants plus the strongest execution-grid and signal-search candidates from the full rerun.

## Adverse-Selection Sweep

- Baseline Sharpe: -0.020
- Best candidate Sharpe: 1.693
- Best params: {"high_thresh": 0.65, "size_mult_strength": 0.3, "spread_mult_strength": 0.25}

## Ablation Study

- Baseline direction AUC: 0.806
- Baseline volatility R2: -0.187
- Comparisons captured: 9
- Best direction AUC: three_class_direction (0.8156)
- Best volatility R2: lightgbm (0.4409)
- Quantile coverage at q=0.75: 0.7773

