# Fresh Rerun Notes

This note supplements [results_snapshot.md](./results_snapshot.md) with the latest terminal-captured findings from the May 10 rerun.

## Completed Fresh Stages

- `train.py`
- `backtest.py`
- `backtest.py --as-sweep`
- `experiments.py`

## Walk-Forward And Main Backtest

- Purged walk-forward mean direction AUC: `0.7854`
- Purged walk-forward mean fold Sharpe: `-1.8387`
- Positive walk-forward folds: `2/6`
- Main backtest ML return: `-0.007%` vs fixed `-0.029%`
- Main backtest ML Sharpe: `-0.616` vs fixed `-0.529`
- Main backtest ML max drawdown: `-0.054%` vs fixed `-0.276%`

## Fresh A/B Harness Findings

Named variants from the fresh rerun:

- `control_ml`: return `-0.000%`, Sharpe `-0.020`, trades `52`
- `inventory_target_tuned`: return `0.002%`, Sharpe `0.156`, trades `54`
- `multilevel_profile`: return `-0.007%`, Sharpe `-0.616`, trades `79`
- `confidence_sizing`: return `-0.011%`, Sharpe `-0.935`, trades `52`
- `calibrated_probs`: return `-0.000%`, Sharpe `-0.020`, trades `52`
- `combined`: return `-0.043%`, Sharpe `-4.053`, trades `70`

Execution-grid search from the same rerun completed all `54` candidates before the exhaustive signal search began. Best execution-grid candidate:

- order levels: `2`
- level step: `0.08%`
- max order amount: `70`
- inventory skew strength: `0.15`
- return: `0.024%`
- Sharpe: `2.558`

## Incomplete Stage

The exhaustive A/B signal-parameter sweep started after the execution-grid search and reported `1440` candidates with an ETA of roughly `38 minutes`. It was stopped in this session to avoid blocking the rest of the paper pipeline.

Implication for the paper:

- Use the fresh named-variant and execution-grid findings above as the current rerun evidence.
- Treat the existing `backtest_ab_results.json` and the A/B figure in `paper/figures/ab_experiments.png` as the last fully completed exhaustive A/B snapshot, not the fresh partial rerun.