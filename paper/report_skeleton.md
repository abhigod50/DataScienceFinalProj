# CS439 Final Project Report Skeleton

## Working Title

Regime-Aware Multi-Horizon ML for Short-Horizon Crypto Market Making

## Research Question

Do multi-horizon, regime-aware ML signals improve short-horizon crypto direction prediction and simulated market-making performance over simpler baselines?

## Abstract

This project studies whether a regime-aware, multi-horizon machine learning pipeline can improve short-horizon crypto market-making decisions relative to simpler baselines. The system combines engineered market, cross-asset, order book, and execution features with boosted-tree classifiers and regressors to predict near-term price direction and future volatility on 5-minute ETH candles. We evaluate the model stack with purged walk-forward validation, a simulated market-making backtest, targeted A/B strategy variants, and feature/model ablations. The current offline pipeline and figures are generated directly from this repository; update the quantitative claims in this section from [paper/results_snapshot.md](./results_snapshot.md) after each rerun.

## 1. Introduction

Write the motivation for short-horizon market making in crypto and why naive fixed-spread quoting is brittle under changing volatility and regime structure.

Points to cover:

- Crypto microstructure shifts quickly across calm, trending, and stressed regimes.
- A market maker needs both direction and volatility awareness, not just a static spread.
- The project goal is not pure directional alpha; it is better quoting and inventory control under realistic fee and fill assumptions.

## 2. Data And Environment

Describe the standalone submission folder as the full offline research environment.

Include:

- Primary data: ETH/USDT 5-minute candles from Binance US.
- Cross-asset context: BTC/USDT and SOL/USDT.
- Reference venue data: Coinbase ETH/USD.
- Cached microstructure inputs: order book and execution feature snapshots.
- Reproducible root: this repository contains training, backtesting, and analysis code without external live services.

## 3. Methods

### 3.1 Feature Engineering

Summarize the feature families:

- Price, return, momentum, volatility, and technical features.
- Cross-asset BTC and SOL context.
- Order book microstructure features.
- Execution-learning features.
- Regime features derived from the regime model.

### 3.2 Prediction Targets

Document the targets used by the pipeline:

- Binary direction at the main horizon.
- Multi-horizon direction labels for faster and mid horizons.
- Future volatility / price range regression.
- Optional 3-class direction framing for up / flat / down diagnostics.

### 3.3 Model Stack

Describe the model components trained in this repo:

- XGBoost direction model.
- LightGBM or XGBoost volatility model depending on runtime backend availability.
- Regime model.
- Neural model and meta-ensemble components when enabled.
- Probability calibration and confidence gating.

### 3.4 Validation And Backtest Design

Explain the evaluation layers:

- Purged walk-forward validation with embargo and Sharpe diagnostics.
- Point-in-time test split backtest against a fixed-spread baseline.
- A/B experiments for inventory targeting, multi-level quoting, and confidence sizing.
- Adverse-selection parameter sweep.
- Standalone feature/model ablations.

## 4. Experimental Setup

Report the command surface used from this folder:

```powershell
pwsh ./scripts/run_train.ps1
pwsh ./scripts/run_backtest.ps1
pwsh ./scripts/run_backtest_ab.ps1
python ./backtest.py --as-sweep
pwsh ./scripts/run_experiments.ps1
python ./paper/generate_paper_assets.py
```

Mention the managed interpreter selection if relevant:

- The PowerShell wrappers prefer a local `.venv`, then the parent `C:\freqtrade\.venv`.

## 5. Results

Use the generated figures and snapshot summary from this repo:

- Walk-forward diagnostics: [paper/figures/walk_forward_summary.png](./figures/walk_forward_summary.png)
- Baseline backtest comparison: [paper/figures/backtest_comparison.png](./figures/backtest_comparison.png)
- A/B experiment comparison: [paper/figures/ab_experiments.png](./figures/ab_experiments.png)
- Adverse-selection tuning: [paper/figures/as_tuning.png](./figures/as_tuning.png)
- Ablation study: [paper/figures/ablation_summary.png](./figures/ablation_summary.png)

### 5.1 Walk-Forward Validation

Summarize whether predictive quality is stable across folds and whether trading performance is consistent or regime-sensitive.

Questions to answer:

- Does AUC remain materially above 0.5 across folds?
- Are trading Sharpe ratios positive consistently, or only in selected windows?
- Which folds appear to align with favorable regimes?

### 5.2 Main Backtest

Compare the promoted ML strategy with the fixed-spread baseline.

Focus on:

- Total return.
- Sharpe ratio.
- Max drawdown.
- Trade count and inventory usage.

### 5.3 A/B Strategy Variants

Explain which execution changes matter most.

Candidate narrative:

- Confidence sizing may improve the risk/return tradeoff versus the control.
- The combined strategy may outperform the control on Sharpe and return.
- Multi-level quoting may improve turnover without necessarily improving Sharpe.

### 5.4 Adverse-Selection Sweep

State whether the AS filter improved or degraded the baseline policy on the current window. If the best AS candidate underperforms the baseline, say so directly.

### 5.5 Ablation Study

Use the ablation figure and JSON output to identify which architectural decisions are justified.

Look for:

- Feature pruning effects.
- SOL cross-asset contribution.
- XGBoost versus LightGBM.
- Multi-horizon choice.
- Ensemble behavior.

## 6. Discussion

Interpret the mismatch, if any, between strong predictive metrics and weak trading performance.

Possible themes:

- Good classification AUC does not guarantee profitable market making after fees and inventory constraints.
- Regime-dependent execution quality may dominate signal quality.
- Some strategy overlays help more than raw model changes.
- The backtest still depends on simplified fill and adverse-selection assumptions.

## 7. Limitations

- Offline simulation only; no live deployment evidence in this submission folder.
- Limited asset universe and a single primary trading pair for training and evaluation.
- Backtest outcomes depend on fill-probability and penalty modeling assumptions.
- Some experiments may be sensitive to test-window selection and class imbalance.

## 8. Conclusion

Write a short conclusion answering the research question with nuance. The likely structure is:

- The model stack improves predictive quality.
- Strategy overlays determine whether that predictive edge survives in trading metrics.
- The strongest paper claim should be tied to the best validated configuration, not the raw control model.

## Appendix A. Reproducibility

Mention:

- Repository root used for all experiments.
- Local cached datasets included in `data/`.
- Result artifacts written to the repository root JSON files.
- Figures and summary regenerated with `python ./paper/generate_paper_assets.py`.

## Appendix B. Current Snapshot Checklist

- Refresh the numbers from [paper/results_snapshot.md](./results_snapshot.md).
- Replace any placeholder wording with the latest quantitative results.
- Confirm that the figures correspond to the same rerun timestamp.