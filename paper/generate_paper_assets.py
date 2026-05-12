from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
FIGURE_DATA_DIR = PAPER_DIR / "figure_data"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def plot_ab_scatter_rows(rows: list[dict], output_png: Path, output_csv: Path, title: str) -> None:
    write_csv(
        output_csv,
        ["experiment", "category", "return_pct", "sharpe_ratio", "total_trades"],
        [
            [
                row.get("name", "unknown"),
                row.get("category", "unknown"),
                float(row.get("return_pct", 0.0)),
                float(row.get("sharpe_ratio", 0.0)),
                "" if row.get("total_trades") is None else int(row.get("total_trades", 0)),
            ]
            for row in rows
        ],
    )

    returns = [float(row.get("return_pct", 0.0)) for row in rows]
    sharpes = [float(row.get("sharpe_ratio", 0.0)) for row in rows]
    sizes = []
    colors = []
    for row in rows:
        trades = row.get("total_trades")
        if trades is None:
            sizes.append(180.0)
        else:
            sizes.append(float(np.clip(float(trades) * 1.8, 60, 500)))

        category = row.get("category")
        if category == "execution_grid_best":
            colors.append("#d62728")
        elif category == "signal_search_best":
            colors.append("#ff7f0e")
        elif category == "control":
            colors.append("#1f77b4")
        else:
            colors.append("#2ca02c")

    plt.figure(figsize=(10.5, 6))
    plt.scatter(returns, sharpes, s=sizes, color=colors, alpha=0.78)
    for row in rows:
        x_val = float(row.get("return_pct", 0.0))
        y_val = float(row.get("sharpe_ratio", 0.0))
        label = row.get("name", "unknown")
        plt.text(x_val, y_val, label, fontsize=8, ha="left", va="bottom")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    plt.axvline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    plt.xlabel("Return %")
    plt.ylabel("Sharpe Ratio")
    plt.title(title)
    plt.grid(alpha=0.25)
    save_figure(output_png)


def plot_walk_forward(data: dict) -> dict:
    folds = data.get("folds", [])
    fold_ids = [int(fold["fold"]) for fold in folds]
    aucs = [float(fold.get("auc", 0.0)) for fold in folds]
    sharpes = [float(fold.get("sharpe_ratio", 0.0)) for fold in folds]
    returns = [float(fold.get("total_return_pct", 0.0)) for fold in folds]

    write_csv(
        FIGURE_DATA_DIR / "walk_forward_summary.csv",
        ["fold", "auc", "accuracy", "sharpe_ratio", "total_return_pct", "total_trades"],
        [
            [
                int(fold.get("fold", 0)),
                float(fold.get("auc", 0.0)),
                float(fold.get("accuracy", 0.0)),
                float(fold.get("sharpe_ratio", 0.0)),
                float(fold.get("total_return_pct", 0.0)),
                int(fold.get("total_trades", 0)),
            ]
            for fold in folds
        ],
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(fold_ids, aucs, marker="o", linewidth=2.0, color="#1f77b4", label="AUC")
    axes[0].axhline(0.5, linestyle="--", linewidth=1.0, color="#888888", label="Chance")
    axes[0].set_ylabel("Direction AUC")
    axes[0].set_title("Purged Walk-Forward Validation")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    bars = axes[1].bar(fold_ids, sharpes, color=["#2ca02c" if val >= 0 else "#d62728" for val in sharpes])
    axes[1].plot(fold_ids, returns, marker="s", linewidth=1.5, color="#ff7f0e", label="Return %")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Sharpe / Return")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for bar, value in zip(bars, sharpes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    save_figure(FIGURES_DIR / "walk_forward_summary.png")
    return data.get("summary", {})


def plot_backtest(data: dict) -> dict:
    ml = data.get("ml_metrics", {})
    fixed = data.get("fixed_metrics", {})
    metrics = ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades"]
    labels = ["Return %", "Sharpe", "Max DD %", "Trades"]
    ml_vals = [float(ml.get(metric, 0.0)) for metric in metrics]
    fixed_vals = [float(fixed.get(metric, 0.0)) for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.36

    write_csv(
        FIGURE_DATA_DIR / "backtest_comparison.csv",
        ["metric", "display_label", "ml_value", "fixed_value"],
        [
            [metric, label, float(ml.get(metric, 0.0)), float(fixed.get(metric, 0.0))]
            for metric, label in zip(metrics, labels)
        ],
    )

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width / 2, ml_vals, width=width, label=ml.get("label", "ML"), color="#1f77b4")
    plt.bar(x + width / 2, fixed_vals, width=width, label=fixed.get("label", "Fixed"), color="#7f7f7f")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    plt.xticks(x, labels)
    plt.ylabel("Metric Value")
    plt.title("Main Backtest Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure(FIGURES_DIR / "backtest_comparison.png")
    return {
        "ml": ml,
        "fixed": fixed,
        "period": data.get("test_period", {}),
    }


def plot_ab_experiments(data: dict) -> dict:
    experiments = data.get("experiments", [])
    named_variant_rows = [
        {
            "name": row.get("name", "unknown"),
            "category": "control" if row.get("name") == "control_ml" else "named_variant",
            "return_pct": float(row.get("ml_metrics", {}).get("total_return_pct", 0.0)),
            "sharpe_ratio": float(row.get("ml_metrics", {}).get("sharpe_ratio", 0.0)),
            "total_trades": int(row.get("ml_metrics", {}).get("total_trades", 0)),
        }
        for row in experiments
    ]

    if named_variant_rows:
        plot_ab_scatter_rows(
            named_variant_rows,
            FIGURES_DIR / "ab_experiments_full_snapshot.png",
            FIGURE_DATA_DIR / "ab_experiments_full_snapshot.csv",
            "A/B Named Variants (Full Rerun)",
        )

    current_rows = list(named_variant_rows)
    exec_best = (data.get("strategy2_exec_grid") or {}).get("best_candidate") or {}
    signal_best = (data.get("strategy1_signal_search") or {}).get("best_candidate") or {}

    if exec_best:
        current_rows.append(
            {
                "name": "execution_grid_best",
                "category": "execution_grid_best",
                "return_pct": float(exec_best.get("ml_metrics", {}).get("total_return_pct", 0.0)),
                "sharpe_ratio": float(exec_best.get("ml_metrics", {}).get("sharpe_ratio", 0.0)),
                "total_trades": exec_best.get("ml_metrics", {}).get("total_trades"),
            }
        )

    if signal_best:
        current_rows.append(
            {
                "name": "signal_search_best",
                "category": "signal_search_best",
                "return_pct": float(signal_best.get("ml_metrics", {}).get("total_return_pct", 0.0)),
                "sharpe_ratio": float(signal_best.get("ml_metrics", {}).get("sharpe_ratio", 0.0)),
                "total_trades": signal_best.get("ml_metrics", {}).get("total_trades"),
            }
        )

    plot_ab_scatter_rows(
        current_rows,
        FIGURES_DIR / "ab_experiments.png",
        FIGURE_DATA_DIR / "ab_experiments.csv",
        "A/B Harness: Variants And Search Winners",
    )
    best_name = data.get("best_experiment")
    best_row = next((row for row in experiments if row.get("name") == best_name), None)
    best_live_config = data.get("best_live_config", {})
    return {
        "best_experiment": best_name,
        "best_metrics": best_live_config.get("metrics", {}),
        "best_named_variant_metrics": (best_row or {}).get("ml_metrics", {}),
        "control_metrics": data.get("control_metrics", {}),
        "fixed_metrics": data.get("fixed_metrics", {}),
        "best_live_config": best_live_config,
        "note": "Main A/B figure includes named variants plus the strongest execution-grid and signal-search candidates from the full rerun.",
    }


def plot_as_tuning(data: dict) -> dict:
    baseline = data.get("baseline_metrics", {})
    top_rows = data.get("top5_by_composite", [])
    labels = ["baseline"] + [
        f"ht={row['params']['high_thresh']}, s={row['params']['spread_mult_strength']}, z={row['params']['size_mult_strength']}"
        for row in top_rows
    ]
    sharpe = [float(baseline.get("sharpe_ratio", 0.0))] + [float(row.get("metrics", {}).get("sharpe_ratio", 0.0)) for row in top_rows]
    returns = [float(baseline.get("total_return_pct", 0.0))] + [float(row.get("metrics", {}).get("total_return_pct", 0.0)) for row in top_rows]

    x = np.arange(len(labels))
    width = 0.38

    write_csv(
        FIGURE_DATA_DIR / "as_tuning.csv",
        ["label", "sharpe_ratio", "total_return_pct"],
        [[labels[0], float(baseline.get("sharpe_ratio", 0.0)), float(baseline.get("total_return_pct", 0.0))]]
        + [
            [
                labels[idx + 1],
                float(row.get("metrics", {}).get("sharpe_ratio", 0.0)),
                float(row.get("metrics", {}).get("total_return_pct", 0.0)),
            ]
            for idx, row in enumerate(top_rows)
        ],
    )

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, sharpe, width=width, label="Sharpe", color="#9467bd")
    plt.bar(x + width / 2, returns, width=width, label="Return %", color="#ff9896")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="#888888")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Metric Value")
    plt.title("Adverse-Selection Tuning")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure(FIGURES_DIR / "as_tuning.png")
    return {
        "baseline": baseline,
        "best_params": data.get("best_params", {}),
        "best_metrics": data.get("best_metrics", {}),
    }


def plot_ablation(data: dict | None) -> dict:
    if not data:
        write_csv(FIGURE_DATA_DIR / "ablation_summary.csv", ["label", "direction_auc", "volatility_r2"], [])
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "Run experiments.py after enabling JSON export\nto generate the ablation chart.", ha="center", va="center")
        plt.axis("off")
        save_figure(FIGURES_DIR / "ablation_summary.png")
        return {}

    comparisons = [
        row for row in data.get("comparisons", [])
        if row.get("direction_auc") is not None and row.get("volatility_r2") is not None
    ]

    section_labels = []
    auc_values = []
    aux_values = []

    baseline = data.get("baseline", {})
    if baseline:
        section_labels.append("baseline")
        auc_values.append(float(baseline.get("direction_auc", 0.0)))
        aux_values.append(float(baseline.get("volatility_r2", 0.0)))

    for row in comparisons:
        section_labels.append(row.get("label", "experiment"))
        auc_values.append(float(row.get("direction_auc", 0.0)))
        aux_values.append(float(row.get("volatility_r2", 0.0)))

    x = np.arange(len(section_labels))
    width = 0.38

    write_csv(
        FIGURE_DATA_DIR / "ablation_summary.csv",
        ["label", "direction_auc", "volatility_r2"],
        [[label, auc, r2] for label, auc, r2 in zip(section_labels, auc_values, aux_values)],
    )

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, auc_values, width=width, label="Direction AUC", color="#1f77b4")
    plt.bar(x + width / 2, aux_values, width=width, label="Volatility R2", color="#17becf")
    plt.xticks(x, section_labels, rotation=25, ha="right")
    plt.title("Ablation Summary")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure(FIGURES_DIR / "ablation_summary.png")
    return data


def write_snapshot(summary: dict) -> None:
    lines = []
    lines.append("# Results Snapshot")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    walk = summary.get("walk_forward", {})
    lines.append("## Walk-Forward")
    lines.append("")
    if walk:
        lines.append(f"- Folds: {_fmt(walk.get('fold_count'), 0)}")
        lines.append(f"- Mean AUC: {_fmt(walk.get('mean_auc'))}")
        lines.append(f"- Mean Sharpe: {_fmt(walk.get('mean_sharpe'))}")
        lines.append(f"- Positive last 6 folds: {_fmt(walk.get('positive_sharpe_last_6'), 0)}")
    else:
        lines.append("- Missing walk-forward summary")
    lines.append("")

    backtest = summary.get("backtest", {})
    ml = backtest.get("ml", {})
    fixed = backtest.get("fixed", {})
    lines.append("## Main Backtest")
    lines.append("")
    if ml and fixed:
        lines.append(f"- ML return: {_fmt(ml.get('total_return_pct'))}% vs fixed {_fmt(fixed.get('total_return_pct'))}%")
        lines.append(f"- ML Sharpe: {_fmt(ml.get('sharpe_ratio'))} vs fixed {_fmt(fixed.get('sharpe_ratio'))}")
        lines.append(f"- ML max drawdown: {_fmt(ml.get('max_drawdown_pct'))}% vs fixed {_fmt(fixed.get('max_drawdown_pct'))}%")
        lines.append(f"- ML trades: {_fmt(ml.get('total_trades'), 0)} vs fixed {_fmt(fixed.get('total_trades'), 0)}")
    else:
        lines.append("- Missing main backtest results")
    lines.append("")

    ab = summary.get("ab", {})
    best = ab.get("best_metrics", {})
    control = ab.get("control_metrics", {})
    lines.append("## A/B Variants")
    lines.append("")
    if best and control:
        best_live = ab.get("best_live_config", {})
        if best_live:
            lines.append(f"- Best named variant: {ab.get('best_experiment', 'n/a')}")
            lines.append(f"- Best overall strategy: {best_live.get('strategy', 'n/a')}")
        else:
            lines.append(f"- Best experiment: {ab.get('best_experiment', 'n/a')}")
        lines.append(f"- Best return: {_fmt(best.get('total_return_pct'))}% vs control {_fmt(control.get('total_return_pct'))}%")
        lines.append(f"- Best Sharpe: {_fmt(best.get('sharpe_ratio'))} vs control {_fmt(control.get('sharpe_ratio'))}")
        if ab.get("note"):
            lines.append(f"- Note: {ab.get('note')}")
    else:
        lines.append("- Missing A/B experiment results")
    lines.append("")

    as_tuning = summary.get("as_tuning", {})
    as_best = as_tuning.get("best_metrics", {})
    as_base = as_tuning.get("baseline", {})
    lines.append("## Adverse-Selection Sweep")
    lines.append("")
    if as_best and as_base:
        lines.append(f"- Baseline Sharpe: {_fmt(as_base.get('sharpe_ratio'))}")
        lines.append(f"- Best candidate Sharpe: {_fmt(as_best.get('sharpe_ratio'))}")
        lines.append(f"- Best params: {json.dumps(as_tuning.get('best_params', {}), sort_keys=True)}")
    else:
        lines.append("- Missing adverse-selection sweep results")
    lines.append("")

    ablation = summary.get("ablation", {})
    lines.append("## Ablation Study")
    lines.append("")
    if ablation:
        baseline = ablation.get("baseline", {})
        lines.append(f"- Baseline direction AUC: {_fmt(baseline.get('direction_auc'))}")
        lines.append(f"- Baseline volatility R2: {_fmt(baseline.get('volatility_r2'))}")
        lines.append(f"- Comparisons captured: {_fmt(len(ablation.get('comparisons', [])), 0)}")
        for finding in ablation.get("headline_findings", []):
            lines.append(f"- {finding}")
    else:
        lines.append("- Ablation JSON not found yet. Rerun experiments after the JSON export patch.")
    lines.append("")

    target = PAPER_DIR / "results_snapshot.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_sources() -> None:
    lines = [
        "# Figure Sources",
        "",
        "Use this file when you need to cite a figure's data source directly in the paper or in Google Docs.",
        "",
        "## walk_forward_summary.png",
        "",
        "- Editable chart data: `paper/figure_data/walk_forward_summary.csv`",
        "- Underlying source file: `walk_forward_results.json`",
        "- Source fields: `folds[*].auc`, `folds[*].accuracy`, `folds[*].sharpe_ratio`, `folds[*].total_return_pct`, `folds[*].total_trades`",
        "",
        "## backtest_comparison.png",
        "",
        "- Editable chart data: `paper/figure_data/backtest_comparison.csv`",
        "- Underlying source file: `backtest_results.json`",
        "- Source fields: `ml_metrics` and `fixed_metrics` for return, Sharpe, drawdown, and trade count",
        "",
        "## ab_experiments.png",
        "",
        "- Editable chart data: `paper/figure_data/ab_experiments.csv`",
        "- Underlying source file: `backtest_ab_results.json`",
        "- Source fields: `experiments[*].ml_metrics`, `strategy2_exec_grid.best_candidate.ml_metrics`, `strategy1_signal_search.best_candidate.ml_metrics`",
        "- Scope: current full A/B rerun with named variants plus the strongest search winners.",
        "",
        "## ab_experiments_full_snapshot.png",
        "",
        "- Editable chart data: `paper/figure_data/ab_experiments_full_snapshot.csv`",
        "- Underlying source file: `backtest_ab_results.json`",
        "- Source fields: `experiments[*].ml_metrics`",
        "- Scope: current full-rerun named variants only.",
        "",
        "## as_tuning.png",
        "",
        "- Editable chart data: `paper/figure_data/as_tuning.csv`",
        "- Underlying source file: `as_tuning_results.json`",
        "- Source fields: `baseline_metrics` and `top5_by_composite[*].metrics`",
        "",
        "## ablation_summary.png",
        "",
        "- Editable chart data: `paper/figure_data/ablation_summary.csv`",
        "- Underlying source file: `experiments_results.json`",
        "- Source fields: `baseline.direction_auc`, `baseline.volatility_r2`, and selected rows from `comparisons`",
        "",
    ]
    (PAPER_DIR / "figure_sources.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    walk_forward = plot_walk_forward(load_json(ROOT / "walk_forward_results.json"))
    backtest = plot_backtest(load_json(ROOT / "backtest_results.json"))
    ab = plot_ab_experiments(load_json(ROOT / "backtest_ab_results.json"))
    as_tuning = plot_as_tuning(load_json(ROOT / "as_tuning_results.json"))

    ablation_path = ROOT / "experiments_results.json"
    ablation = plot_ablation(load_json(ablation_path) if ablation_path.exists() else None)

    summary = {
        "walk_forward": walk_forward,
        "backtest": backtest,
        "ab": ab,
        "as_tuning": as_tuning,
        "ablation": ablation,
    }
    write_snapshot(summary)
    write_figure_sources()
    print(f"Wrote figures to {FIGURES_DIR}")
    print(f"Wrote editable figure data to {FIGURE_DATA_DIR}")
    print(f"Wrote snapshot to {PAPER_DIR / 'results_snapshot.md'}")


if __name__ == "__main__":
    main()