"""
Step 5: Generate comprehensive visualizations and analysis.

Produces publication-quality charts comparing attacks and defenses:
  1. Baseline model performance (confusion matrix, per-class metrics)
  2. Attack success rate comparison
  3. Defense effectiveness comparison
  4. Perturbation analysis
  5. Training history curves
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.gridspec import GridSpec

from config import RESULTS_DIR, FIGURES_DIR, LABEL_NAMES

# Use a clean style
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
sns.set_theme(style="whitegrid", palette="muted")

COLORS = {
    "baseline": "#4C72B0",
    "textfooler": "#DD8452",
    "deepwordbug": "#55A868",
    "bertattack": "#C44E52",
    "adv_training": "#8172B3",
    "spelling": "#937860",
    "ensemble": "#DA8BC3",
}


def load_results():
    """Load all saved results."""
    results = {}

    files = {
        "training": "baseline_training_results.json",
        "evaluation": "baseline_evaluation.json",
        "attacks": "attack_results.json",
        "defenses": "defense_results.json",
    }

    for key, filename in files.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
            print(f"  Loaded {filename}")
        else:
            print(f"  Warning: {filename} not found")
            results[key] = None

    return results


def plot_training_curves(results):
    """Plot training loss and accuracy curves."""
    if not results.get("training"):
        return

    history = results["training"]["training_history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    eval_loss = [h["eval_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    eval_acc = [h["eval_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curves
    ax1.plot(epochs, train_loss, "o-", color=COLORS["baseline"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, eval_loss, "s--", color=COLORS["textfooler"], label="Eval Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Evaluation Loss")
    ax1.legend()
    ax1.set_xticks(epochs)

    # Accuracy curves
    ax2.plot(epochs, train_acc, "o-", color=COLORS["baseline"], label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, eval_acc, "s--", color=COLORS["textfooler"], label="Eval Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Evaluation Accuracy")
    ax2.legend()
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.8, 1.0)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_training_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_confusion_matrix(results):
    """Plot confusion matrix heatmap."""
    if not results.get("evaluation"):
        return

    cm = np.array(results["evaluation"]["confusion_matrix"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=ax1, cbar_kws={"shrink": 0.8},
    )
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_title("Confusion Matrix (Counts)")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=ax2, cbar_kws={"shrink": 0.8},
    )
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_per_class_metrics(results):
    """Plot per-class precision, recall, F1."""
    if not results.get("evaluation"):
        return

    report = results["evaluation"]["classification_report"]
    metrics_data = {
        "Precision": [report[l]["precision"] for l in LABEL_NAMES],
        "Recall": [report[l]["recall"] for l in LABEL_NAMES],
        "F1-Score": [report[l]["f1-score"] for l in LABEL_NAMES],
    }

    x = np.arange(len(LABEL_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics (Baseline Model)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylim(0.8, 1.05)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_per_class_metrics.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_attack_comparison(results):
    """Plot attack success rates and perturbation rates."""
    if not results.get("attacks"):
        return

    attacks = results["attacks"]
    attack_names = [k for k in attacks if "error" not in attacks[k]]

    if not attack_names:
        print("  No successful attack results to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Attack success rate
    success_rates = [attacks[a]["attack_success_rate"] for a in attack_names]
    colors = [COLORS.get(a, "#999999") for a in attack_names]
    bars = axes[0].bar(attack_names, success_rates, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Attack Success Rate (%)")
    axes[0].set_title("Attack Success Rate")
    axes[0].set_ylim(0, 105)

    # Perturbation rate
    perturb_rates = [attacks[a]["avg_word_perturbation_rate"] for a in attack_names]
    bars = axes[1].bar(attack_names, perturb_rates, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, perturb_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Avg Word Perturbation Rate (%)")
    axes[1].set_title("Perturbation Rate")

    # Queries per attack
    queries = [attacks[a]["avg_queries_per_attack"] for a in attack_names]
    bars = axes[2].bar(attack_names, queries, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, queries):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Avg Queries per Attack")
    axes[2].set_title("Query Efficiency")

    plt.suptitle("Adversarial Attack Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_attack_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_defense_comparison(results):
    """Plot defense effectiveness — before vs after defense."""
    attacks_data = results.get("attacks")
    defense_data = results.get("defenses")

    if not attacks_data or not defense_data:
        print("  Missing attack or defense results for comparison")
        return

    attack_types = ["textfooler", "deepwordbug"]
    defense_names = list(defense_data.keys())

    fig, axes = plt.subplots(1, len(attack_types), figsize=(7 * len(attack_types), 6))
    if len(attack_types) == 1:
        axes = [axes]

    for idx, attack_type in enumerate(attack_types):
        ax = axes[idx]

        # Baseline (no defense)
        labels = ["No Defense"]
        values = [attacks_data.get(attack_type, {}).get("attack_success_rate", 0)]
        bar_colors = [COLORS["baseline"]]

        # Each defense
        defense_color_map = {
            "adversarial_training": COLORS["adv_training"],
            "spelling_correction": COLORS["spelling"],
            "ensemble": COLORS["ensemble"],
        }

        for defense_name in defense_names:
            d = defense_data[defense_name]
            if attack_type in d and "error" not in d[attack_type]:
                rate = d[attack_type]["attack_success_rate"]
                clean_name = defense_name.replace("_", " ").title()
                labels.append(clean_name)
                values.append(rate)
                bar_colors.append(defense_color_map.get(defense_name, "#999999"))

        bars = ax.bar(labels, values, color=bar_colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title(f"Defense vs {attack_type.upper()}")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Defense Effectiveness Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "05_defense_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_accuracy_impact(results):
    """Plot model accuracy: clean vs under attack vs with defense."""
    attacks_data = results.get("attacks")
    defense_data = results.get("defenses")
    eval_data = results.get("evaluation")

    if not eval_data:
        return

    baseline_acc = eval_data["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Clean\n(No Attack)"]
    values = [baseline_acc]
    colors_list = [COLORS["baseline"]]

    # Under attack (accuracy = 100 - success_rate, roughly)
    if attacks_data:
        for attack_name in ["textfooler", "deepwordbug", "bertattack"]:
            if attack_name in attacks_data and "error" not in attacks_data[attack_name]:
                success_rate = attacks_data[attack_name]["attack_success_rate"]
                # Effective accuracy under attack
                effective_acc = baseline_acc * (1 - success_rate / 100)
                categories.append(f"Under\n{attack_name}")
                values.append(effective_acc)
                colors_list.append(COLORS.get(attack_name, "#999999"))

    # With defenses against TextFooler
    if defense_data and attacks_data and "textfooler" in attacks_data:
        base_success = attacks_data["textfooler"]["attack_success_rate"]
        for defense_name, d_results in defense_data.items():
            if "textfooler" in d_results and "error" not in d_results["textfooler"]:
                def_success = d_results["textfooler"]["attack_success_rate"]
                effective_acc = baseline_acc * (1 - def_success / 100)
                clean_name = defense_name.replace("_", " ").title()
                categories.append(f"{clean_name}\nvs TextFooler")
                values.append(effective_acc)
                colors_list.append("#2ecc71")

    bars = ax.bar(categories, values, color=colors_list, alpha=0.85, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Effective Accuracy (%)")
    ax.set_title("Model Accuracy: Clean vs Attacked vs Defended", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(y=baseline_acc, color="gray", linestyle="--", alpha=0.5, label="Baseline Accuracy")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_accuracy_impact.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def generate_summary_table(results):
    """Generate a text summary table of all results."""
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("COMPREHENSIVE RESULTS SUMMARY")
    summary_lines.append("=" * 70)

    if results.get("evaluation"):
        e = results["evaluation"]
        summary_lines.append(f"\nBaseline Model Performance:")
        summary_lines.append(f"  Accuracy: {e['accuracy']*100:.2f}%")
        summary_lines.append(f"  F1 (Macro): {e['f1_macro']*100:.2f}%")
        summary_lines.append(f"  F1 (Weighted): {e['f1_weighted']*100:.2f}%")

    if results.get("attacks"):
        summary_lines.append(f"\nAdversarial Attack Results:")
        summary_lines.append(f"  {'Attack':<15} {'Success%':>10} {'Perturb%':>10} {'Queries':>10}")
        summary_lines.append(f"  {'-'*45}")
        for name, m in results["attacks"].items():
            if "error" not in m:
                summary_lines.append(
                    f"  {name:<15} {m['attack_success_rate']:>9.1f}% "
                    f"{m['avg_word_perturbation_rate']:>9.1f}% "
                    f"{m['avg_queries_per_attack']:>10.1f}"
                )

    if results.get("defenses"):
        summary_lines.append(f"\nDefense Results (Attack Success Rate — lower is better):")
        for defense_name, d_results in results["defenses"].items():
            summary_lines.append(f"  {defense_name}:")
            for attack_name, m in d_results.items():
                if "error" not in m:
                    summary_lines.append(f"    vs {attack_name}: {m['attack_success_rate']:.1f}%")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(path, "w") as f:
        f.write(summary_text)
    print(f"\nSummary saved to {path}")


def main():
    print("=" * 60)
    print("STEP 5: Generating Visualizations")
    print("=" * 60)

    print("\nLoading results...")
    results = load_results()

    print("\nGenerating plots...")
    plot_training_curves(results)
    plot_confusion_matrix(results)
    plot_per_class_metrics(results)
    plot_attack_comparison(results)
    plot_defense_comparison(results)
    plot_accuracy_impact(results)

    print("\nGenerating summary...")
    generate_summary_table(results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Done! Project complete.")


if __name__ == "__main__":
    main()
