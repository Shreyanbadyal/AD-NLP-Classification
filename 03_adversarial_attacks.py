"""
Step 3: Run adversarial attacks against the baseline model.

Implements three attack strategies:
  1. TextFooler — word-level synonym substitution
  2. DeepWordBug — character-level perturbations
  3. BERT-Attack — context-aware word replacement using BERT MLM

Each attack is evaluated on a subset of the test set, and results
(success rate, perturbation rate, examples) are saved for analysis.
"""

import os
import sys
import json

from textattack.datasets import HuggingFaceDataset
from datasets import load_dataset

from config import (
    BASELINE_MODEL_PATH, RESULTS_DIR, NUM_ATTACK_SAMPLES,
)
from utils.model_utils import get_model_wrapper
from utils.attack_utils import build_attack, run_attack_evaluation, save_attack_results


def main():
    print("=" * 60)
    print("STEP 3: Adversarial Attacks on Baseline Model")
    print("=" * 60)

    # Load model wrapper
    model_wrapper = get_model_wrapper(BASELINE_MODEL_PATH)

    # Load dataset in TextAttack format
    print("Loading AG News test set for attacks...")
    dataset = HuggingFaceDataset("ag_news", split="test")

    # Define attacks to run
    # Skip deepwordbug if results already exist
    attack_results_path = os.path.join(RESULTS_DIR, "attack_results.json")
    existing_results = {}
    if os.path.exists(attack_results_path):
        with open(attack_results_path) as f:
            existing_results = json.load(f)

    attack_names = ["textfooler", "deepwordbug"]
    all_results = {}

    for attack_name in attack_names:
        # Skip if we already have successful results for this attack
        if attack_name in existing_results and "error" not in existing_results[attack_name]:
            print(f"\nSkipping {attack_name} — results already exist")
            all_results[attack_name] = existing_results[attack_name]
            continue

        try:
            # Build attack
            attack = build_attack(model_wrapper, attack_name)

            # Run attack
            metrics = run_attack_evaluation(
                attack=attack,
                dataset=dataset,
                num_examples=NUM_ATTACK_SAMPLES,
                attack_name=attack_name,
            )

            all_results[attack_name] = metrics

        except Exception as e:
            print(f"\nError running {attack_name}: {e}")
            all_results[attack_name] = {"error": str(e)}

    # Save all results
    save_attack_results(all_results, "attack_results.json")

    # Print summary table
    print("\n" + "=" * 60)
    print("ATTACK SUMMARY")
    print("=" * 60)
    print(f"{'Attack':<15} {'Success%':>10} {'Perturb%':>10} {'Queries':>10} {'Time(s)':>10}")
    print("-" * 55)
    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"{name:<15} {'ERROR':>10}")
        else:
            print(
                f"{name:<15} "
                f"{metrics['attack_success_rate']:>9.1f}% "
                f"{metrics['avg_word_perturbation_rate']:>9.1f}% "
                f"{metrics['avg_queries_per_attack']:>10.1f} "
                f"{metrics['elapsed_seconds']:>10.1f}"
            )

    # Print example adversarial perturbations
    print("\n" + "=" * 60)
    print("EXAMPLE ADVERSARIAL PERTURBATIONS")
    print("=" * 60)
    for name, metrics in all_results.items():
        if "error" in metrics or not metrics.get("examples"):
            continue
        print(f"\n--- {name.upper()} ---")
        for i, ex in enumerate(metrics["examples"][:3]):
            print(f"\n  Example {i+1}:")
            print(f"  Original ({ex['original_label']}): {ex['original']}")
            print(f"  Perturbed ({ex['perturbed_label']}): {ex['perturbed']}")

    print("\nDone! Attack results saved to results/attack_results.json")


if __name__ == "__main__":
    main()
