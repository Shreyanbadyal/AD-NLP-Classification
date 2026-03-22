"""
Adversarial attack utilities — building attacks and running evaluations.
"""

import json
import os
import time
from datetime import datetime

import textattack
from textattack.attack_recipes import DeepWordBugGao2018
from textattack.datasets import HuggingFaceDataset
from textattack.attacker import Attacker, AttackArgs

# Import components for building custom attacks (avoids TensorFlow dependency)
from textattack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    WordSwapEmbedding,
    WordSwapMaskedLM,
)
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics.sentence_encoders import SBERT as BERTSimilarity

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, NUM_ATTACK_SAMPLES


def _build_textfooler(model_wrapper):
    """
    Build TextFooler attack using BERT sentence similarity instead of USE.
    Original: Jin et al., 2020 — word-level synonym substitution.
    """
    goal_function = UntargetedClassification(model_wrapper)
    transformation = WordSwapEmbedding(max_candidates=50)
    constraints = [
        RepeatModification(),
        StopwordModification(),
        MaxWordsPerturbed(max_percent=0.4),
        PartOfSpeech(allow_verb_noun_swap=True),
        BERTSimilarity(threshold=0.7, metric="cosine", compare_against_original=True),
    ]
    search_method = GreedyWordSwapWIR(wir_method="delete")
    return Attack(goal_function, constraints, transformation, search_method)


def _build_bertattack(model_wrapper):
    """
    Build BERT-Attack using BERT sentence similarity instead of USE.
    Original: Li et al., 2020 — context-aware word replacement via BERT MLM.
    """
    goal_function = UntargetedClassification(model_wrapper)
    transformation = WordSwapMaskedLM(
        method="bert-attack", max_candidates=48, min_confidence=5e-4
    )
    constraints = [
        RepeatModification(),
        StopwordModification(),
        MaxWordsPerturbed(max_percent=0.4),
        BERTSimilarity(threshold=0.2, metric="cosine", compare_against_original=True),
    ]
    search_method = GreedyWordSwapWIR(wir_method="unk")
    return Attack(goal_function, constraints, transformation, search_method)


ATTACK_BUILDERS = {
    "textfooler": _build_textfooler,
    "deepwordbug": lambda mw: DeepWordBugGao2018.build(mw),
    "bertattack": _build_bertattack,
}


def build_attack(model_wrapper, attack_name):
    """
    Build a TextAttack attack.

    Args:
        model_wrapper: TextAttack-compatible model wrapper
        attack_name: one of 'textfooler', 'deepwordbug', 'bertattack'

    Returns:
        textattack.Attack object
    """
    attack_name = attack_name.lower()
    if attack_name not in ATTACK_BUILDERS:
        raise ValueError(
            f"Unknown attack: {attack_name}. Choose from {list(ATTACK_BUILDERS.keys())}"
        )

    attack = ATTACK_BUILDERS[attack_name](model_wrapper)
    print(f"Built attack: {attack_name}")
    return attack


def run_attack_evaluation(attack, dataset, num_examples=None, attack_name="attack"):
    """
    Run an adversarial attack on a dataset and collect results.

    Args:
        attack: textattack.Attack object
        dataset: textattack.datasets.Dataset
        num_examples: number of examples to attack
        attack_name: name for logging

    Returns:
        dict with attack metrics
    """
    if num_examples is None:
        num_examples = NUM_ATTACK_SAMPLES

    print(f"\n{'='*60}")
    print(f"Running {attack_name} on {num_examples} examples...")
    print(f"{'='*60}")

    attack_args = AttackArgs(
        num_examples=num_examples,
        log_to_csv=os.path.join(RESULTS_DIR, f"{attack_name}_log.csv"),
        disable_stdout=False,
        random_seed=42,
    )

    attacker = Attacker(attack, dataset, attack_args)
    start_time = time.time()
    results = attacker.attack_dataset()
    elapsed = time.time() - start_time

    # Compute metrics
    total = len(results)
    successful = sum(
        1 for r in results if isinstance(r, textattack.attack_results.SuccessfulAttackResult)
    )
    failed = sum(
        1 for r in results if isinstance(r, textattack.attack_results.FailedAttackResult)
    )
    skipped = sum(
        1 for r in results if isinstance(r, textattack.attack_results.SkippedAttackResult)
    )

    # Compute average perturbation metrics for successful attacks
    avg_word_perturbed = 0
    avg_query_count = 0
    perturbed_examples = []

    for r in results:
        if isinstance(r, textattack.attack_results.SuccessfulAttackResult):
            avg_query_count += r.num_queries
            original_text = r.original_text()
            perturbed_text = r.perturbed_text()
            orig_words = original_text.split()
            pert_words = perturbed_text.split()
            diff_count = sum(1 for a, b in zip(orig_words, pert_words) if a != b)
            diff_count += abs(len(orig_words) - len(pert_words))
            avg_word_perturbed += diff_count / max(len(orig_words), 1)

            perturbed_examples.append({
                "original": original_text[:200],
                "perturbed": perturbed_text[:200],
                "original_label": r.original_result.ground_truth_output,
                "perturbed_label": r.perturbed_result.output,
            })

    if successful > 0:
        avg_word_perturbed /= successful
        avg_query_count /= successful

    metrics = {
        "attack_name": attack_name,
        "total_samples": total,
        "successful_attacks": successful,
        "failed_attacks": failed,
        "skipped": skipped,
        "attack_success_rate": round(successful / max(total - skipped, 1) * 100, 2),
        "avg_word_perturbation_rate": round(avg_word_perturbed * 100, 2),
        "avg_queries_per_attack": round(avg_query_count, 1),
        "elapsed_seconds": round(elapsed, 1),
        "examples": perturbed_examples[:10],  # Save first 10 examples
    }

    print(f"\n--- {attack_name} Results ---")
    print(f"  Attack Success Rate: {metrics['attack_success_rate']}%")
    print(f"  Successful: {successful} | Failed: {failed} | Skipped: {skipped}")
    print(f"  Avg Word Perturbation: {metrics['avg_word_perturbation_rate']}%")
    print(f"  Avg Queries: {metrics['avg_queries_per_attack']}")
    print(f"  Time: {metrics['elapsed_seconds']}s")

    return metrics


def save_attack_results(all_metrics, filename="attack_results.json"):
    """Save attack results to JSON."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Results saved to {filepath}")
