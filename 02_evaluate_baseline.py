"""
Step 2: Evaluate the baseline model in detail.

Generates per-class metrics, confusion matrix data, and confidence analysis.
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    DEVICE, LABEL_NAMES, EVAL_BATCH_SIZE,
    MAX_EVAL_SAMPLES, BASELINE_MODEL_PATH, RESULTS_DIR,
)
from utils.data_utils import load_ag_news, tokenize_dataset, create_dataloader
from utils.model_utils import load_trained_model


def compute_confidence_stats(model, dataloader, device):
    """Compute prediction confidence statistics."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    correct_confidences = []
    incorrect_confidences = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing confidence"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            max_probs = torch.max(probs, dim=-1).values

            for i in range(len(labels)):
                conf = max_probs[i].item()
                if preds[i] == labels[i]:
                    correct_confidences.append(conf)
                else:
                    incorrect_confidences.append(conf)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "correct_confidences": correct_confidences,
        "incorrect_confidences": incorrect_confidences,
    }


def main():
    print("=" * 60)
    print("STEP 2: Detailed Baseline Evaluation")
    print("=" * 60)

    # Load data
    _, test_data = load_ag_news(max_eval=MAX_EVAL_SAMPLES)
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_PATH)
    test_dataset = tokenize_dataset(test_data, tokenizer)
    test_loader = create_dataloader(test_dataset, EVAL_BATCH_SIZE)

    # Load model
    model, _ = load_trained_model(BASELINE_MODEL_PATH)

    # Compute detailed metrics
    stats = compute_confidence_stats(model, test_loader, DEVICE)
    preds = stats["predictions"]
    labels = stats["labels"]

    # Overall accuracy
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

    # Per-class report
    report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True)
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    for label in LABEL_NAMES:
        m = report[label]
        print(f"{label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"{'':>12}", end="")
    for name in LABEL_NAMES:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:>12}", end="")
        for j in range(len(LABEL_NAMES)):
            print(f"{cm[i][j]:>10}", end="")
        print()

    # Confidence analysis
    correct_conf = stats["correct_confidences"]
    incorrect_conf = stats["incorrect_confidences"]
    print(f"\nConfidence Analysis:")
    print(f"  Correct predictions   — Mean: {np.mean(correct_conf):.4f}, Std: {np.std(correct_conf):.4f}")
    if incorrect_conf:
        print(f"  Incorrect predictions — Mean: {np.mean(incorrect_conf):.4f}, Std: {np.std(incorrect_conf):.4f}")
    print(f"  Total correct: {len(correct_conf)} | Total incorrect: {len(incorrect_conf)}")

    # Save evaluation results
    eval_results = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confidence_stats": {
            "correct_mean": round(float(np.mean(correct_conf)), 4),
            "correct_std": round(float(np.std(correct_conf)), 4),
            "incorrect_mean": round(float(np.mean(incorrect_conf)), 4) if incorrect_conf else None,
            "incorrect_std": round(float(np.std(incorrect_conf)), 4) if incorrect_conf else None,
        },
    }

    results_path = os.path.join(RESULTS_DIR, "baseline_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to {results_path}")


if __name__ == "__main__":
    main()
