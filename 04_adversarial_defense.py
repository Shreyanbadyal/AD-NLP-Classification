"""
Step 4: Implement and evaluate defense strategies.

Defense strategies:
  1. Adversarial Training — augment training data with adversarial examples
  2. Input Preprocessing — spelling correction as a defense layer
  3. Ensemble Voting — combine multiple model predictions

Each defense is evaluated against the same attacks to measure improvement.
"""

import os
import sys
import json
import time
import copy
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from textattack.datasets import HuggingFaceDataset

from config import (
    MODEL_NAME, NUM_LABELS, DEVICE, LABEL_NAMES,
    LEARNING_RATE, WEIGHT_DECAY, MAX_SEQ_LENGTH,
    BASELINE_MODEL_PATH, ADV_TRAINED_MODEL_PATH, RESULTS_DIR,
    ADV_TRAIN_SAMPLES, ADV_TRAIN_EPOCHS, NUM_ATTACK_SAMPLES,
)
from utils.model_utils import load_trained_model, HuggingFaceModelWrapper
from utils.attack_utils import build_attack, run_attack_evaluation, save_attack_results

# ─── Defense 1: Adversarial Training ────────────────────────────────────────

def generate_adversarial_examples(model_wrapper, num_examples):
    """Generate adversarial examples using TextFooler for training augmentation."""
    print(f"\nGenerating {num_examples} adversarial examples for training...")

    attack = build_attack(model_wrapper, "textfooler")
    dataset = HuggingFaceDataset("ag_news", split="train")

    from textattack.attacker import Attacker, AttackArgs
    import textattack

    attack_args = AttackArgs(
        num_examples=num_examples,
        disable_stdout=True,
        random_seed=42,
    )

    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()

    adv_texts = []
    adv_labels = []

    for r in results:
        if isinstance(r, textattack.attack_results.SuccessfulAttackResult):
            # Use the perturbed text with the ORIGINAL (correct) label
            adv_texts.append(r.perturbed_text())
            adv_labels.append(r.original_result.ground_truth_output)
        elif isinstance(r, textattack.attack_results.FailedAttackResult):
            # Attack failed — model was robust, still use original
            adv_texts.append(r.original_text())
            adv_labels.append(r.original_result.ground_truth_output)

    print(f"  Generated {len(adv_texts)} adversarial training examples")
    return adv_texts, adv_labels


def adversarial_training(model_path, adv_texts, adv_labels, save_path):
    """Fine-tune model on adversarial examples."""
    print(f"\nAdversarial fine-tuning for {ADV_TRAIN_EPOCHS} epochs...")

    model, tokenizer = load_trained_model(model_path)
    model.train()

    # Tokenize adversarial examples
    encodings = tokenizer(
        adv_texts,
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(adv_labels, dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE / 2, weight_decay=WEIGHT_DECAY)
    total_steps = len(dataloader) * ADV_TRAIN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(ADV_TRAIN_EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Adv Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1} — Avg Loss: {total_loss/len(dataloader):.4f}")

    # Save adversarially trained model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Adversarially trained model saved to {save_path}")
    return model, tokenizer


# ─── Defense 2: Input Preprocessing ─────────────────────────────────────────

class SpellingCorrectionDefense:
    """
    Defense that applies spelling correction to input text before classification.
    This can reverse character-level perturbations (e.g., DeepWordBug).
    """

    def __init__(self, model_wrapper):
        try:
            from autocorrect import Speller
            self.spell = Speller(lang="en")
        except ImportError:
            print("Warning: autocorrect not installed. Using identity defense.")
            self.spell = None
        self.model_wrapper = model_wrapper

    def correct_text(self, text):
        """Apply spelling correction."""
        if self.spell is None:
            return text
        return self.spell(text)

    def __call__(self, text_input_list):
        """Correct text then classify."""
        corrected = [self.correct_text(t) for t in text_input_list]
        return self.model_wrapper(corrected)


class PreprocessingModelWrapper(HuggingFaceModelWrapper):
    """Model wrapper with spelling correction preprocessing."""

    def __init__(self, model, tokenizer, device=None):
        super().__init__(model, tokenizer, device)
        try:
            from autocorrect import Speller
            self.spell = Speller(lang="en")
        except ImportError:
            self.spell = None

    def __call__(self, text_input_list):
        if self.spell:
            text_input_list = [self.spell(t) for t in text_input_list]
        return super().__call__(text_input_list)


# ─── Defense 3: Ensemble Voting ─────────────────────────────────────────────

class EnsembleModelWrapper(HuggingFaceModelWrapper):
    """
    Ensemble defense: average predictions from baseline + adversarially trained model.
    """

    def __init__(self, model1, model2, tokenizer, device=None):
        super().__init__(model1, tokenizer, device)
        self.model2 = model2
        self.model2.to(self.device)
        self.model2.eval()

    def __call__(self, text_input_list):
        encodings = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits1 = self.model(**encodings).logits
            logits2 = self.model2(**encodings).logits
            # Average logits from both models
            avg_logits = (logits1 + logits2) / 2

        return avg_logits.cpu().numpy()


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def evaluate_defense(model_wrapper, attack_name, dataset, label):
    """Run a single attack against a defended model."""
    print(f"\n  Evaluating {label} against {attack_name}...")
    try:
        attack = build_attack(model_wrapper, attack_name)
        metrics = run_attack_evaluation(
            attack=attack,
            dataset=dataset,
            num_examples=NUM_ATTACK_SAMPLES // 2,  # Fewer samples for speed
            attack_name=f"{label}_{attack_name}",
        )
        return metrics
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("STEP 4: Adversarial Defense Strategies")
    print("=" * 60)

    dataset = HuggingFaceDataset("ag_news", split="test")
    all_defense_results = {}

    # ── Defense 1: Adversarial Training ──
    print("\n" + "─" * 40)
    print("DEFENSE 1: Adversarial Training")
    print("─" * 40)

    baseline_model, baseline_tokenizer = load_trained_model(BASELINE_MODEL_PATH)
    baseline_wrapper = HuggingFaceModelWrapper(baseline_model, baseline_tokenizer)

    # Generate adversarial examples
    adv_texts, adv_labels = generate_adversarial_examples(
        baseline_wrapper, ADV_TRAIN_SAMPLES
    )

    # Adversarial fine-tuning
    adv_model, adv_tokenizer = adversarial_training(
        BASELINE_MODEL_PATH, adv_texts, adv_labels, ADV_TRAINED_MODEL_PATH
    )
    adv_wrapper = HuggingFaceModelWrapper(adv_model, adv_tokenizer)

    # Evaluate adversarially trained model
    adv_train_results = {}
    for attack_name in ["textfooler", "deepwordbug"]:
        metrics = evaluate_defense(adv_wrapper, attack_name, dataset, "adv_trained")
        adv_train_results[attack_name] = metrics
    all_defense_results["adversarial_training"] = adv_train_results

    # ── Defense 2: Spelling Correction ──
    print("\n" + "─" * 40)
    print("DEFENSE 2: Input Preprocessing (Spelling Correction)")
    print("─" * 40)

    baseline_model2, baseline_tokenizer2 = load_trained_model(BASELINE_MODEL_PATH)
    preprocessing_wrapper = PreprocessingModelWrapper(
        baseline_model2, baseline_tokenizer2
    )

    preprocessing_results = {}
    for attack_name in ["textfooler", "deepwordbug"]:
        metrics = evaluate_defense(
            preprocessing_wrapper, attack_name, dataset, "spelling_defense"
        )
        preprocessing_results[attack_name] = metrics
    all_defense_results["spelling_correction"] = preprocessing_results

    # ── Defense 3: Ensemble ──
    print("\n" + "─" * 40)
    print("DEFENSE 3: Ensemble Voting")
    print("─" * 40)

    ensemble_wrapper = EnsembleModelWrapper(
        baseline_model2, adv_model, baseline_tokenizer2
    )

    ensemble_results = {}
    for attack_name in ["textfooler", "deepwordbug"]:
        metrics = evaluate_defense(
            ensemble_wrapper, attack_name, dataset, "ensemble"
        )
        ensemble_results[attack_name] = metrics
    all_defense_results["ensemble"] = ensemble_results

    # Save all defense results
    save_attack_results(all_defense_results, "defense_results.json")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("DEFENSE COMPARISON SUMMARY")
    print("=" * 60)

    # Load baseline attack results for comparison
    baseline_attack_path = os.path.join(RESULTS_DIR, "attack_results.json")
    if os.path.exists(baseline_attack_path):
        with open(baseline_attack_path) as f:
            baseline_attacks = json.load(f)
    else:
        baseline_attacks = {}

    print(f"\n{'Defense':<25} {'Attack':<15} {'Success Rate':>12} {'Δ from Base':>12}")
    print("-" * 65)

    for attack_name in ["textfooler", "deepwordbug"]:
        base_rate = baseline_attacks.get(attack_name, {}).get("attack_success_rate", "N/A")
        print(f"{'No Defense (Baseline)':<25} {attack_name:<15} {str(base_rate)+' %':>12} {'—':>12}")

        for defense_name, defense_results in all_defense_results.items():
            if attack_name in defense_results and "error" not in defense_results[attack_name]:
                rate = defense_results[attack_name]["attack_success_rate"]
                delta = ""
                if isinstance(base_rate, (int, float)):
                    delta = f"{rate - base_rate:+.1f}%"
                print(f"{defense_name:<25} {attack_name:<15} {str(rate)+' %':>12} {delta:>12}")
        print()

    print("Done! Defense results saved to results/defense_results.json")


if __name__ == "__main__":
    main()
