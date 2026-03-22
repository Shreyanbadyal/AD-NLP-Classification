"""
Step 1: Train a baseline DistilBERT classifier on AG News dataset.

This script fine-tunes a pre-trained DistilBERT model for 4-class news
classification. The trained model is saved for subsequent attack evaluation.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from config import (
    MODEL_NAME, NUM_LABELS, DEVICE, LABEL_NAMES,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY,
    MAX_TRAIN_SAMPLES, MAX_EVAL_SAMPLES,
    BASELINE_MODEL_PATH, RESULTS_DIR,
)
from utils.data_utils import load_ag_news, tokenize_dataset, create_dataloader


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/total:.4f}",
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_NAMES,
        output_dict=True,
    )
    return avg_loss, accuracy, report, all_preds, all_labels


def main():
    print("=" * 60)
    print("STEP 1: Training Baseline DistilBERT on AG News")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE} | Batch: {TRAIN_BATCH_SIZE}")
    print()

    # Load data
    train_data, test_data = load_ag_news(MAX_TRAIN_SAMPLES, MAX_EVAL_SAMPLES)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_data, tokenizer)
    test_dataset = tokenize_dataset(test_data, tokenizer)

    train_loader = create_dataloader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = create_dataloader(test_dataset, EVAL_BATCH_SIZE)

    # Initialize model
    print(f"Loading pre-trained {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(DEVICE)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    training_history = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch
        )
        eval_loss, eval_acc, eval_report, _, _ = evaluate(model, test_loader, DEVICE)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "eval_loss": round(eval_loss, 4),
            "eval_accuracy": round(eval_acc, 4),
        }
        training_history.append(epoch_metrics)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Eval Loss:  {eval_loss:.4f} | Eval Acc:  {eval_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Final evaluation with full report
    print("\n--- Final Evaluation ---")
    _, final_acc, final_report, _, _ = evaluate(model, test_loader, DEVICE)
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(classification_report(
        *[[] for _ in range(2)],  # placeholder
    ) if False else "")

    # Print classification report nicely
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 55)
    for label in LABEL_NAMES:
        m = final_report[label]
        print(f"{label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")
    print("-" * 55)
    m = final_report["weighted avg"]
    print(f"{'Weighted Avg':<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")

    # Save model
    print(f"\nSaving model to {BASELINE_MODEL_PATH}...")
    model.save_pretrained(BASELINE_MODEL_PATH)
    tokenizer.save_pretrained(BASELINE_MODEL_PATH)

    # Save training metrics
    results = {
        "model": MODEL_NAME,
        "dataset": "ag_news",
        "num_labels": NUM_LABELS,
        "training_samples": len(train_data),
        "eval_samples": len(test_data),
        "final_accuracy": round(final_acc, 4),
        "training_history": training_history,
        "classification_report": final_report,
        "training_time_seconds": round(elapsed, 1),
    }
    results_path = os.path.join(RESULTS_DIR, "baseline_training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to {results_path}")
    print("\nDone! Baseline model is ready.")


if __name__ == "__main__":
    main()
