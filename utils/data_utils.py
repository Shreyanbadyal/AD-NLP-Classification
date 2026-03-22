"""
Data loading and preprocessing utilities for AG News dataset.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE


class TextClassificationDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized text classification data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_ag_news(max_train=None, max_eval=None):
    """
    Load AG News dataset from HuggingFace.

    Returns:
        train_data: list of dicts with 'text' and 'label'
        test_data: list of dicts with 'text' and 'label'
    """
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")

    train_data = dataset["train"]
    test_data = dataset["test"]

    if max_train and max_train < len(train_data):
        train_data = train_data.shuffle(seed=42).select(range(max_train))
        print(f"  Using {max_train} training samples (subset)")

    if max_eval and max_eval < len(test_data):
        test_data = test_data.shuffle(seed=42).select(range(max_eval))
        print(f"  Using {max_eval} evaluation samples (subset)")

    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data


def tokenize_dataset(data, tokenizer=None):
    """
    Tokenize a HuggingFace dataset for DistilBERT.

    Returns:
        TextClassificationDataset ready for DataLoader
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts = list(data["text"])
    labels = list(data["label"])

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    return TextClassificationDataset(encodings, labels)


def create_dataloader(dataset, batch_size=None, shuffle=False):
    """Create a PyTorch DataLoader from a TextClassificationDataset."""
    if batch_size is None:
        batch_size = EVAL_BATCH_SIZE

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
