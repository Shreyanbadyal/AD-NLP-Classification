"""
Model utilities and TextAttack-compatible wrapper.
"""

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import textattack

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, NUM_LABELS, MAX_SEQ_LENGTH, DEVICE


class HuggingFaceModelWrapper(textattack.models.wrappers.ModelWrapper):
    """
    Custom TextAttack model wrapper for HuggingFace models.
    This allows TextAttack to interface with our fine-tuned model.
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or DEVICE
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text_input_list):
        """
        Accept a list of strings, return prediction logits as numpy array.
        """
        encodings = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits

        return logits.cpu().numpy()


def load_trained_model(model_path, device=None):
    """
    Load a fine-tuned model and tokenizer from disk.

    Returns:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
    """
    if device is None:
        device = DEVICE

    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def get_model_wrapper(model_path, device=None):
    """
    Load model and return a TextAttack-compatible wrapper.
    """
    model, tokenizer = load_trained_model(model_path, device)
    return HuggingFaceModelWrapper(model, tokenizer, device)
