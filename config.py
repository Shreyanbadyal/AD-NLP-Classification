"""
Central configuration for the Adversarial NLP project.
All hyperparameters and paths are defined here for easy experimentation.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Create directories if they don't exist
for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_SEQ_LENGTH = 256

# ─── AG News Labels ─────────────────────────────────────────────────────────
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# ─── Training Hyperparameters ────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_TRAIN_SAMPLES = 8000    # Use subset for faster training (set None for full)
MAX_EVAL_SAMPLES = 1000     # Use subset for faster evaluation

# ─── Attack Configuration ───────────────────────────────────────────────────
NUM_ATTACK_SAMPLES = 50     # Number of samples to attack (attacks are slow)
ATTACK_MAX_WORDS_PERTURBED = 0.3  # Max fraction of words that can be changed

# ─── Adversarial Training ───────────────────────────────────────────────────
ADV_TRAIN_SAMPLES = 200     # Number of adversarial examples to generate for training
ADV_TRAIN_EPOCHS = 2        # Additional epochs for adversarial fine-tuning

# ─── Saved Model Paths ──────────────────────────────────────────────────────
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_distilbert")
ADV_TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "adv_trained_distilbert")
