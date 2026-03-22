# Adversarial Attacks and Defenses on NLP Classification Models

## Overview
This project investigates the vulnerability of deep learning-based text classification models to adversarial attacks and evaluates defense mechanisms to improve robustness. We use the AG News dataset with a fine-tuned DistilBERT model as our baseline, then systematically attack it using state-of-the-art adversarial techniques and measure the effectiveness of various defense strategies.

## Dataset
**AG News** — A benchmark news classification dataset with 4 categories:
- World (0)
- Sports (1)
- Business (2)
- Science/Technology (3)

Training: 120,000 samples | Test: 7,600 samples

## Project Structure
```
adversarial-nlp-project/
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── 01_train_baseline.py       # Train baseline DistilBERT classifier
├── 02_evaluate_baseline.py    # Evaluate baseline model performance
├── 03_adversarial_attacks.py  # Run adversarial attacks (TextFooler, DeepWordBug, BERT-Attack)
├── 04_adversarial_defense.py  # Implement and evaluate defense strategies
├── 05_visualize_results.py    # Generate comparison charts and analysis
├── utils/
│   ├── __init__.py
│   ├── data_utils.py          # Dataset loading and preprocessing
│   ├── model_utils.py         # Model wrapper for TextAttack compatibility
│   └── attack_utils.py        # Attack helper functions
├── results/                   # Saved metrics and attack logs
├── figures/                   # Generated visualizations
├── models/                    # Saved model checkpoints
└── README.md
```

## Methodology

### Baseline Model
- **Architecture**: DistilBERT (distilbert-base-uncased) fine-tuned for 4-class classification
- **Training**: 3 epochs, AdamW optimizer, linear learning rate schedule

### Adversarial Attacks
1. **TextFooler** (Jin et al., 2020) — Word-level synonym substitution using counter-fitted embeddings
2. **DeepWordBug** (Gao et al., 2018) — Character-level perturbations (swap, substitute, delete, insert)
3. **BERT-Attack** (Li et al., 2020) — Context-aware word replacement using BERT masked language model

### Defense Strategies
1. **Adversarial Training** — Augment training data with adversarial examples and retrain
2. **Input Preprocessing Defense** — Spelling correction and text normalization as a preprocessing shield
3. **Ensemble Defense** — Combine predictions from multiple models for robustness

## Results
Results are generated after running the full pipeline. See `figures/` for visualizations and `results/` for detailed metrics.

## Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python 01_train_baseline.py
python 02_evaluate_baseline.py
python 03_adversarial_attacks.py
python 04_adversarial_defense.py
python 05_visualize_results.py
```

## References
- Jin, D., et al. "Is BERT Really Robust? A Strong Baseline for Natural Language Attack and Defense." AAAI 2020.
- Gao, J., et al. "Black-Box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers." IEEE S&P Workshop 2018.
- Li, L., et al. "BERT-ATTACK: Adversarial Attack Against BERT Using BERT." EMNLP 2020.
- Morris, J., et al. "TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP." EMNLP 2020 (Demo).

## Author
College Major Project — Adversarial NLP Research
