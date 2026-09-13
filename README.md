# Bank Deposit Prediction

Predicting whether a bank client will subscribe to a term deposit, using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing). Built for AMS 580 at Stony Brook University.

Classical ML models (Logistic Regression, Random Forest, KNN, XGBoost, LightGBM) are compared against a custom PyTorch tabular transformer (FT-Transformer style) with learned categorical embeddings. `duration` is dropped from the feature set to avoid target leakage, since call length is only known after the outcome is decided.

**Results:** Optuna-tuned LightGBM reached the highest balanced accuracy (73.6%), while Optuna-tuned XGBoost reached the highest test AUC (78.3%). The full writeup, including methodology and metric discussion, is in [report.pdf](report.pdf).

## Contents

- [`final.ipynb`](final.ipynb) — main notebook: preprocessing, classical models, evaluation
- [`colab_gpu_tabular_transformer.ipynb`](colab_gpu_tabular_transformer.ipynb) — transformer model, trained on GPU (Colab)
- [`testing.ipynb`](testing.ipynb) — exploratory / scratch notebook
- [`data/`](data/) — train/test splits
- [`transformer_outputs/`](transformer_outputs/) — trained transformer weights, predictions, metrics
- [`report.pdf`](report.pdf) / [`report.tex`](report.tex) — full paper

## Stack

Python, PyTorch, scikit-learn, XGBoost, LightGBM, Optuna, imbalanced-learn

## Team

Thien Huynh, Shane Patandin, Ishaan Reddy, Matthew Tranquada, Prerana Somani
