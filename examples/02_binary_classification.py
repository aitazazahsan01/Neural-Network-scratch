"""
examples/02_binary_classification.py
=====================================
Demonstration: binary classification on a synthetic 2-class dataset.

WHAT THIS EXAMPLE TEACHES
--------------------------
1. How to preprocess real (synthetic) data: standardise features.
2. How Sigmoid + BCE work together for binary classification.
3. How to visualise a 2-D decision boundary evolving during training.
4. Overfitting: how a model with too much capacity fits noise.
5. Dropout's effect on overfitting.
6. Comparison of all four optimizers: SGD → Momentum → RMSProp → Adam.

DATASET
-------
sklearn's make_classification with 2 features (so we can plot it):
    • 1000 samples, 2 informative features, 2 classes
    • Some overlap between classes (not perfectly separable)

ARCHITECTURE
------------
Input (2) → Dense(16, ReLU) → Dropout(0.3) → Dense(8, ReLU) → Dense(1, Sigmoid)
Loss: Binary Cross-Entropy
Optimizer: Adam (lr=0.01)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from neuralnet import Sequential, DenseLayer, DropoutLayer
from neuralnet.utils.data_utils import train_val_split, standardize
from neuralnet.utils.metrics import classification_report_nn
from neuralnet.utils.visualizer import plot_history, plot_decision_boundary, plot_weight_histograms
from neuralnet import SGD, SGDMomentum, RMSProp, Adam


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(0)


# ---------------------------------------------------------------------------
# 1. Generate Dataset
# ---------------------------------------------------------------------------

print("=" * 55)
print("Binary Classification — Synthetic 2-Class Dataset")
print("=" * 55)

X_raw, y_raw = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42,
)
y_raw = y_raw.reshape(-1, 1).astype(float)

print(f"\nDataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features, 2 classes")
print(f"Class distribution: {int(y_raw.sum())} positives, {int(len(y_raw)-y_raw.sum())} negatives")


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------

X_train_raw, X_val_raw, y_train, y_val = train_val_split(
