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
    X_raw, y_raw, val_fraction=0.2, seed=42
)

# Standardise features using training set statistics
# WHY? If feature scales differ greatly, gradient updates are dominated
# by the large-scale feature → training is slow and unstable.
X_train, mean, std = standardize(X_train_raw)
X_val, _, _        = standardize(X_val_raw, X_ref=X_train_raw)

print(f"\nTrain: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")
print(f"Feature means (should be ~0): {X_train.mean(axis=0).round(4)}")
print(f"Feature stds  (should be ~1): {X_train.std(axis=0).round(4)}")


# ---------------------------------------------------------------------------
# 3. Build Model
# ---------------------------------------------------------------------------

def build_model(with_dropout: bool = True) -> Sequential:
    """Build the classification model."""
    layers = [
        DenseLayer(16, activation="relu", name="Dense-1"),
    ]
    if with_dropout:
        layers.append(DropoutLayer(rate=0.3, name="Dropout"))
    layers += [
        DenseLayer(8, activation="relu",    name="Dense-2"),
        DenseLayer(1, activation="sigmoid", name="Output"),
    ]
    m = Sequential(layers)
    m.compile(loss="bce", optimizer="adam", learning_rate=0.01)
    return m


model = build_model(with_dropout=True)
model.summary()


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

print("Training with Adam + Dropout...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
    verbose_every=20,
)


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("EVALUATION ON VALIDATION SET")
print("=" * 55)
model.evaluate(X_val, y_val)

y_val_pred = model.predict(X_val)
classification_report_nn(y_val, y_val_pred, class_names=["Class 0", "Class 1"])


# ---------------------------------------------------------------------------
# 6. Visualise training curves
# ---------------------------------------------------------------------------

plot_history(
    history,
    title="Binary Classification — Adam + Dropout",
    save_path=os.path.join(os.path.dirname(__file__), "binary_history.png"),
)

