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


# ---------------------------------------------------------------------------
# 7. Decision boundary (standardised space → need to transform grid)
# ---------------------------------------------------------------------------

# Build full-scale dataset for boundary plot (use standardised coordinates)
X_all_raw = X_raw
y_all     = y_raw.ravel()

# Standardise full set with training statistics
X_all, _, _ = standardize(X_all_raw, X_ref=X_train_raw)

plot_decision_boundary(
    model, X_all, y_all,
    title="Decision Boundary — Binary Classification (Standardised Space)",
    resolution=300,
    save_path=os.path.join(os.path.dirname(__file__), "binary_boundary.png"),
)


# ---------------------------------------------------------------------------
# 8. Overfitting demonstration — model WITHOUT dropout vs WITH dropout
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("OVERFITTING EXPERIMENT")
print("Large model with no dropout vs with dropout")
print("=" * 55)

np.random.seed(42)

# Build a deliberately over-parameterised model (no dropout) to show overfitting
over_model = Sequential([
    DenseLayer(128, activation="relu",    name="Dense-1"),
    DenseLayer(128, activation="relu",    name="Dense-2"),
    DenseLayer(64,  activation="relu",    name="Dense-3"),
    DenseLayer(1,   activation="sigmoid", name="Output"),
])
over_model.compile(loss="bce", optimizer="adam", learning_rate=0.01)

np.random.seed(42)

# Regularised version (same arch but with Dropout)
reg_model = Sequential([
    DenseLayer(128, activation="relu",    name="Dense-1"),
    DropoutLayer(rate=0.5),
    DenseLayer(128, activation="relu",    name="Dense-2"),
    DropoutLayer(rate=0.5),
    DenseLayer(64,  activation="relu",    name="Dense-3"),
    DenseLayer(1,   activation="sigmoid", name="Output"),
])
reg_model.compile(loss="bce", optimizer="adam", learning_rate=0.01)

print("\nTraining overfit model (no dropout)...")
hist_over = over_model.fit(
    X_train, y_train,
    epochs=300, batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1, verbose_every=50,
)

print("\nTraining regularised model (with dropout)...")
hist_reg = reg_model.fit(
    X_train, y_train,
    epochs=300, batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1, verbose_every=50,
)

# Plot overfitting comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Overfitting: Large Model Without vs With Dropout", fontsize=13, fontweight="bold")

epochs_r = range(1, 301)

for ax, hist, label_prefix, color_train, color_val in [
    (axes[0], hist_over, "No Dropout", "#E53935", "#FF7043"),
    (axes[1], hist_reg,  "Dropout",    "#1565C0", "#0288D1"),
]:
    ax.plot(epochs_r, hist["train_loss"], color=color_train, linewidth=2,
            label=f"{label_prefix} Train Loss")
    ax.plot(epochs_r, hist["val_loss"],   color=color_val,   linewidth=2,
            linestyle="--", label=f"{label_prefix} Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title(label_prefix)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.2])

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "overfitting_comparison.png"), dpi=150)
plt.show()

gap_over  = min(hist_over["val_loss"]) - min(hist_over["train_loss"])
gap_reg   = min(hist_reg["val_loss"])  - min(hist_reg["train_loss"])
print(f"\nOverfit model  — train/val loss gap: {gap_over:.4f}")
print(f"Dropout model  — train/val loss gap: {gap_reg:.4f}")
print("Smaller gap = less overfitting. Dropout clearly helps!")


# ---------------------------------------------------------------------------
# 9. Optimizer comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 55)
print("OPTIMIZER COMPARISON: SGD → Momentum → RMSProp → Adam")
print("=" * 55)

def simple_model(optimizer):
    m = Sequential([
        DenseLayer(16, activation="relu",    name="Dense-1"),
        DenseLayer(8,  activation="relu",    name="Dense-2"),
        DenseLayer(1,  activation="sigmoid", name="Output"),
    ])
    m.compile(loss="bce", optimizer=optimizer)
    return m

optimizers = {
    "SGD (lr=0.05)":      SGD(lr=0.05),
    "Momentum (lr=0.01)": SGDMomentum(lr=0.01, beta=0.9),
    "RMSProp (lr=0.01)":  RMSProp(lr=0.01),
    "Adam (lr=0.01)":     Adam(lr=0.01),
}

histories = {}
for name, opt in optimizers.items():
    print(f"\n  Training with {name}...")
    np.random.seed(42)
    m = simple_model(opt)
    h = m.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    histories[name] = h
    final_loss = h["train_loss"][-1]
    final_acc  = h["train_acc"][-1]
    print(f"    Final loss: {final_loss:.4f}  |  Final acc: {final_acc:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#F44336", "#FF9800", "#4CAF50", "#2196F3"]
for (name, hist), color in zip(histories.items(), colors):
    ax.plot(hist["train_loss"], label=name, color=color, linewidth=2)

ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.set_title("Optimizer Comparison — Training Loss", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "optimizer_comparison.png"), dpi=150)
plt.show()

