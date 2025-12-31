"""
examples/03_mnist_digits.py
============================
Demonstration: handwritten digit recognition on the MNIST dataset.

WHAT THIS EXAMPLE TEACHES
--------------------------
1. Full multi-class classification pipeline.
2. Softmax output layer + Categorical Cross-Entropy loss.
3. One-hot encoding of class labels.
4. The fused SoftmaxCCE loss for numerical stability.
5. Adam optimizer for fast convergence on a real dataset.
6. Feature normalisation for image data (pixel scaling to [0,1]).
7. Per-class evaluation with a classification report.
8. How to inspect and visualise prediction confidence.

DATASET
-------
MNIST: 70,000 grayscale 28×28 images of handwritten digits (0–9).
We fetch a subset (up to 20,000 samples) via sklearn's fetch_openml.
Each image is flattened to a 784-dimensional vector.

WHY MULTI-CLASS NEEDS SOFTMAX
-------------------------------
Binary classification outputs a single probability p for one class.
Multi-class (K classes) needs a probability distribution over all K:
    [p(class=0), p(class=1), ..., p(class=K-1)]  with Σ = 1.

Softmax produces exactly this:
    P(class=k) = exp(z_k) / Σ_j exp(z_j)

ARCHITECTURE
------------
Input (784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(10, Linear)
Loss: SoftmaxCCE  (applies Softmax internally, more stable)
Optimizer: Adam (lr=0.001)
Expected accuracy: ~92–97% in 20 epochs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from neuralnet import Sequential, DenseLayer, DropoutLayer
from neuralnet.utils.data_utils import train_val_split, normalize, one_hot_encode
from neuralnet.utils.metrics import classification_report_nn, confusion_matrix_nn
from neuralnet.utils.visualizer import plot_history, plot_weight_histograms


# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------

print("=" * 60)
print("MNIST Digit Recognition")
print("=" * 60)
print("\nLoading MNIST via sklearn (may download on first run)...")

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_full = mnist.data.astype(float)
    y_full = mnist.target.astype(int)
    print(f"Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")
except Exception as e:
    print(f"Could not load MNIST: {e}")
    print("Generating random stand-in data for structure demonstration...")
    np.random.seed(0)
    X_full = np.random.rand(5000, 784) * 255
    y_full = np.random.randint(0, 10, 5000)


# ---------------------------------------------------------------------------
# 2. Subset & Preprocess
# ---------------------------------------------------------------------------

MAX_SAMPLES = 20_000       # use a subset to keep training fast
np.random.seed(42)
idx       = np.random.permutation(len(X_full))[:MAX_SAMPLES]
X_sub     = X_full[idx]
y_sub     = y_full[idx]

print(f"\nUsing {len(X_sub)} samples")
print(f"Class distribution: { {k: int((y_sub==k).sum()) for k in range(10)} }")

# Train/val split
X_train_raw, X_val_raw, y_train_int, y_val_int = train_val_split(
    X_sub, y_sub, val_fraction=0.15, seed=42
)

# Pixel normalisation: scale to [0, 1] — prevents large-magnitude inputs
# from producing huge pre-activations early in training.
X_train, x_min, x_max = normalize(X_train_raw)
X_val,   _,     _     = normalize(X_val_raw, X_ref=X_train_raw)

# One-hot encode labels
# Labels are integers [0..9] → one-hot matrices of shape (m, 10)
Y_train = one_hot_encode(y_train_int, n_classes=10)
Y_val   = one_hot_encode(y_val_int,   n_classes=10)

print(f"\nTraining set:   {X_train.shape[0]} samples  X:{X_train.shape}  Y:{Y_train.shape}")
print(f"Validation set: {X_val.shape[0]} samples  X:{X_val.shape}  Y:{Y_val.shape}")
print(f"\nPixel value range after normalisation: [{X_train.min():.2f}, {X_train.max():.2f}]")


# ---------------------------------------------------------------------------
# 3. Build Model
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

np.random.seed(42)

model = Sequential([
    # Hidden layer 1: 784 → 256 neurons, ReLU
    # He init is auto-selected because activation is ReLU
    DenseLayer(256, activation="relu", name="Dense-1"),
    DropoutLayer(rate=0.3, name="Dropout-1"),

    # Hidden layer 2: 256 → 128 neurons, ReLU
    DenseLayer(128, activation="relu", name="Dense-2"),
    DropoutLayer(rate=0.2, name="Dropout-2"),

    # Output layer: 128 → 10 neurons (one per digit class)
    # Using Linear activation because SoftmaxCCE applies Softmax internally.
    # This is numerically more stable than Softmax → CCE separately.
    DenseLayer(10, activation="linear", name="Output"),
])

model.compile(
    loss="softmax_cce",   # Softmax + Categorical Cross-Entropy (fused)
    optimizer="adam",
    learning_rate=0.001,
)

model.summary()

print("NOTE: Output layer uses 'linear' activation.")
print("      The SoftmaxCCE loss applies Softmax internally.")
print("      This is numerically more stable (avoids log(softmax(z))).")


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)
print(f"Epochs: 20  |  Batch size: 128  |  Optimizer: Adam\n")

history = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, Y_val),
    verbose=1,
    verbose_every=2,
)


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION ON VALIDATION SET")
print("=" * 60)

eval_result = model.evaluate(X_val, Y_val)

# Detailed per-class report
y_val_pred = model.predict(X_val)
classification_report_nn(
    Y_val, y_val_pred,
    class_names=[str(d) for d in range(10)],
)


# ---------------------------------------------------------------------------
# 6. Confusion Matrix
# ---------------------------------------------------------------------------

C = confusion_matrix_nn(Y_val, y_val_pred, n_classes=10)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(C, cmap="Blues", interpolation="nearest")
fig.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix — MNIST Validation Set", fontsize=13, fontweight="bold")

# Annotate cells
thresh = C.max() / 2.0
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(C[i, j]),
                ha="center", va="center",
                color="white" if C[i, j] > thresh else "black",
                fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "mnist_confusion.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 7. Training curves
# ---------------------------------------------------------------------------

plot_history(
    history,
    title="MNIST Training — Adam + Dropout (SoftmaxCCE)",
    save_path=os.path.join(os.path.dirname(__file__), "mnist_history.png"),
)


# ---------------------------------------------------------------------------
# 8. Visualise sample predictions
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

# Select 20 random validation samples
idx_sample = np.random.choice(len(X_val), 20, replace=False)
X_sample   = X_val[idx_sample]
y_sample   = y_val_int[idx_sample]

# Get probabilities (apply softmax to the linear output)
logits = model.predict(X_sample)
from scipy.special import softmax as scipy_softmax

# Apply softmax ourselves (our model outputs logits with SoftmaxCCE)
probs  = np.exp(logits - logits.max(axis=1, keepdims=True))
probs  /= probs.sum(axis=1, keepdims=True)

pred_classes = np.argmax(probs, axis=1)
confidence   = probs.max(axis=1)

fig, axes = plt.subplots(4, 5, figsize=(14, 11))
fig.suptitle("Sample Predictions — MNIST Validation Set\n(green=correct, red=wrong)", fontsize=13)

for ax, img, true_lbl, pred_lbl, conf in zip(
    axes.ravel(), X_sample, y_sample, pred_classes, confidence
):
    ax.imshow(img.reshape(28, 28), cmap="gray", interpolation="nearest")
    color = "green" if pred_lbl == true_lbl else "red"
    ax.set_title(f"Pred: {pred_lbl}  (conf={conf:.2f})\nTrue: {true_lbl}", color=color, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "mnist_predictions.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 9. Weight histograms (initialisation effect)
# ---------------------------------------------------------------------------

plot_weight_histograms(
    model,
    title="Weight Distributions After Training (He Init → ReLU layers)",
    save_path=os.path.join(os.path.dirname(__file__), "mnist_weights.png"),
)

print(f"\n✅ Final validation accuracy: {eval_result['accuracy']:.4f}")
print("\nDone! Saved plots in examples/")
