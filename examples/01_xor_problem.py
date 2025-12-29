"""
examples/01_xor_problem.py
==========================
Demonstration: solving the XOR problem with a neural network.

THE XOR PROBLEM — WHY IT MATTERS
----------------------------------
XOR (exclusive-OR) is historically significant in neural network research.
In 1969, Minsky & Papert proved that a *single-layer* (linear) perceptron
cannot solve XOR because XOR is not linearly separable.

The truth table:
    x1   x2   XOR
    ──   ──   ───
     0    0    0
     0    1    1
     1    0    1
     1    1    0

On a 2-D plot, no single straight line can separate the 1s from the 0s.
Only by adding a hidden layer (a non-linear transformation) can a network
learn the XOR function.

This example proves that backpropagation with non-linear activations
can solve problems that linear models fundamentally cannot.

WHAT TO OBSERVE
---------------
• With a hidden layer + ReLU/Tanh: loss converges to ~0, predictions ≈ correct.
• Without a hidden layer (Linear only): loss stalls near 0.25 — cannot learn XOR.
• Visualise the decision boundary to see the network "drawing" a non-linear border.

ARCHITECTURE
------------
Input (2) → Hidden (4, Tanh) → Output (1, Sigmoid)
Loss: BinaryCrossEntropy
Optimizer: Adam (lr=0.05)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from neuralnet import Sequential, DenseLayer
from neuralnet.utils.visualizer import plot_history, plot_decision_boundary


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# XOR truth table — all 4 combinations
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)  # shape (4, 1)

print("XOR Problem")
print("=" * 40)
print("Input X:")
print(X)
print("\nLabels y (XOR):")
print(y.ravel())


# ---------------------------------------------------------------------------
# Experiment 1: Linear model (single layer) — should FAIL
# ---------------------------------------------------------------------------

print("\n" + "=" * 40)
print("EXPERIMENT 1: Single-Layer (Linear) Model")
print("This CANNOT solve XOR — watch the loss stall!")
print("=" * 40)

model_linear = Sequential([
    DenseLayer(1, activation="sigmoid", name="Output"),
])
model_linear.compile(loss="bce", optimizer="adam", learning_rate=0.05)
history_linear = model_linear.fit(
    X, y,
    epochs=3000,
    batch_size=-1,   # full-batch (tiny dataset)
    verbose=1,
    verbose_every=500,
)

preds_linear = model_linear.predict(X)
print("\nLinear model predictions (should be wrong for XOR):")
for xi, pi, yi in zip(X, preds_linear.ravel(), y.ravel()):
    correct = "✓" if round(float(pi)) == int(yi) else "✗"
    print(f"  x={xi}  → ŷ={pi:.4f}  (true={int(yi)}) {correct}")


# ---------------------------------------------------------------------------
# Experiment 2: Two-layer network with hidden layer — should SUCCEED
# ---------------------------------------------------------------------------

print("\n" + "=" * 40)
print("EXPERIMENT 2: Two-Layer Network (Hidden Layer)")
print("This CAN solve XOR via non-linear transformation!")
print("=" * 40)

np.random.seed(42)  # reproducibility

model = Sequential([
    DenseLayer(4,  activation="tanh",    name="Hidden"),
    DenseLayer(1,  activation="sigmoid", name="Output"),
])
model.compile(loss="bce", optimizer="adam", learning_rate=0.05)
model.summary()

history = model.fit(
    X, y,
    epochs=3000,
    batch_size=-1,   # full-batch (tiny dataset)
    verbose=1,
    verbose_every=300,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print("\nFinal predictions vs ground truth:")
preds = model.predict(X)
all_correct = True
for xi, pi, yi in zip(X, preds.ravel(), y.ravel()):
    predicted_class = int(round(float(pi)))
    correct = "✓" if predicted_class == int(yi) else "✗"
    if predicted_class != int(yi):
        all_correct = False
    print(f"  x={xi}  → ŷ={pi:.4f}  rounded={predicted_class}  (true={int(yi)}) {correct}")

if all_correct:
    print("\n✅ XOR SOLVED! The network learned a non-linear decision boundary.")
else:
    print("\n⚠  Not all samples correct. Try running again or increasing epochs.")

final_loss = history["train_loss"][-1]
print(f"\nFinal train loss: {final_loss:.6f}")


# ---------------------------------------------------------------------------
# Weight inspection: what did the hidden layer learn?
# ---------------------------------------------------------------------------

print("\n" + "=" * 40)
print("WEIGHT INSPECTION — What did the hidden layer learn?")
print("=" * 40)
hidden = model.layers[0]
output = model.layers[1]

print(f"\nHidden layer weights W  (shape {hidden.W.shape}):")
print(np.round(hidden.W, 4))
print(f"\nHidden layer biases  b  (shape {hidden.b.shape}):")
print(np.round(hidden.b, 4))

print(f"\nOutput layer weights W  (shape {output.W.shape}):")
print(np.round(output.W, 4))
print(f"\nOutput layer bias    b  (shape {output.b.shape}):")
print(np.round(output.b, 4))

print("\n(The hidden layer transforms the non-linearly-separable XOR inputs")
print("into a higher-dimensional space where the output layer CAN draw a")
print("straight line to separate them — this is the key insight!)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Training curves comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XOR Problem — Training Comparison", fontsize=14, fontweight="bold")

epochs_range = range(1, 3001)

ax = axes[0]
ax.plot(history_linear["train_loss"], color="#F44336", linewidth=2, label="Linear (1 layer)")
ax.plot(history["train_loss"],        color="#4CAF50", linewidth=2, label="Network (2 layers)")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.set_title("Loss Convergence")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

ax = axes[1]
ax.plot(history_linear["train_acc"], color="#F44336", linewidth=2, label="Linear (1 layer)")
ax.plot(history["train_acc"],        color="#4CAF50", linewidth=2, label="Network (2 layers)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy")
ax.set_ylim([0, 1.05])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "xor_loss_comparison.png"), dpi=150)
plt.show()

# Decision boundary
plot_decision_boundary(
    model, X, y.ravel(),
    title="XOR Decision Boundary (Two-Layer Network)",
    resolution=400,
    save_path=os.path.join(os.path.dirname(__file__), "xor_boundary.png"),
)

