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
