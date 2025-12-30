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
