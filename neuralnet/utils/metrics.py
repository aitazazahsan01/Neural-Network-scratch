"""
neuralnet/utils/metrics.py
==========================
Evaluation metrics for classification.
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples.

    Works for both binary and multi-class outputs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth.
        Binary:      shape (m,) or (m, 1), values 0 or 1.
        Multi-class: shape (m, K) one-hot OR (m,) integer labels.
    y_pred : np.ndarray
        Model outputs.
        Binary:      shape (m, 1) probabilities in [0,1].
        Multi-class: shape (m, K) probabilities (softmax outputs).

    Returns
    -------
    float in [0, 1]
    """
    # --- Derive predicted class labels ---
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        # Multi-class: argmax
