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
