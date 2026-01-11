"""
neuralnet/utils/data_utils.py
==============================
Data preparation utilities: splitting, batching, encoding, normalisation.
"""

import numpy as np
from ..tensor import to_array, DTYPE


# ---------------------------------------------------------------------------
# Train / Validation Split
# ---------------------------------------------------------------------------

def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple:
    """Split arrays (X, y) into training and validation subsets.

    Parameters
    ----------
    X : np.ndarray, shape (m, ...)
    y : np.ndarray, shape (m, ...)
    val_fraction : float
        Fraction of samples to use for validation. Default 0.2 (20%).
    shuffle : bool
        Whether to shuffle before splitting.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    (X_train, X_val, y_train, y_val)
    """
    if seed is not None:
        np.random.seed(seed)

    m = X.shape[0]
    idx = np.random.permutation(m) if shuffle else np.arange(m)

    n_val = int(m * val_fraction)
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    return (
        to_array(X[train_idx]),
        to_array(X[val_idx]),
        to_array(y[train_idx]),
        to_array(y[val_idx]),
    )


# ---------------------------------------------------------------------------
# Mini-Batch Generator
# ---------------------------------------------------------------------------

class MiniBatchGenerator:
    """Iterate over (X, y) in randomised mini-batches.

    WHY MINI-BATCHES?
    -----------------
    • Full-batch GD: exact gradient but expensive for large datasets.
    • Stochastic (1 sample): cheap but very noisy.
    • Mini-batch: balances noise and efficiency.
      The noise actually helps escape sharp local minima.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    batch_size : int
