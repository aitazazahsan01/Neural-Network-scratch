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
        Samples per mini-batch.
    shuffle : bool
        Shuffle data at the start of each epoch.

    Usage
    -----
    >>> gen = MiniBatchGenerator(X_train, y_train, batch_size=32)
    >>> for X_batch, y_batch in gen:
    ...     ...
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        self.X = to_array(X)
        self.y = to_array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        m = self.X.shape[0]
        idx = np.random.permutation(m) if self.shuffle else np.arange(m)
        for start in range(0, m, self.batch_size):
            end     = min(start + self.batch_size, m)
            b_idx   = idx[start:end]
            yield self.X[b_idx], self.y[b_idx]

    def __len__(self) -> int:
        return int(np.ceil(self.X.shape[0] / self.batch_size))


# ---------------------------------------------------------------------------
# One-Hot Encoding
# ---------------------------------------------------------------------------

def one_hot_encode(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Convert integer class labels to one-hot encoded matrix.

    Parameters
    ----------
    y : np.ndarray of int, shape (m,)
        Integer labels in range [0, n_classes-1].
    n_classes : int or None
        Number of classes. Inferred from max(y)+1 if None.

    Returns
    -------
    Y : np.ndarray of float, shape (m, n_classes)
        One-hot encoded labels. Row i has a 1 in column y[i].

    Example
    -------
    >>> one_hot_encode(np.array([0, 2, 1]), n_classes=3)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    y = y.astype(int).ravel()
    if n_classes is None:
        n_classes = int(y.max()) + 1
    m = len(y)
    Y = np.zeros((m, n_classes), dtype=DTYPE)
    Y[np.arange(m), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Normalisation / Standardisation
# ---------------------------------------------------------------------------

def normalize(
    X: np.ndarray,
    X_ref: np.ndarray | None = None,
) -> tuple:
    """Min-max scale *X* to [0, 1] column-wise.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
    X_ref : np.ndarray, optional
        Reference array to compute statistics from (e.g. training set).
        If None, statistics are computed from X itself.

    Returns
    -------
    (X_scaled, x_min, x_max)
    """
    ref = X if X_ref is None else X_ref
    x_min = ref.min(axis=0)
    x_max = ref.max(axis=0)
    denom = np.where(x_max - x_min == 0, 1.0, x_max - x_min)  # avoid /0
    return ((X - x_min) / denom).astype(DTYPE), x_min, x_max


def standardize(
    X: np.ndarray,
    X_ref: np.ndarray | None = None,
) -> tuple:
    """Z-score standardise *X* to zero mean and unit variance column-wise.

    WHY STANDARDISE?
    ----------------
    If features have very different scales (e.g. age in [0,100] vs
    salary in [10000,200000]), gradient updates will be dominated by
    the large-scale features. Standardisation puts all features on the
    same scale, making training much more stable and faster.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
    X_ref : np.ndarray, optional
        Reference array for statistics.

    Returns
    -------
    (X_scaled, mean, std)
    """
    ref  = X if X_ref is None else X_ref
    mean = ref.mean(axis=0)
    std  = ref.std(axis=0)
    std  = np.where(std == 0, 1.0, std)   # avoid /0 for constant features
