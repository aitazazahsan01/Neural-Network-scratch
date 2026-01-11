"""
neuralnet/tensor.py
===================
Thin NumPy wrapper providing consistent dtype management and shape utilities
used throughout the library.

WHY THIS MODULE EXISTS
----------------------
NumPy operations sometimes silently change dtypes (e.g., integer division
producing int64 arrays, or mixing float32/float64 mid-computation). All
neural-network math here runs in float64 for numerical precision. This
module ensures every array that enters our library is cast once, and only
once, to float64.

The helpers below also make the rest of the code more self-documenting:
calling `ensure_2d(X)` is more expressive than writing `X.reshape(-1, 1)`.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DTYPE = np.float64          # single source of truth for precision
EPSILON = 1e-8              # small constant used to avoid log(0) or /0


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def to_array(x: object, dtype: type = DTYPE) -> np.ndarray:
    """Convert *x* to a NumPy array with the library default dtype.

    Parameters
    ----------
    x : array-like
        Any object that ``np.array()`` can convert (list, tuple, np.ndarray …)
    dtype : numpy dtype, optional
        Target dtype. Defaults to float64.

    Returns
    -------
    np.ndarray
    """
    return np.array(x, dtype=dtype)


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Guarantee that *x* is at least 2-dimensional.

    A 1-D array of shape (n,) is reshaped to a column vector (n, 1).
    Arrays that are already ≥ 2-D are returned unchanged.

    This prevents broadcasting surprises in matrix multiplications where
    the difference between shape (n,) and shape (n, 1) matters.
    """
    x = to_array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def ensure_column(x: np.ndarray) -> np.ndarray:
    """Reshape a 1-D array to a column vector (n, 1); 2-D unchanged."""
    x = to_array(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def clip(x: np.ndarray, min_val: float = EPSILON, max_val: float = 1.0 - EPSILON) -> np.ndarray:
    """Element-wise clip to [min_val, max_val].

    Used to prevent log(0) inside loss functions.
    """
    return np.clip(x, min_val, max_val)


def batch_size(X: np.ndarray) -> int:
    """Return the number of samples in *X* (first dimension)."""
    return X.shape[0]


def fan_in_fan_out(shape: tuple) -> tuple:
    """Compute fan_in and fan_out from a weight matrix shape (n_in, n_out).

    These values drive variance-based weight initialisation (Xavier, He).

    fan_in  = number of input connections to a single neuron
    fan_out = number of output connections from a single neuron
    """
    assert len(shape) == 2, "Weight matrix must be 2-D (n_in, n_out)."
    return shape[0], shape[1]
