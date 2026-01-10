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
