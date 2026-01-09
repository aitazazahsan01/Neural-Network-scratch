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
