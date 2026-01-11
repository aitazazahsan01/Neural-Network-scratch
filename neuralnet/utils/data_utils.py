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
