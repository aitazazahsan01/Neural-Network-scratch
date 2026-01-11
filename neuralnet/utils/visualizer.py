"""
neuralnet/utils/visualizer.py
==============================
Training visualisation utilities using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_history(
    history: dict,
    title: str = "Training History",
    save_path: str | None = None,
) -> None:
    """Plot training and validation loss and accuracy curves.

    Parameters
    ----------
    history : dict
        Returned by ``model.fit()``.
        Keys: "train_loss", "val_loss", "train_acc", "val_acc".
