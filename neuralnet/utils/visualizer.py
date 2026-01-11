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
    title : str
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    has_val = any(v is not None for v in history.get("val_loss", [None]))
    epochs  = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Loss subplot ---
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    if has_val:
        val_losses = [v for v in history["val_loss"] if v is not None]
        ax.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label="Val Loss",
            color="#F44336",
            linewidth=2,
            linestyle="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log") if min(history["train_loss"]) > 0 else None

    # --- Accuracy subplot ---
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc", color="#4CAF50", linewidth=2)
    if has_val:
        val_accs = [v for v in history["val_acc"] if v is not None]
        ax.plot(
            range(1, len(val_accs) + 1),
            val_accs,
            label="Val Acc",
            color="#FF9800",
            linewidth=2,
            linestyle="--",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over Epochs")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
