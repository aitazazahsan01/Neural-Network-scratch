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
        print(f"Saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Decision boundary (2-D input only)
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    resolution: int = 300,
    save_path: str | None = None,
) -> None:
    """Plot a colour-coded decision boundary for a 2-D input space.

    Parameters
    ----------
    model : Sequential (or any object with a .predict(X) method)
    X : np.ndarray, shape (m, 2)   — must be 2-D features
    y : np.ndarray, shape (m,) or (m, 1)   — class labels (int or 0/1)
    title : str
    resolution : int
        Grid resolution. Higher = smoother boundary but slower.
    save_path : str, optional
    """
    assert X.shape[1] == 2, "plot_decision_boundary requires exactly 2 features."

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid  = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)

    # Determine the predicted class label for colouring
    if probs.shape[1] > 1:
        Z = np.argmax(probs, axis=1)
    else:
        Z = (probs.ravel() >= 0.5).astype(int)

    Z = Z.reshape(xx.shape)

    # True labels
    if y.ndim == 2 and y.shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y.ravel().astype(int)

    n_classes = len(np.unique(y_labels))
    cmap_bg   = plt.cm.get_cmap("RdYlBu", n_classes)
    cmap_pts  = plt.cm.get_cmap("tab10", n_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.5)

    for k in range(n_classes):
        mask = y_labels == k
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=30, label=f"Class {k}",
            edgecolors="k", linewidths=0.4,
            color=cmap_pts(k),
        )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Weight histograms
# ---------------------------------------------------------------------------

def plot_weight_histograms(
    model,
    title: str = "Weight Distributions",
    save_path: str | None = None,
) -> None:
    """Plot histograms of weight values for each Dense layer.

    Useful for diagnosing:
        • Exploding weights  → very wide distribution
        • Dead weights       → distribution collapsed near zero
        • Healthy weights    → approximately Gaussian, moderate spread
    """
    from ..layers import DenseLayer

    dense_layers = [
        (i, l) for i, l in enumerate(model.layers)
        if isinstance(l, DenseLayer) and l._built
    ]

    if not dense_layers:
        print("No built Dense layers to plot.")
        return

    n = len(dense_layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (idx, layer) in zip(axes, dense_layers):
        weights = layer.W.ravel()
        ax.hist(weights, bins=40, color="#5C6BC0", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, label="zero")
        ax.set_title(
            f"Layer {idx + 1}  ({layer.n_in}→{layer.n_out})\n"
            f"σ={weights.std():.4f}  μ={weights.mean():.4f}"
        )
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
