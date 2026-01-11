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
    y_true : np.ndarray
        Ground truth.
        Binary:      shape (m,) or (m, 1), values 0 or 1.
        Multi-class: shape (m, K) one-hot OR (m,) integer labels.
    y_pred : np.ndarray
        Model outputs.
        Binary:      shape (m, 1) probabilities in [0,1].
        Multi-class: shape (m, K) probabilities (softmax outputs).

    Returns
    -------
    float in [0, 1]
    """
    # --- Derive predicted class labels ---
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        # Multi-class: argmax
        pred_labels = np.argmax(y_pred, axis=1)
    else:
        # Binary: threshold at 0.5
        pred_labels = (y_pred.ravel() >= 0.5).astype(int)

    # --- Derive true class labels ---
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        # One-hot encoded
        true_labels = np.argmax(y_true, axis=1)
    else:
        true_labels = y_true.ravel().astype(int)

    return float(np.mean(pred_labels == true_labels))


def confusion_matrix_nn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int | None = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Returns
    -------
    C : np.ndarray, shape (n_classes, n_classes)
        C[i, j] = number of samples with true class i predicted as class j.
    """
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        pred_labels = np.argmax(y_pred, axis=1)
    else:
        pred_labels = (y_pred.ravel() >= 0.5).astype(int)

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        true_labels = np.argmax(y_true, axis=1)
    else:
        true_labels = y_true.ravel().astype(int)

    if n_classes is None:
        n_classes = int(max(true_labels.max(), pred_labels.max())) + 1

    C = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        C[t, p] += 1
    return C


def classification_report_nn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list | None = None,
) -> None:
    """Print a classification report with precision, recall, and F1-score.

    METRIC DEFINITIONS
    ------------------
    Precision  = TP / (TP + FP)
        Of all samples predicted as class k, how many actually are class k?

    Recall     = TP / (TP + FN)
        Of all samples that are class k, how many did the model find?

    F1-score   = 2 · Precision · Recall / (Precision + Recall)
        Harmonic mean of precision and recall.
        High F1 requires *both* precision and recall to be high.

    Support    = TP + FN = actual number of class-k samples.
    """
    C = confusion_matrix_nn(y_true, y_pred)
    n_classes = C.shape[0]

    if class_names is None:
        class_names = [str(k) for k in range(n_classes)]

    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 54)

    for k in range(n_classes):
        tp = C[k, k]
        fp = C[:, k].sum() - tp
        fn = C[k, :].sum() - tp

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        support   = C[k, :].sum()

        print(
            f"{class_names[k]:<12} "
            f"{precision:>10.4f} "
            f"{recall:>10.4f} "
            f"{f1:>10.4f} "
            f"{support:>10}"
        )

    acc = C.diagonal().sum() / C.sum()
    print("-" * 54)
    print(f"{'Overall Accuracy':<12} {acc:>10.4f}")
    print()
