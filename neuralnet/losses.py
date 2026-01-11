"""
neuralnet/losses.py
===================
Loss functions (objective functions / cost functions) and their exact gradients.

WHAT IS A LOSS FUNCTION?
-------------------------
The loss L(y, ŷ) measures how wrong the network's prediction ŷ is compared
to the ground truth y. It is a scalar. The entire training process is about
minimising L over the dataset.

Concretely:
    • After the forward pass we compute ŷ = network.forward(X).
    • We compute L = loss.compute(y, ŷ).
    • We need dL/dŷ to start backpropagation.

We average the loss across the m samples in a mini-batch:

    L = (1/m) Σ_{i=1}^{m}  ℓ(y_i, ŷ_i)

The gradient returned by `loss.gradient(y, ŷ)` is therefore also
scaled by 1/m, giving the *mean* gradient per sample.

Available loss functions
------------------------
* MSE               – regression; minimises squared error
* BinaryCrossEntropy – binary classification (sigmoid output)
* CategoricalCrossEntropy – multi-class (softmax output, one-hot labels)
* SoftmaxCCE        – numerically stable fusion of Softmax + CCE
                      (use this instead of Softmax + CCE separately)
"""

import numpy as np
from .tensor import clip, EPSILON, DTYPE


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Loss:
    """Abstract base for all loss functions."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.compute(y_true, y_pred)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


# ---------------------------------------------------------------------------
# Mean Squared Error — Regression
# ---------------------------------------------------------------------------

class MSE(Loss):
    """Mean Squared Error: L = (1/m) Σ (y - ŷ)².

    Use when the target is a continuous real value (regression).

    FORWARD (loss value)
    --------------------
        L = (1 / (2m)) Σ_{i=1}^{m} (y_i - ŷ_i)²

    We divide by 2 for cleaner gradient expressions (the 2 cancels).

    BACKWARD (gradient w.r.t. ŷ)
    ------------------------------
        dL/dŷ_i = (ŷ_i - y_i) / m

    Full derivation:
        dL/dŷ_k = d/dŷ_k [(1/2m) Σ (y_j - ŷ_j)²]
                = (1/2m) · 2(ŷ_k - y_k)
                = (ŷ_k - y_k) / m

    The gradient is *positive* when ŷ > y (prediction too high) and
    *negative* when ŷ < y (prediction too low). This makes intuitive
    sense: gradient descent will push ŷ down or up accordingly.
    """

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        return float(np.sum((y_true - y_pred) ** 2) / (2.0 * m))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]
        return ((y_pred - y_true) / m).astype(DTYPE)


# ---------------------------------------------------------------------------
# Binary Cross-Entropy — Binary Classification
# ---------------------------------------------------------------------------

class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy (Log Loss): for binary classifiers (sigmoid output).

    Each label y_i ∈ {0, 1}. The network outputs ŷ_i ∈ (0, 1) via sigmoid.

    FORWARD (loss value)
    --------------------
        L = -(1/m) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]

    Intuition:
        • If y=1: loss = -log(ŷ). High ŷ → low loss. Low ŷ → high loss.
        • If y=0: loss = -log(1-ŷ). Low ŷ → low loss. High ŷ → high loss.

    WHY CROSS-ENTROPY INSTEAD OF MSE FOR CLASSIFICATION?
    -----------------------------------------------------
    With MSE + Sigmoid, the loss landscape has many flat regions because
    σ'(z) ≈ 0 when |z| is large. Cross-entropy + Sigmoid eliminates this
    problem: the combined gradient dL/dZ = ŷ - y is clean and never
    vanishes due to saturation.

    BACKWARD (gradient w.r.t. ŷ)
    --------------------------------
        dL/dŷ = -(1/m) [y/ŷ - (1-y)/(1-ŷ)]
              = (1/m) [(ŷ - y) / (ŷ(1-ŷ))]

    When combined with sigmoid's backward (dZ = dA · ŷ(1-ŷ)),
    the denominator cancels and we get dL/dZ = (ŷ - y)/m — clean!

    Numerical stability: ŷ is clipped away from 0 and 1 to avoid log(0).
    """

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        y_pred_clipped = clip(y_pred)
        loss = -np.sum(
            y_true * np.log(y_pred_clipped) +
            (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        ) / m
        return float(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]
        y_pred_clipped = clip(y_pred)
        # dL/dŷ = (1/m) * [-(y/ŷ) + (1-y)/(1-ŷ)]
        grad = (
            -y_true / y_pred_clipped +
            (1.0 - y_true) / (1.0 - y_pred_clipped)
        ) / m
        return grad.astype(DTYPE)


# ---------------------------------------------------------------------------
# Categorical Cross-Entropy — Multi-class Classification
# ---------------------------------------------------------------------------

class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy for multi-class classification.

    Labels y_true are one-hot encoded: shape (m, K) where K = number of classes.
    Predictions y_pred are softmax outputs: shape (m, K), each row sums to 1.

    FORWARD (loss value)
    --------------------
        L = -(1/m) Σ_{i=1}^{m} Σ_{k=1}^{K} y_ik · log(ŷ_ik)

    Because y is one-hot (only one k per row equals 1), this simplifies to:

        L = -(1/m) Σ_{i=1}^{m} log(ŷ_{i, true_class_i})

    Only the log-probability of the correct class contributes to the loss.

    BACKWARD (gradient w.r.t. ŷ)
    --------------------------------
        dL/dŷ_ik = -(1/m) · y_ik / ŷ_ik

    NOTE: When used together with Softmax, use SoftmaxCCE below instead.
    That class computes the combined gradient dL/dZ = (ŷ - y)/m directly,
    which is more stable and avoids the Softmax Jacobian entirely.
    """

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        y_pred_clipped = clip(y_pred)
        # Only sum terms where y_true = 1 (exploits one-hot structure)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return float(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]
        y_pred_clipped = clip(y_pred)
        return (-y_true / y_pred_clipped / m).astype(DTYPE)


# ---------------------------------------------------------------------------
# Softmax + CCE fused — numerically stable, for multi-class output layers
# ---------------------------------------------------------------------------

class SoftmaxCCE(Loss):
    """Numerically stable fusion of Softmax activation + Categorical Cross-Entropy.

    WHY FUSE THEM?
    --------------
    The standalone CCE gradient dL/dŷ = -y/ŷ, when multiplied by the
    full Softmax Jacobian, simplifies beautifully:

        dL/dZ = ŷ - y     (one line — no Jacobian computation needed)

    This is both faster and more numerically stable than computing them
    separately. It also avoids dividing by potentially very small ŷ values.

    DERIVATION
    -----------
    Let a_k = softmax(z)_k = exp(z_k) / Σ exp(z_j).
    The CCE loss for one sample: L = -Σ_k y_k log(a_k).

    dL/dz_j = Σ_k dL/da_k · da_k/dz_j

    The Softmax Jacobian gives:
        da_k/dz_j = a_k(δ_kj - a_j)

    Substituting dL/da_k = -y_k/a_k:
        dL/dz_j = Σ_k (-y_k/a_k) · a_k(δ_kj - a_j)
                = Σ_k -y_k(δ_kj - a_j)
                = -y_j + a_j Σ_k y_k
                = a_j - y_j          (since Σ_k y_k = 1 for one-hot)

    Result (across all m samples):
        dL/dZ = (A - Y) / m

    This is the cleanest gradient in all of deep learning.

    USAGE NOTE
    ----------
    When you use this as your loss function, the output layer should use
    a *Linear* activation (not Softmax). The Softmax is computed here
    internally. The `gradient()` returns dL/dZ (pre-activation gradient),
    so the output layer should skip its own activation backward.
    """

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        Z_shifted = Z - Z.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """y_pred here should be *logits* (pre-softmax) or softmax outputs.

        We reapply softmax internally for the loss so either works.
        For consistency with gradient(), treat y_pred as logits.
        """
        # If y_pred looks like logits (not bounded 0-1), apply softmax
        A = self._softmax(y_pred)
        m = y_true.shape[0]
        A_clipped = clip(A)
        loss = -np.sum(y_true * np.log(A_clipped)) / m
        return float(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Returns dL/dZ = (softmax(Z) - Y) / m.

        y_pred should be the raw logits Z (output of Linear layer).
        """
        m = y_true.shape[0]
        A = self._softmax(y_pred)
        return ((A - y_true) / m).astype(DTYPE)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "mse":                    MSE,
    "mean_squared_error":     MSE,
    "bce":                    BinaryCrossEntropy,
    "binary_crossentropy":    BinaryCrossEntropy,
    "cce":                    CategoricalCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "softmax_cce":            SoftmaxCCE,
}


def get_loss(name: str | object) -> Loss:
    """Return a loss instance by name or pass through an instance."""
    if isinstance(name, str):
        key = name.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown loss '{name}'. Available: {list(_REGISTRY.keys())}"
            )
        return _REGISTRY[key]()
    return name
