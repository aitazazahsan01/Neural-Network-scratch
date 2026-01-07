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
