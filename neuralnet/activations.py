"""
neuralnet/activations.py
========================
Activation functions and their exact derivatives.

WHAT IS AN ACTIVATION FUNCTION?
---------------------------------
A neuron computes a *linear* transformation first:

    Z = X @ W + b       (pre-activation / "logit")

Without a non-linear step the entire network — regardless of depth —
would collapse into a single linear transformation. A non-linear
activation f breaks this collapse:

    A = f(Z)            (post-activation output)

Each activation has a different shape, saturation behaviour, and
gradient profile. Choosing the wrong one is a common source of
training failure.

THE DERIVATIVE REQUIREMENT
---------------------------
Backpropagation needs the *derivative* of the activation with respect
to its input Z (the pre-activation). Each class below implements:

    forward(Z)          → A  (used in forward pass)
    backward(dA, Z)     → dZ (used in backward pass)

where dA is the gradient flowing back from the layer above, and the
chain rule gives:

    dZ = dA * f'(Z)     (* is element-wise multiplication)

Note: Softmax is a special case — its gradient is typically merged with
the cross-entropy loss for numerical stability (see SoftmaxCCE shortcut
in losses.py). The standalone Softmax.backward() below handles the
general case via the Jacobian.

Available activations
---------------------
* Linear    – identity; used in the output of regression models
* ReLU      – max(0, z); workhorse of modern deep learning
* LeakyReLU – leaky variant that avoids "dying ReLU"
* Sigmoid   – 1/(1+e^−z); maps ℝ → (0,1); used in binary output
* Tanh      – 2σ(2z)−1; maps ℝ → (−1,1); centered version of sigmoid
* Softmax   – normalised exponential; maps ℝ^K → probability simplex
"""

import numpy as np
from .tensor import clip, EPSILON, DTYPE


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Activation:
    """Abstract base for all activation functions."""

    def forward(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return self.forward(Z)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


# ---------------------------------------------------------------------------
# Concrete activations
# ---------------------------------------------------------------------------

class Linear(Activation):
    """Identity activation: f(z) = z, f'(z) = 1.

    Use this on the *output* layer of a regression model when the target
    is an unbounded real number. It applies no squashing.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return Z.astype(DTYPE)

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        # f'(z) = 1, so dZ = dA * 1 = dA
        return dA.astype(DTYPE)


class ReLU(Activation):
    """Rectified Linear Unit: f(z) = max(0, z).

    FORWARD
    -------
        A = max(0, Z)   element-wise

    BACKWARD (derivative)
    ------
        f'(z) = 1  if z > 0
                0  if z ≤ 0

        dZ = dA * f'(Z)

    WHY RELU?
    ---------
    1. No saturation for positive inputs → gradient flows freely.
    2. Cheap to compute (just a threshold).
    3. Introduces sparsity (many neurons output 0).

    DYING RELU PROBLEM
    ------------------
    If a neuron's pre-activation is always negative (e.g., after a large
    negative weight update), f'(z) = 0 forever — the neuron is "dead" and
    never recovers. Mitigations: He init, careful learning rate, LeakyReLU.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, Z).astype(DTYPE)

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        # Derivative is 1 where Z > 0, else 0.
        # (Z > 0) produces a boolean mask; multiplied by dA gives dZ.
        dZ = dA * (Z > 0).astype(DTYPE)
        return dZ


class LeakyReLU(Activation):
    """Leaky ReLU: f(z) = z if z>0 else alpha*z.

    Derivative:
        f'(z) = 1     if z > 0
                alpha  if z ≤ 0

    The small non-zero slope *alpha* (default 0.01) keeps gradients
    alive for negative pre-activations — solving the dying ReLU problem.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.where(Z > 0, Z, self.alpha * Z).astype(DTYPE)

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = np.where(Z > 0, 1.0, self.alpha).astype(DTYPE)
        return dA * dZ

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class Sigmoid(Activation):
    """Logistic sigmoid: f(z) = 1 / (1 + e^{-z}).

    Maps ℝ → (0, 1). Commonly used on the *output* neuron of a binary
    classifier so the output can be interpreted as a probability.

    FORWARD
    -------
        A = σ(Z) = 1 / (1 + exp(-Z))

    BACKWARD
    --------
    The derivative has an elegant form in terms of the *output* A:

