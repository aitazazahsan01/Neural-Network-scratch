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

