"""
neuralnet/optimizers.py
=======================
Gradient-based optimisation algorithms.

WHAT DOES AN OPTIMIZER DO?
---------------------------
After backpropagation computes the gradient dL/dW for every layer,
the optimizer decides *how* to update the parameters W.

The simplest update rule (vanilla SGD):

    W ← W - lr * dW

But this naive rule has problems in practice:
    1. Noisy gradients from mini-batches cause erratic updates.
    2. The same learning rate for all parameters is often suboptimal.
    3. Flat regions ("saddle points") slow learning dramatically.

The optimizers below each address these problems progressively.

LEARNING RATE (η / lr)
-----------------------
The learning rate controls step size in parameter space.

    Too high  → overshoots minima, loss diverges or oscillates.
    Too low   → converges very slowly, gets stuck in local minima.

Finding a good lr is often the most impactful hyperparameter choice.
Typical values: 1e-4 to 1e-1 depending on optimizer and architecture.

Implemented optimizers
-----------------------
1. SGD (Stochastic Gradient Descent)
   W ← W - lr * dW
   Baseline; works for simple problems but noisy.

2. SGD with Momentum
   v ← β*v + (1-β)*dW      (exponential moving average of gradients)
   W ← W - lr * v
   Reduces oscillations, speeds up convergence in consistent directions.

3. RMSProp (Root Mean Square Propagation)
   s ← β*s + (1-β)*dW²
   W ← W - lr * dW / √(s + ε)
   Adapts lr per-parameter; larger s → smaller effective step.
   Excellent for non-stationary objectives (e.g., RNN training).

4. Adam (Adaptive Moment Estimation)
   m ← β₁*m + (1-β₁)*dW          (1st moment / momentum)
   v ← β₂*v + (1-β₂)*dW²         (2nd moment / RMSProp)
   m̂ ← m / (1-β₁ᵗ)               (bias correction)
   v̂ ← v / (1-β₂ᵗ)               (bias correction)
   W ← W - lr * m̂ / (√v̂ + ε)

   Adam combines Momentum + RMSProp with bias correction for the
   fact that m and v start at 0 (biased toward 0 early in training).
   Recommended default for most networks.

Each optimizer stores state (velocities, squared-gradient accumulations)
as a dict keyed by layer index and parameter name.
"""

import numpy as np
from .tensor import DTYPE, EPSILON


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Optimizer:
    """Abstract base for all optimizers."""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        self._state: dict = {}          # persistent state across update calls
        self._step: int = 0             # global step counter (used for bias correction)

    def update(self, layer_id: int, params: dict, grads: dict) -> dict:
        """Update *params* using *grads* and return the new params dict.

        Parameters
        ----------
        layer_id : int
            Unique identifier for the layer (used as state key).
        params : dict
            {"W": np.ndarray, "b": np.ndarray}
        grads : dict
            {"W": np.ndarray, "b": np.ndarray}

        Returns
        -------
        dict : updated params
        """
        raise NotImplementedError

    def step(self) -> None:
        """Increment the global step counter (called once per batch)."""
        self._step += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent.

    Update rule:
        W ← W - lr * dW

    "Stochastic" because we compute the gradient on a *mini-batch*
    (a random subset of the training data) rather than the full dataset.

    Mini-batch gradient descent is a compromise between:
        • Full-batch GD: accurate gradient, but slow (uses all data).
        • SGD (single sample): fast, but extremely noisy gradient.

    Mini-batches (typically 32–512 samples) give a good gradient
    estimate while keeping computation tractable.

    Parameters
    ----------
    lr : float
        Learning rate η (step size). Default 0.01.
    """

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def update(self, layer_id: int, params: dict, grads: dict) -> dict:
        updated = {}
        for key in params:
            if grads.get(key) is None:
                updated[key] = params[key]
                continue
            # W ← W - η·dW
            updated[key] = params[key] - self.lr * grads[key]
        return updated


# ---------------------------------------------------------------------------
# SGD + Momentum
# ---------------------------------------------------------------------------

class SGDMomentum(Optimizer):
