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
    """SGD with Momentum (Polyak, 1964).

    Maintains a velocity vector v that accumulates an exponentially
    decaying average of past gradients:

        v ← β·v + (1-β)·dW
        W ← W - lr·v

    INTUITION
    ----------
    Think of a ball rolling down a hill:
        • In directions with consistent gradients, velocity builds up → faster.
        • In directions with oscillating gradients, velocity averages out → smoother.
    Momentum dampens oscillations and accelerates convergence in the
    consistent gradient direction.

    WHY (1-β) INSTEAD OF JUST β·v + dW?
    ------------------------------------
    The (1-β) factor scales the gradient so that v's magnitude is
    comparable to dW regardless of β. Some texts omit (1-β) and absorb
    it into lr — both formulations are equivalent with a rescaled lr.

    Parameters
    ----------
    lr : float
        Learning rate. Typical: 0.001–0.1.
    beta : float
        Momentum decay factor. Default 0.9 (recommended).
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(lr)
        self.beta = beta

    def update(self, layer_id: int, params: dict, grads: dict) -> dict:
        if layer_id not in self._state:
            # Initialise velocity to zeros, matching parameter shapes
            self._state[layer_id] = {
                key: np.zeros_like(params[key], dtype=DTYPE)
                for key in params
            }

        updated = {}
        for key in params:
            if grads.get(key) is None:
                updated[key] = params[key]
                continue

            v = self._state[layer_id][key]

            # Velocity update: v ← β·v + (1-β)·dW
            v = self.beta * v + (1.0 - self.beta) * grads[key]
            self._state[layer_id][key] = v

            # Parameter update: W ← W - lr·v
            updated[key] = params[key] - self.lr * v

        return updated


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

class RMSProp(Optimizer):
    """RMSProp (Hinton, unpublished but widely cited, ~2012).

    Maintains an exponential moving average of *squared* gradients:

        s ← β·s + (1-β)·dW²
        W ← W - lr · dW / √(s + ε)

    INTUITION
    ----------
    If a parameter has consistently *large* gradients, s is large →
    effective lr is small → step size is reduced automatically.

    If a parameter has small gradients, s is small → effective lr is
    large → larger relative steps.

    This *adapts* the learning rate per parameter. Parameters in flat
    regions get larger updates; parameters in steep regions get smaller.

    ε prevents division by zero (default 1e-8).

    Parameters
    ----------
    lr : float
        Global learning rate. Default 0.001.
    beta : float
        Decay factor for squared-gradient moving average. Default 0.9.
    epsilon : float
        Numerical stability constant. Default 1e-8.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta: float = 0.9,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon

    def update(self, layer_id: int, params: dict, grads: dict) -> dict:
        if layer_id not in self._state:
            self._state[layer_id] = {
                key: np.zeros_like(params[key], dtype=DTYPE)
                for key in params
            }

        updated = {}
        for key in params:
            if grads.get(key) is None:
                updated[key] = params[key]
                continue

            s = self._state[layer_id][key]

            # s ← β·s + (1-β)·dW²
            s = self.beta * s + (1.0 - self.beta) * (grads[key] ** 2)
            self._state[layer_id][key] = s

            # W ← W - lr · dW / √(s + ε)
            updated[key] = params[key] - self.lr * grads[key] / (np.sqrt(s) + self.epsilon)

        return updated


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam(Optimizer):
    """Adam: Adaptive Moment Estimation (Kingma & Ba, 2015).

    Combines momentum (1st moment) and RMSProp (2nd moment):

        m  ← β₁·m  + (1-β₁)·dW          (biased 1st moment estimate)
        v  ← β₂·v  + (1-β₂)·dW²         (biased 2nd moment estimate)

    Bias correction (compensates for initialisation at zero):
        m̂  = m  / (1 - β₁ᵗ)
        v̂  = v  / (1 - β₂ᵗ)

    Update:
        W  ← W - lr · m̂ / (√v̂ + ε)

    WHY BIAS CORRECTION?
    ---------------------
    At step t=1, m = (1-β₁)·dW₁. If β₁=0.9, m ≈ 0.1·dW₁ — far smaller
    than the true gradient. Dividing by (1-β₁¹) = 0.1 restores the
    estimate to the true gradient scale. As t grows, β₁ᵗ → 0 and the
    correction vanishes (moments have "warmed up").

    HYPERPARAMETER DEFAULTS (Kingma & Ba recommendation)
    -------------------------------------------------------
        lr   = 0.001
        β₁   = 0.9
        β₂   = 0.999
        ε    = 1e-8

    These defaults work well on a very wide range of problems without tuning.

    WHY IS ADAM THE DEFAULT CHOICE?
    --------------------------------
    • Adaptive lr per parameter → no manual lr schedule needed.
    • Momentum → stable convergence.
    • Bias correction → correct early-step behaviour.
    • Computationally cheap: O(parameters) memory and time.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, layer_id: int, params: dict, grads: dict) -> dict:
        if layer_id not in self._state:
            self._state[layer_id] = {
                "m": {key: np.zeros_like(params[key], dtype=DTYPE) for key in params},
                "v": {key: np.zeros_like(params[key], dtype=DTYPE) for key in params},
            }

        # Bias-correction denominator uses the *current* step count
        t = self._step + 1   # +1 because step() is called after update

        updated = {}
        for key in params:
            if grads.get(key) is None:
                updated[key] = params[key]
                continue

            m = self._state[layer_id]["m"][key]
            v = self._state[layer_id]["v"][key]

            # 1st moment (momentum)
            m = self.beta1 * m + (1.0 - self.beta1) * grads[key]
            # 2nd moment (RMSProp)
            v = self.beta2 * v + (1.0 - self.beta2) * (grads[key] ** 2)

            # Store updated moments
            self._state[layer_id]["m"][key] = m
            self._state[layer_id]["v"][key] = v

            # Bias-corrected estimates
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)

            # Parameter update
            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"
        )


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "sgd":          SGD,
    "momentum":     SGDMomentum,
    "sgd_momentum": SGDMomentum,
    "rmsprop":      RMSProp,
    "adam":         Adam,
}


def get_optimizer(name: str | object, **kwargs) -> Optimizer:
    """Return an optimizer instance by name or pass through an instance."""
    if isinstance(name, str):
        key = name.lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown optimizer '{name}'. "
                f"Available: {list(_REGISTRY.keys())}"
            )
