"""
neuralnet/initializers.py
=========================
Weight and bias initialisation strategies.

WHY INITIALISATION MATTERS
---------------------------
Before training begins every weight W must be assigned a starting value.
This choice has a dramatic effect on whether the network can learn at all.

Consider what happens during the forward pass:

    Z = X @ W + b          (pre-activation, shape [m, n_out])
    A = activation(Z)

If W is *too large*, |Z| grows exponentially through each layer →
activations saturate (sigmoid/tanh near 0 or 1, gradient ≈ 0) →
gradients vanish in the lower layers → training stalls.

If W is *too small*, |Z| shrinks exponentially → activations near 0 →
gradients also collapse → same problem.

The goal is to keep the variance of activations *roughly constant* across
every layer so that gradients flow without exploding or vanishing.

VARIANCE ANALYSIS (Xavier initialisation)
------------------------------------------
Assume inputs x_i are i.i.d. with mean 0, variance Var(x).
A single neuron computes z = Σ w_i x_i (n_in terms).

    Var(z) = n_in · Var(w) · Var(x)

To maintain Var(z) = Var(x) we need:

    Var(w) = 1 / n_in          ← "LeCun" init for sigmoid/tanh

Averaging fan_in and fan_out (Glorot & Bengio, 2010):

    Var(w) = 2 / (fan_in + fan_out)   ← Xavier / Glorot

For ReLU (He et al., 2015) — ReLU zeros out half the neurons so
variance is halved; compensate by doubling:

    Var(w) = 2 / fan_in        ← He / Kaiming

Available classes
-----------------
* ZeroInit          – all zeros (good for biases, BAD for weights)
* RandomNormal      – N(0, σ); naive baseline
* XavierUniform     – Uniform[−√(6/(fi+fo)), +√(6/(fi+fo))]
* XavierNormal      – N(0, √(2/(fi+fo)))
* HeNormal          – N(0, √(2/fan_in))   (recommended for ReLU)
* HeUniform         – Uniform[−√(6/fan_in), +√(6/fan_in)]
"""

import numpy as np
from .tensor import fan_in_fan_out, DTYPE


class ZeroInit:
    """Return an all-zero array of the requested shape.

    Appropriate for biases (b is a learned offset, not a feature detector).
    *Never* use this for weights: all neurons would produce identical outputs,
    receive identical gradients, and remain symmetric forever — the network
    cannot break symmetry and no learning occurs.
    """

    def __call__(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape, dtype=DTYPE)

    def __repr__(self) -> str:
        return "ZeroInit()"


class RandomNormal:
    """Sample weights from N(0, sigma).

    Parameters
    ----------
    sigma : float
        Standard deviation. Default is 0.01 (small but non-zero).

    When to use
    -----------
    Quick experiments and sanity checks. For deep networks, prefer
    Xavier or He to avoid vanishing / exploding activations.
    """

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = sigma

    def __call__(self, shape: tuple) -> np.ndarray:
        return np.random.randn(*shape).astype(DTYPE) * self.sigma

    def __repr__(self) -> str:
        return f"RandomNormal(sigma={self.sigma})"


class XavierUniform:
    """Glorot uniform initialisation (Glorot & Bengio, 2010).

    Draws from Uniform(−limit, +limit) where:

        limit = sqrt(6 / (fan_in + fan_out))

    This keeps the variance of outputs the same as the variance of inputs
    under the assumption that the activation is linear (or approximately so,
    e.g. tanh near zero). Recommended for sigmoid and tanh activations.
    """

    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = fan_in_fan_out(shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape).astype(DTYPE)

    def __repr__(self) -> str:
        return "XavierUniform()"


class XavierNormal:
    """Glorot normal initialisation.

    Draws from N(0, std) where:

        std = sqrt(2 / (fan_in + fan_out))
    """

    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = fan_in_fan_out(shape)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape).astype(DTYPE) * std

    def __repr__(self) -> str:
        return "XavierNormal()"


class HeNormal:
    """He / Kaiming normal initialisation (He et al., 2015).

    Draws from N(0, std) where:

        std = sqrt(2 / fan_in)

    Derived by accounting for ReLU's property of zeroing ~half its inputs,
    which halves the variance. Multiplying by 2 compensates for this.

    This is the *recommended default* whenever you use ReLU or LeakyReLU.
    """

    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in, _ = fan_in_fan_out(shape)
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(DTYPE) * std

    def __repr__(self) -> str:
