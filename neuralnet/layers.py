"""
neuralnet/layers.py
===================
Neural network layers.

A NEURON — THE ATOMIC UNIT
---------------------------
A single neuron with n inputs computes:

    z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
      = wᵀx + b

Where:
    x ∈ ℝⁿ  — input vector
    w ∈ ℝⁿ  — weight vector (one weight per input)
    b ∈ ℝ   — bias (shifts the activation threshold)
    z ∈ ℝ   — pre-activation (the "logit")

Then:
    a = f(z)  — post-activation output

The weights and bias are the *learnable parameters* — the entire job of
training is to find W and b such that the network output is close to y.

A DENSE LAYER — VECTORISED NEURONS
------------------------------------
A Dense layer with n_in inputs and n_out outputs stacks n_out neurons
side by side. For a mini-batch of m samples:

    Z = X @ W + b

    Shape breakdown:
        X : (m, n_in)   — m samples, n_in features each
        W : (n_in, n_out) — weight matrix
        b : (1, n_out)   — bias row, broadcast across m samples
        Z : (m, n_out)   — pre-activations

    A = f(Z)  : (m, n_out)  — post-activations

This is called a "fully connected" or "dense" layer because every input
neuron is connected to every output neuron.

FORWARD PASS
------------
The forward pass chains layers:

    A₀ = X                       (input)
    Z₁ = A₀ @ W₁ + b₁
    A₁ = f₁(Z₁)
    Z₂ = A₁ @ W₂ + b₂
    A₂ = f₂(Z₂)
    ...
    ŷ  = Aₗ                      (output)

    L = loss(y, ŷ)

We *cache* (A_prev, Z) at each layer — they are needed during backprop.

BACKPROPAGATION & THE CHAIN RULE
----------------------------------
After the forward pass we have L. We want to update each W and b to
reduce L. By the chain rule we propagate gradients *backwards*:

    Layer l receives: dA_l = dL/dA_l  (gradient of loss w.r.t. its output)

Step 1 — Gradient through activation:
    dZ_l = dA_l * f'(Z_l)           (* = element-wise)

Step 2 — Gradient w.r.t. weights:
    dW_l = A_{l-1}ᵀ @ dZ_l / m

    Derivation:
        L is a scalar, Z_l = A_{l-1} @ W_l + b_l
        dL/dW_l = A_{l-1}ᵀ @ dZ_l   (chain rule + transpose rule)
        Divide by m because Z contains m samples stacked row-wise.

Step 3 — Gradient w.r.t. bias:
    db_l = mean(dZ_l, axis=0, keepdims=True)

    Derivation:
        Each bias b_k is added to every sample's pre-activation z_k.
        dL/db_k = Σ_{i=1}^{m} dL/dz_ik / m  →  mean across samples.

Step 4 — Gradient w.r.t. input (to pass to layer l-1):
    dA_{l-1} = dZ_l @ W_lᵀ

    Derivation:
        Z_l = A_{l-1} @ W_l  →  dL/dA_{l-1} = dZ_l @ W_lᵀ

This repeats from the last layer back to the first layer, with each layer
receiving the dA that the layer *above* it passed back.

Layers implemented
------------------
* DenseLayer    – fully connected layer
* DropoutLayer  – stochastic regularisation (training only)
"""

import numpy as np
from .tensor import DTYPE, batch_size
from .activations import Activation, Linear, get_activation
from .initializers import HeNormal, XavierNormal, ZeroInit, get_initializer


# ---------------------------------------------------------------------------
# Dense Layer
# ---------------------------------------------------------------------------

class DenseLayer:
    """A fully connected (Dense) layer.

    Parameters
    ----------
    n_out : int
        Number of neurons (output units) in this layer.
    activation : str or Activation
        Activation function. Default "relu".
        Use "linear" for the output of a regression or softmax model.
        Use "sigmoid" for binary output.
        Use "softmax" for multi-class output (or use SoftmaxCCE loss +
        "linear" activation for better numerical stability).
    weight_init : str or Initializer, optional
        Weight initialisation strategy. Default adapts to activation:
        "he_normal" for ReLU/LeakyReLU, "xavier_normal" for others.
    bias_init : str or Initializer, optional
        Bias initialisation strategy. Default "zeros".
    name : str, optional
        Human-readable layer name (for model summary).

    Attributes (set during build)
    ----------------------------
    W : np.ndarray  shape (n_in, n_out)
    b : np.ndarray  shape (1, n_out)

    Gradient attributes (set during backward)
    ------------------------------------------
    dW : np.ndarray  shape (n_in, n_out)
    db : np.ndarray  shape (1, n_out)
    """

    def __init__(
        self,
        n_out: int,
        activation: str | Activation = "relu",
        weight_init: str | object | None = None,
        bias_init: str | object = "zeros",
        name: str | None = None,
    ) -> None:
        self.n_out = n_out
        self.activation = get_activation(activation)
        self.bias_init = get_initializer(bias_init)
        self.name = name

        # Resolve weight initialiser — default depends on activation type
        if weight_init is None:
            act_name = type(self.activation).__name__.lower()
