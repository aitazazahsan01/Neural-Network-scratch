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
            if act_name in ("relu", "leakyrelu"):
                weight_init = "he_normal"
            else:
                weight_init = "xavier_normal"
        self.weight_init = get_initializer(weight_init)

        # Learnable parameters (initialised during build)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

        # Gradients (computed during backward)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

        # Forward-pass cache needed by backward
        self._cache: dict = {}

        self._built = False

    # ------------------------------------------------------------------
    # Build (called lazily on first forward pass)
    # ------------------------------------------------------------------

    def build(self, n_in: int) -> None:
        """Allocate and initialise W and b given the input dimension n_in.

        This is called automatically on the first forward pass so that the
        user doesn't need to specify input dimensions explicitly — they are
        inferred from the data (like Keras' `build` method).
        """
        self.n_in = n_in
        self.W = self.weight_init((n_in, self.n_out))   # (n_in, n_out)
        self.b = self.bias_init((1, self.n_out))         # (1, n_out)
        self._built = True

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, A_prev: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute Z = A_prev @ W + b, then A = activation(Z).

        Parameters
        ----------
        A_prev : np.ndarray, shape (m, n_in)
            Output from the previous layer (or raw input X for layer 1).
        training : bool
            Whether we are in training mode. Used by Dropout.

        Returns
        -------
        A : np.ndarray, shape (m, n_out)
        """
        if not self._built:
            self.build(A_prev.shape[1])

        # Linear transformation: Z = X·W + b
        # b is (1, n_out) — NumPy broadcasts it to (m, n_out)
        Z = A_prev @ self.W + self.b       # (m, n_out)

        # Non-linear activation: A = f(Z)
        A = self.activation.forward(Z)     # (m, n_out)

        # Cache what we need for the backward pass
        self._cache = {"A_prev": A_prev, "Z": Z}

        return A

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Backpropagate gradient through this layer.

        Parameters
        ----------
        dA : np.ndarray, shape (m, n_out)
            Gradient of loss w.r.t. the *output* of this layer (dL/dA).
            For the last layer this comes from the loss function.
            For hidden layers this is dA_{l} passed back from layer l+1.

        Returns
        -------
        dA_prev : np.ndarray, shape (m, n_in)
            Gradient to pass to the previous layer (dL/dA_{l-1}).

        Side effects
        ------------
        Sets self.dW and self.db — the optimizer reads these to update W, b.

        STEP-BY-STEP DERIVATION
        -----------------------
        Given:
            Z = A_prev @ W + b      (forward pass)
            A = f(Z)                (activation)

        Chain rule:
            dL/dZ = dL/dA * df/dZ = dA * f'(Z)          [element-wise *]

        Gradients for parameters:
            dL/dW = A_prevᵀ @ dZ / m
            dL/db = mean(dZ, axis=0, keepdims=True)

        Gradient to pass backward:
            dL/dA_prev = dZ @ Wᵀ
        """
        A_prev = self._cache["A_prev"]     # (m, n_in)
        Z      = self._cache["Z"]          # (m, n_out)
        m      = A_prev.shape[0]

        # Step 1: gradient through activation function
        dZ = self.activation.backward(dA, Z)   # (m, n_out)

        # Step 2: gradient w.r.t. W
        self.dW = (A_prev.T @ dZ) / m          # (n_in, n_out)

        # Step 3: gradient w.r.t. b (mean across samples, keep shape (1, n_out))
        self.db = np.mean(dZ, axis=0, keepdims=True)  # (1, n_out)

        # Step 4: gradient to pass to the previous layer
        dA_prev = dZ @ self.W.T                # (m, n_in)

        return dA_prev.astype(DTYPE)

    # ------------------------------------------------------------------
    # Parameter access (used by optimizer)
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict:
        """Return learnable parameters as a dict."""
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict:
        """Return computed gradients as a dict."""
        return {"W": self.dW, "b": self.db}

    def set_params(self, params: dict) -> None:
        """Set parameters (used by optimizer after update step)."""
        self.W = params["W"]
        self.b = params["b"]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def n_params(self) -> int:
        """Total number of learnable parameters in this layer."""
        if not self._built:
            return 0
        return self.W.size + self.b.size

    def summary_row(self) -> dict:
        """Return a dict suitable for the model summary table."""
        name = self.name or f"Dense({self.n_out})"
        output_shape = f"(None, {self.n_out})"
        return {
            "Layer":        name,
            "Output Shape": output_shape,
            "Activation":   repr(self.activation),
            "# Params":     self.n_params(),
        }

    def __repr__(self) -> str:
        built_info = f", n_in={self.n_in}" if self._built else ""
        return (
            f"DenseLayer(n_out={self.n_out}, "
            f"activation={self.activation}{built_info})"
        )


# ---------------------------------------------------------------------------
# Dropout Layer
# ---------------------------------------------------------------------------

class DropoutLayer:
    """Dropout regularisation layer (Srivastava et al., 2014).

    WHAT IS DROPOUT?
    ----------------
    During *training*, randomly sets a fraction (rate) of neuron outputs
    to zero on each forward pass. The remaining outputs are *scaled up* by
    1/(1-rate) so that the expected value is preserved (inverted dropout).

    WHY DOES THIS HELP?
    -------------------
    Overfitting occurs when the network memorises training data rather than
    learning generalisable patterns. Dropout prevents co-adaptation of
    neurons: no single neuron can rely on the presence of any other neuron,
    forcing the network to learn more robust, distributed representations.

    Effect: acts as implicit ensemble training — each forward pass trains a
    different "thinned" sub-network. At test time (training=False) no
    dropout is applied and the full network is used (because weights were
    already scaled up during training via the inverted method).

    INVERTED DROPOUT (the standard approach)
    -----------------------------------------
    Training:
        mask   = Bernoulli(1 - rate)   shape (m, n)
        A_drop = A * mask / (1 - rate)   ← scale up to preserve E[A_drop] = E[A]

    Inference:
        A_drop = A                        ← no change needed

    Parameters
    ----------
    rate : float in (0, 1)
        Fraction of neurons to *drop* (set to zero).
        Typical values: 0.2–0.5.
    """

    def __init__(self, rate: float = 0.5, name: str | None = None) -> None:
        assert 0.0 < rate < 1.0, "Dropout rate must be in (0, 1)."
        self.rate = rate
        self.name = name or f"Dropout({rate})"
        self._mask: np.ndarray | None = None

        # Expose as no-op for optimizer (no learnable params)
        self.W = None
        self.b = None
        self.dW = None
        self.db = None

    def forward(self, A_prev: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            # Bernoulli mask: 1 with probability (1-rate), 0 with probability rate
            self._mask = (np.random.rand(*A_prev.shape) > self.rate).astype(DTYPE)
            return (A_prev * self._mask / (1.0 - self.rate)).astype(DTYPE)
        # Inference: return unchanged
        return A_prev.astype(DTYPE)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        # Only the neurons that were kept contribute to the gradient
        return (dA * self._mask / (1.0 - self.rate)).astype(DTYPE)

    @property
    def params(self) -> dict:
        return {}

    @property
    def grads(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass  # no parameters to set

    def n_params(self) -> int:
        return 0

    def summary_row(self) -> dict:
        return {
            "Layer":        self.name,
            "Output Shape": "(None, same as input)",
            "Activation":   "—",
            "# Params":     0,
        }

    def __repr__(self) -> str:
        return f"DropoutLayer(rate={self.rate})"
