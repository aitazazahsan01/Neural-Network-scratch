"""
Gradient checking script — verifies our backprop implementation is correct.

WHAT IS GRADIENT CHECKING?
----------------------------
We compare analytically computed gradients (from backprop) against
numerically estimated gradients (finite differences).

The numerical gradient is:
    df/dθ ≈ [f(θ + ε) - f(θ - ε)] / (2ε)

This is the *two-sided finite difference* approximation. It is accurate
to O(ε²) — much better than the one-sided version O(ε).

If our backprop is correct, the relative difference should be < 1e-5:
    ||grad_analytic - grad_numeric|| / (||grad_analytic|| + ||grad_numeric||)

If it's > 1e-3, there's almost certainly a bug.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from neuralnet import Sequential, DenseLayer
from neuralnet.losses import BinaryCrossEntropy, MSE, SoftmaxCCE
from neuralnet.utils.data_utils import one_hot_encode


def numerical_gradient(model, X, y, epsilon=1e-5):
    """Compute numerical gradients for all weights using finite differences."""
    num_grads = {}

    # Run one forward + backward to get the current loss
    y_pred = model.forward(X, training=False)
    loss_fn = model._loss

    # Iterate over each layer and each parameter
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, 'W') or layer.W is None:
            continue

        for param_name in ['W', 'b']:
            param = getattr(layer, param_name)
            grad = np.zeros_like(param)

            # Iterate over every element of the parameter matrix
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index

                # Save original value
                orig = param[idx].copy()

                # Loss at (param + epsilon)
                param[idx] = orig + epsilon
                y_pred_plus = model.forward(X, training=False)
                loss_plus   = loss_fn.compute(y, y_pred_plus)

                # Loss at (param - epsilon)
