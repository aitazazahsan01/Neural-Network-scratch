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
                param[idx] = orig - epsilon
                y_pred_minus = model.forward(X, training=False)
                loss_minus   = loss_fn.compute(y, y_pred_minus)

                # Restore
                param[idx] = orig

                # Finite difference approximation
                grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                it.iternext()

            num_grads[(i, param_name)] = grad

    return num_grads


def relative_difference(a, b):
    """||a-b|| / (||a|| + ||b|| + 1e-10)"""
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-10)


def run_gradient_check(model, X, y, label=""):
    print(f"\n{'=' * 55}")
    print(f"Gradient Check: {label}")
    print(f"{'=' * 55}")

    # Forward + backward to compute analytic gradients
    y_pred = model.forward(X, training=False)
    model.backward(y, y_pred)

    # Numerical gradients
    num_grads = numerical_gradient(model, X, y)

    all_passed = True
    for (i, param_name), num_grad in num_grads.items():
        layer = model.layers[i]
        if param_name == 'W':
            analytic_grad = layer.dW
        else:
            analytic_grad = layer.db

        if analytic_grad is None:
            print(f"  Layer {i} {param_name}: SKIP (no gradient computed)")
            continue

        rel_diff = relative_difference(analytic_grad, num_grad)
        status   = "[PASS]" if rel_diff < 1e-5 else "[FAIL]"
        if rel_diff >= 1e-5:
            all_passed = False

        print(
            f"  Layer {i} ({type(layer).__name__}) {param_name}: "
            f"rel_diff = {rel_diff:.2e}  {status}"
        )

    if all_passed:
        print(f"\n[ALL PASS] ALL GRADIENT CHECKS PASSED for: {label}")
    else:
        print(f"\n[FAIL] SOME CHECKS FAILED for: {label}")

    return all_passed


np.random.seed(0)

# ─── Test 1: MSE + Linear output (regression) ────────────────────────────────
X1 = np.random.randn(8, 3).astype(np.float64)
y1 = np.random.randn(8, 1).astype(np.float64)

m1 = Sequential([
    DenseLayer(4,  activation="tanh",   weight_init="xavier_normal"),
    DenseLayer(1,  activation="linear", weight_init="xavier_normal"),
])
m1.compile(loss="mse", optimizer="sgd")
run_gradient_check(m1, X1, y1, label="MSE + Tanh + Linear")


# ─── Test 2: BCE + Sigmoid output (binary classification) ────────────────────
X2 = np.random.randn(8, 4).astype(np.float64)
y2 = (np.random.rand(8, 1) > 0.5).astype(np.float64)
