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

