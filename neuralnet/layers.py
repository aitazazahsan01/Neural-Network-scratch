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
