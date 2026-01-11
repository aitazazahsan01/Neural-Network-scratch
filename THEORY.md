# Neural Networks from Scratch — Complete Theory Guide

> **How to use this document**: Read it alongside the source code. Every section
> references specific files and line numbers where the mathematics is implemented.
> Mathematical equations use standard notation. Code snippets show the exact
> NumPy translation of each formula.

---

## Table of Contents

1. [From Biology to Mathematics — What is a Neuron?](#1-from-biology-to-mathematics)
2. [The Dense Layer — Vectorising Many Neurons](#2-the-dense-layer)
3. [Activation Functions — Breaking Linearity](#3-activation-functions)
4. [The Forward Pass — Chaining Layers](#4-the-forward-pass)
5. [Loss Functions — Measuring Wrongness](#5-loss-functions)
6. [Gradient Descent — The Optimisation Engine](#6-gradient-descent)
7. [Backpropagation — The Chain Rule in Action](#7-backpropagation)
8. [A Complete Worked Example](#8-complete-worked-example)
9. [Weight Initialisation — Why Starting Values Matter](#9-weight-initialisation)
10. [Optimizers — Smarter Gradient Descent](#10-optimizers)
11. [Overfitting, Regularisation & Generalisation](#11-overfitting-and-regularisation)
12. [The Learning Rate — The Most Important Hyperparameter](#12-the-learning-rate)
13. [What Happens Internally During Training](#13-what-happens-during-training)
14. [Connecting Every Formula to the Code](#14-connecting-to-code)

---

## 1. From Biology to Mathematics

### The Biological Neuron

A biological neuron receives signals from other neurons through **dendrites**, sums
those signals in the **cell body**, and fires an output signal through the **axon**
if the total input exceeds a threshold.

### The Mathematical Neuron (Perceptron)

Frank Rosenblatt formalised this in 1958. A single mathematical neuron with
**n** inputs computes:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  = wᵀx + b
```

Then applies a non-linear gate:

```
a = f(z)
```

| Term | Meaning | Biological Analogy |
|------|---------|-------------------|
| `x` | input vector | signals arriving at dendrites |
| `w` | weight vector | synapse strengths |
| `b` | bias | baseline firing threshold |
| `z` | pre-activation ("logit") | accumulated charge in cell body |
| `f` | activation function | threshold firing mechanism |
| `a` | output ("activation") | signal sent along axon |

**What the weights represent**: Each weight `wᵢ` encodes how *important*
input `xᵢ` is. A large positive `wᵢ` means "when `xᵢ` is large, fire more".
A large negative `wᵢ` means "when `xᵢ` is large, suppress firing".

**What the bias does**: The bias shifts the activation threshold.
Without bias, the neuron can only fire based on input magnitudes.
With bias, the neuron has a baseline tendency to fire or not fire
independent of input.

> **Code**: `neuralnet/layers.py` — `DenseLayer.forward()`
> The line `Z = A_prev @ self.W + self.b` implements `z = Xw + b` for a
> whole batch of m samples simultaneously.

---

## 2. The Dense Layer

### From One Neuron to Many

A **Dense (fully connected) layer** stacks `n_out` neurons side-by-side,
each with its own weights connecting to every input.

For a mini-batch of **m** samples with **n_in** features each:

```
Z = X @ W + b

Shapes:
  X  : (m, n_in)     — m samples, n_in features
  W  : (n_in, n_out) — n_out neurons, each with n_in weights
  b  : (1, n_out)    — one bias per neuron, broadcast over m
  Z  : (m, n_out)    — pre-activation for all neurons, all samples
```

Each element `Z[i, k]` is the pre-activation of neuron `k` for sample `i`:

```
Z[i, k] = Σⱼ X[i,j] · W[j,k] + b[0,k]
```

This is a matrix multiplication — the most fundamental operation in deep learning.

### Why Matrices?

Without vectorisation we'd compute one neuron at a time:
```python
# slow version (conceptual only)
for i in range(m):
    for k in range(n_out):
        Z[i, k] = sum(X[i, j] * W[j, k] for j in range(n_in)) + b[0, k]
```

Matrix multiplication does this in one call: `Z = X @ W + b`.
NumPy (and GPU libraries) are heavily optimised for exactly this operation.

---

## 3. Activation Functions

### Why Non-Linearity is Essential

Consider two linear layers stacked:

```
A₁ = X W₁ + b₁
A₂ = A₁ W₂ + b₂
   = (X W₁ + b₁) W₂ + b₂
   = X (W₁W₂) + (b₁W₂ + b₂)
   = X W_combined + b_combined
```

**No matter how many linear layers you stack, the result is always a single
linear transformation.** Non-linear activations break this collapse.

### ReLU — Rectified Linear Unit

```
f(z) = max(0, z)

f'(z) = 1   if z > 0
         0   if z ≤ 0
```

**Derivative**: A simple step function. For positive inputs, gradient flows
freely (f'=1). For negative inputs, gradient is zero — the neuron is
temporarily "off". This creates *sparsity*: on any given input, typically
only ~50% of neurons are active.

**The Dying ReLU problem**: If a neuron's weight update pushes all its inputs
into the negative regime permanently, f'(z)=0 forever — the neuron is "dead"
and contributes nothing. Solution: He initialisation, careful learning rate,
or LeakyReLU.

### Sigmoid

```
f(z) = 1 / (1 + e^{-z})

f'(z) = f(z) · (1 - f(z))
```

Maps ℝ → (0, 1). The output is interpretable as a *probability*.

**Saturation problem**: For |z| >> 0, f'(z) ≈ 0. Gradients vanish when
backpropagating through many sigmoid layers. This is why deep networks
rarely use sigmoid in hidden layers.

### Softmax

```
f(z)_k = exp(z_k) / Σⱼ exp(z_j)    for k = 1..K
```

Converts K real numbers into a probability distribution:
- All outputs ∈ (0, 1)
- All outputs sum to 1

**Numerical stability**: Subtract `max(z)` before computing exponentials:

```
f(z)_k = exp(z_k - max(z)) / Σⱼ exp(z_j - max(z))
```

This is identical mathematically (the constant cancels) but prevents overflow.

---

## 4. The Forward Pass

The forward pass evaluates the network left-to-right for a given input X.

```
A₀ = X                      (input, no transformation)

Z₁ = A₀ @ W₁ + b₁          (linear)
A₁ = f₁(Z₁)                (activation)

Z₂ = A₁ @ W₂ + b₂
A₂ = f₂(Z₂)

...

Zₗ = Aₗ₋₁ @ Wₗ + bₗ
Aₗ = fₗ(Zₗ)                 (final output = ŷ)

L = loss(y, ŷ)
```

**Caching**: During the forward pass, each layer stores `(A_prev, Z)` in a
cache. This is necessary because the backward pass needs them to compute
gradients. Storing them avoids recomputing the entire forward pass during backprop.

> **Code**: `neuralnet/layers.py` `DenseLayer.forward()` — the line
> `self._cache = {"A_prev": A_prev, "Z": Z}` stores the cache.

---

## 5. Loss Functions

### What is a Loss Function?

The loss `L(y, ŷ)` is a scalar measuring prediction error. We average over
m samples:

```
L = (1/m) Σᵢ ℓ(yᵢ, ŷᵢ)
```

The entire learning process is: **minimise L** with respect to all W and b.

### Mean Squared Error (Regression)

```
L = (1/2m) Σᵢ (yᵢ - ŷᵢ)²

dL/dŷ = (ŷ - y) / m
```

The factor 1/2 makes the derivative clean (the 2 from the power cancels).

### Binary Cross-Entropy (Binary Classification)

```
L = -(1/m) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

dL/dŷ = (1/m) [-(y/ŷ) + (1-y)/(1-ŷ)]
```

**Intuition**: If the true label is 1 and we predict ŷ=0.01, the loss is
`-log(0.01) ≈ 4.6` — very large. If we predict ŷ=0.99, the loss is
`-log(0.99) ≈ 0.01` — tiny. The loss penalises confident wrong answers harshly.

**Why not MSE for classification?** With sigmoid output, MSE creates a
loss landscape with many flat regions (because σ'(z)≈0 saturates the
gradient). Cross-entropy combined with sigmoid produces a perfectly clean
gradient: `dL/dz = (ŷ - y)/m`, which never vanishes.

### Categorical Cross-Entropy (Multi-Class)

```
L = -(1/m) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
  = -(1/m) Σᵢ log(ŷᵢ,true_class)    (since y is one-hot)

dL/dŷ = -(1/m) y / ŷ    (element-wise division)
```

### The Softmax + CCE Miracle

When Softmax output feeds into CCE, the combined gradient simplifies
beautifully. The full derivation:

Let `a_k = softmax(z)_k`. CCE loss for one sample: `L = -Σₖ yₖ log(aₖ)`.

```
dL/dzⱼ = Σₖ (dL/daₖ) · (daₖ/dzⱼ)

dL/daₖ = -yₖ/aₖ

daₖ/dzⱼ = aₖ(δₖⱼ - aⱼ)     (Softmax Jacobian)

dL/dzⱼ = Σₖ (-yₖ/aₖ) · aₖ(δₖⱼ - aⱼ)
        = Σₖ -yₖ(δₖⱼ - aⱼ)
        = -yⱼ + aⱼ Σₖ yₖ
        = aⱼ - yⱼ         (since Σₖ yₖ = 1 for one-hot)
```

**Final result**:
```
dL/dZ = (A - Y) / m
```

This is remarkably clean: the gradient is simply the difference between the
predicted probability and the one-hot true label. No Jacobian computation
needed. This is why we fuse them in `SoftmaxCCE`.

---

## 6. Gradient Descent

### The Optimisation Problem

We want to find W* that minimises L:

```
W* = argmin_W L(W)
```

L is a function of potentially millions of parameters — we cannot find the
minimum analytically. Instead, we use iterative gradient descent.

### Intuition: Walking Downhill

Imagine the loss surface as a hilly landscape. You're standing at some point
(your current W). The gradient ∇L tells you which direction is "uphill".
To descend, you step in the *opposite* direction:

```
W ← W - η · ∇_W L
```

where η (eta) is the **learning rate** — the step size.

### Why Gradient Descent Works

The first-order Taylor expansion of L around W:

```
L(W + δ) ≈ L(W) + ∇L · δ
```

If we choose `δ = -η · ∇L` (gradient descent direction):

```
L(W + δ) ≈ L(W) - η · ||∇L||²
```

Since `η > 0` and `||∇L||² ≥ 0`, the loss *decreases* (or stays the same).
Gradient descent is guaranteed to reduce loss on each step (for small enough η).

### Stochastic vs. Mini-Batch vs. Full-Batch

