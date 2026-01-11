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
