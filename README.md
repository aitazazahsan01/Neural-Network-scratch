<div align="center">

# 🧠 Neural Network from Scratch

**A complete, mathematically documented neural network library built with Python + NumPy only.**

*No PyTorch. No TensorFlow. No Keras. Every gradient, every weight update, written by hand.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![MNIST Accuracy](https://img.shields.io/badge/MNIST%20Accuracy-97.03%25-brightgreen?style=flat-square)](#results)
[![Gradients Verified](https://img.shields.io/badge/Gradients-Numerically%20Verified-blueviolet?style=flat-square)](#verification)

</div>

---

## What Is This?

This project implements a fully working neural network **library** from the ground up — not just a single script, but a modular framework with the same structure as PyTorch or Keras, except every internal calculation is written in plain NumPy and thoroughly documented with its mathematical derivation.

The goal is **deep understanding**, not just working code. Every file answers the question: *why does this formula look like this?*

---

## Architecture Diagram

![Neural Network Architecture](assets/nn_architecture.jpg)

A **Dense layer** stacks `n_out` neurons side-by-side. For a batch of `m` samples:

```
Z = X @ W + b        →    A = f(Z)
(m, n_in) @ (n_in, n_out) + (1, n_out)  =  (m, n_out)
```

Every input connects to every output neuron — hence "fully connected" or "dense."

---

## Backpropagation — The Chain Rule in Action

![Backpropagation Flow](assets/backprop_flow.jpg)

After the forward pass computes the loss `L`, gradients flow **backwards** through every layer using the chain rule:

```
dZ   = dA ⊙ f'(Z)          ← gradient through activation
dW   = Aᵀ · dZ             ← gradient for weights   (Aᵀ is prev layer output)
db   = Σ dZ                 ← gradient for biases
dA_prev = dZ · Wᵀ          ← gradient to pass to layer below
```

> **Key insight:** `loss.gradient()` already returns the mean gradient (÷m), so the layer backward receives dZ pre-scaled — **no second division by m**.

---

## Project Structure

```
NN_Scratch/
│
├── 📦 neuralnet/                   ← Core library (NumPy only)
│   ├── tensor.py                   ← dtype utilities (float64 everywhere)
│   ├── initializers.py             ← Zero · Random · Xavier · He
│   ├── activations.py              ← ReLU · LeakyReLU · Sigmoid · Tanh · Softmax
│   ├── losses.py                   ← MSE · BCE · CCE · SoftmaxCCE (fused)
│   ├── layers.py                   ← DenseLayer · DropoutLayer  ← THE HEART
│   ├── optimizers.py               ← SGD → Momentum → RMSProp → Adam
│   ├── network.py                  ← Sequential model (training loop)
│   └── utils/
│       ├── data_utils.py           ← split · batch · one-hot · normalise
│       ├── metrics.py              ← accuracy · precision · recall · F1
│       └── visualizer.py          ← loss curves · decision boundary · weight histograms
│
├── 📂 examples/
│   ├── 01_xor_problem.py           ← Proves backprop works (non-linearity proof)
│   ├── 02_binary_classification.py ← Overfitting demo + optimizer comparison
│   ├── 03_mnist_digits.py          ← Real digit recognition, 97% accuracy
│   └── gradient_check.py          ← Numerical gradient verification
│
