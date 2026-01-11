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

