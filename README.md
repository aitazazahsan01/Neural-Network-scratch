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
├── 📖 THEORY.md                    ← 14-section mathematical guide
├── requirements.txt
└── README.md
```

---

## Training Pipeline

```mermaid
flowchart TD
    A([Raw Data]) --> B[Preprocess\nStandardise · One-Hot · Split]
    B --> C[Build Model\nSequential + DenseLayer + Dropout]
    C --> D[Compile\nLoss + Optimizer]
    D --> E{Training Loop}

    E --> F[Shuffle Data]
    F --> G[Mini-Batch]
    G --> H[Forward Pass\nZ=XW+b, A=f Z]
    H --> I[Compute Loss\nL = loss y, ŷ]
    I --> J[Backward Pass\ndW, db via chain rule]
    J --> K[Optimizer Step\nW ← W - η·update]
    K --> L{More batches?}
    L -- Yes --> G
    L -- No --> M[Epoch Metrics\ntrain_loss, val_loss, acc]
    M --> N{More epochs?}
    N -- Yes --> F
    N -- No --> O([Trained Model])

    style A fill:#238636,color:#fff
    style O fill:#238636,color:#fff
    style I fill:#da3633,color:#fff
    style J fill:#8957e5,color:#fff
    style K fill:#1f6feb,color:#fff
```

---

## Optimizers — A Progressive Story

```mermaid
graph LR
    A[SGD\nW ← W - η·dW] -->|add momentum| B[SGD + Momentum\nv ← βv + dW\nW ← W - η·v]
    B -->|adaptive lr| C[RMSProp\ns ← βs + dW²\nW ← W - η·dW/√s]
    C -->|both together + bias fix| D[Adam ⭐\nm̂/√v̂ with bias correction]

    style A fill:#21262d,color:#8b949e,stroke:#30363d
    style B fill:#21262d,color:#e3b341,stroke:#d29922
    style C fill:#21262d,color:#58a6ff,stroke:#1f6feb
    style D fill:#238636,color:#fff,stroke:#2ea043
```

Each optimizer is implemented from scratch with its full derivation in [`optimizers.py`](neuralnet/optimizers.py).

---

## Activation Functions

```mermaid
graph LR
    subgraph "Hidden Layers"
        R[ReLU\nmax 0,z\nf'= 1 if z>0\nelse 0]
        LR[LeakyReLU\nαz if z≤0\nFixes dying ReLU]
        T[Tanh\ne^z - e^-z / e^z + e^-z\nZero-centred]
    end
    subgraph "Output Layer"
        S[Sigmoid\n1 / 1+e^-z\nBinary classification]
        SM[Softmax\ne^zk / Σe^zj\nMulti-class]
    end

    style R fill:#1f6feb,color:#fff,stroke:none
    style LR fill:#388bfd,color:#fff,stroke:none
    style T fill:#8957e5,color:#fff,stroke:none
    style S fill:#da3633,color:#fff,stroke:none
    style SM fill:#238636,color:#fff,stroke:none
```

---

## Results

### ✅ Gradient Check — All 18 Parameters Verified

Every gradient is verified against numerical finite differences before trusting the backprop implementation:

| Loss + Activation | Relative Difference | Status |
|---|---|---|
| MSE + Tanh + Linear | `~5e-11` | ✅ PASS |
| BCE + ReLU + Tanh + Sigmoid | `~4e-11` | ✅ PASS |
| SoftmaxCCE + ReLU + Tanh + Linear | `~6e-11` | ✅ PASS |

> Numerical gradient: `df/dθ ≈ [f(θ+ε) − f(θ−ε)] / 2ε` — accurate to machine precision.

---

### 🔁 XOR Problem — Proving Non-Linearity is Necessary

XOR is not linearly separable. A single-layer model **mathematically cannot** solve it:

| Model | Final Loss | Accuracy |
|---|---|---|
| Single layer (Linear) | 0.6931 | 50% ❌ (stuck at random chance) |
| 2-layer (Hidden: Tanh) | **0.000050** | **100%** ✅ |

---

### 🔵 Binary Classification — 1,000 Synthetic Samples

| Metric | Value |
|---|---|
| Architecture | `2 → Dense(16,ReLU) → Dropout(0.3) → Dense(8,ReLU) → Dense(1,Sigmoid)` |
| Optimizer | Adam, lr=0.01 |
| Val Accuracy | **96.5%** |
| Val F1-Score | **0.965** |

---

### 🔢 MNIST Digit Recognition — 20,000 Samples

| Metric | Value |
|---|---|
| Architecture | `784 → Dense(256) → Dropout → Dense(128) → Dropout → Dense(10)` |
| Loss | SoftmaxCCE (fused, numerically stable) |
| Optimizer | Adam, lr=0.001 |
| Epochs | 20 |
| **Val Accuracy** | **97.03%** 🎯 |

Per-class breakdown:

| Digit | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|
| 0 | 0.969 | 0.990 | **0.980** |
| 1 | 0.981 | 0.978 | **0.979** |
| 2 | 0.970 | 0.960 | 0.965 |
| 3 | 0.972 | 0.958 | 0.965 |
| 4 | 0.954 | 0.975 | 0.964 |
| 5 | 0.960 | 0.960 | 0.960 |
| 6 | 0.980 | 0.987 | **0.984** |
| 7 | 0.977 | 0.986 | **0.981** |
| 8 | 0.973 | 0.951 | 0.962 |
| 9 | 0.964 | 0.954 | 0.959 |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/NN_Scratch.git
cd NN_Scratch
pip install -r requirements.txt

# Verify gradients first (optional but educational)
python -m examples.gradient_check

# Run examples in order
python -m examples.01_xor_problem          # ~2 seconds
python -m examples.02_binary_classification # ~30 seconds
python -m examples.03_mnist_digits          # ~5 minutes (downloads MNIST once)
```

---

## Usage

```python
from neuralnet import Sequential, DenseLayer, DropoutLayer
from neuralnet.utils.data_utils import train_val_split, standardize, one_hot_encode
from neuralnet.utils.visualizer import plot_history

# Build
model = Sequential([
    DenseLayer(256, activation="relu"),
    DropoutLayer(rate=0.3),
    DenseLayer(128, activation="relu"),
    DenseLayer(10,  activation="linear"),  # + SoftmaxCCE
])

# Compile
model.compile(loss="softmax_cce", optimizer="adam", learning_rate=0.001)
model.summary(input_shape=(784,))  # shows param counts

# Train
history = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, Y_val),
)

# Evaluate
model.evaluate(X_val, Y_val)
plot_history(history, title="My Model")
```

---

## Weight Initialisation

| Strategy | Formula | Best For |
|---|---|---|
| `zeros` | `0` | Biases only — **never weights** |
| `random_normal` | `N(0, σ)` | Baseline / experiments |
| `xavier_uniform` | `U(−√6/(fi+fo), +√6/(fi+fo))` | Sigmoid / Tanh |
| `xavier_normal` | `N(0, √2/(fi+fo))` | Sigmoid / Tanh |
| `he_normal` ⭐ | `N(0, √2/fan_in)` | **ReLU (default)** |
| `he_uniform` | `U(−√6/fan_in, +√6/fan_in)` | ReLU variant |

> Zeros fail because all neurons receive identical gradients → zero learning.
> He init compensates for ReLU zeroing ~50% of neurons, keeping variance stable across layers.

---

## Theory Guide

[`THEORY.md`](THEORY.md) is a standalone 26 KB mathematical companion covering:

| Section | What You'll Learn |
|---|---|
| 1. From Biology to Math | Perceptron model, weights, bias |
| 2. The Dense Layer | Matrix multiplication explained |
| 3. Activation Functions | Why non-linearity matters |
| 4. The Forward Pass | Chaining layers, caching |
| 5. Loss Functions | MSE, BCE, CCE with derivations |
| 6. Gradient Descent | Why it works (Taylor expansion proof) |
| 7. Backpropagation | Full chain-rule derivation |
| **8. Worked Example** | **Step-by-step numerical trace through XOR** |
| 9. Initialisation | Variance analysis, Xavier vs He |
| 10. Optimizers | SGD → Adam with bias-correction proof |
| 11. Overfitting | Dropout, L2, early stopping |
| 12. Learning Rate | What goes wrong at each extreme |
| 13. Training Internals | What happens each epoch |
| 14. Code ↔ Math Table | Every formula mapped to its exact line of code |

---

## Dependencies

| Package | Role |
|---|---|
| `numpy` | All matrix math — the only ML dependency |
| `matplotlib` | Plotting training curves and decision boundaries |
| `scikit-learn` | **Data generation only** in examples (not used in the library itself) |

---

<div align="center">

*Built for learning. Every line of code is a lesson.*

**Start with [`THEORY.md`](THEORY.md) → then read [`layers.py`](neuralnet/layers.py)**

</div>
