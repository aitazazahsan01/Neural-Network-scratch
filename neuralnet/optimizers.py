"""
neuralnet/optimizers.py
=======================
Gradient-based optimisation algorithms.

WHAT DOES AN OPTIMIZER DO?
---------------------------
After backpropagation computes the gradient dL/dW for every layer,
the optimizer decides *how* to update the parameters W.

The simplest update rule (vanilla SGD):

    W ← W - lr * dW

But this naive rule has problems in practice:
    1. Noisy gradients from mini-batches cause erratic updates.
    2. The same learning rate for all parameters is often suboptimal.
    3. Flat regions ("saddle points") slow learning dramatically.

The optimizers below each address these problems progressively.

LEARNING RATE (η / lr)
-----------------------
The learning rate controls step size in parameter space.

    Too high  → overshoots minima, loss diverges or oscillates.
    Too low   → converges very slowly, gets stuck in local minima.

Finding a good lr is often the most impactful hyperparameter choice.
Typical values: 1e-4 to 1e-1 depending on optimizer and architecture.

Implemented optimizers
-----------------------
1. SGD (Stochastic Gradient Descent)
   W ← W - lr * dW
   Baseline; works for simple problems but noisy.

2. SGD with Momentum
   v ← β*v + (1-β)*dW      (exponential moving average of gradients)
   W ← W - lr * v
   Reduces oscillations, speeds up convergence in consistent directions.

3. RMSProp (Root Mean Square Propagation)
   s ← β*s + (1-β)*dW²
   W ← W - lr * dW / √(s + ε)
   Adapts lr per-parameter; larger s → smaller effective step.
   Excellent for non-stationary objectives (e.g., RNN training).

4. Adam (Adaptive Moment Estimation)
   m ← β₁*m + (1-β₁)*dW          (1st moment / momentum)
