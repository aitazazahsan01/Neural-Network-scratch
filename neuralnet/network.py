"""
neuralnet/network.py
====================
The Sequential model — orchestrates layers, training, and inference.

WHAT THIS MODULE DOES
----------------------
`Sequential` is the glue that connects all the pieces:

    1. Stores a list of layers in order.
    2. Runs the *forward pass*: chains A₀ → A₁ → ... → ŷ.
    3. Runs the *backward pass*: chains dAₗ → dAₗ₋₁ → ... → dA₀.
    4. Calls the *optimizer* to update W and b in every layer.
    5. Orchestrates the *training loop*: mini-batches, epochs, logging.

THE TRAINING LOOP (pseudocode)
-------------------------------
    for epoch in 1..epochs:
        shuffle training data
        for each mini-batch (X_batch, y_batch):
            ŷ      = forward(X_batch)          # forward pass
            L      = loss.compute(y, ŷ)        # loss
            dA     = loss.gradient(y, ŷ)       # initial gradient
            for layer in reversed(layers):     # backward pass
                dA = layer.backward(dA)
            for layer in layers:               # optimizer step
                optimizer.update(layer.W, layer.dW, ...)
        log train loss, val loss, metrics

OVERFITTING & GENERALISATION
------------------------------
During training, the model sees the training set repeatedly. Over many
epochs it can *memorise* the training data rather than learning general
patterns — this is overfitting:

    train_loss  ↓  (keeps improving)
    val_loss    ↑  (gets worse — the network isn't generalising)

Tracking validation loss separately (via `validation_data` argument)
lets you detect overfitting early and stop training.

Prevention strategies:
    • Less model capacity (fewer layers/neurons)
    • Dropout (DropoutLayer)
    • L2 regularisation (weight_decay parameter)
    • Early stopping (monitoring val_loss)
    • More training data
"""

import numpy as np
import time
from .tensor import DTYPE, to_array
from .losses import get_loss, Loss, SoftmaxCCE
from .optimizers import get_optimizer, Optimizer
from .layers import DenseLayer, DropoutLayer
from .utils.metrics import accuracy

