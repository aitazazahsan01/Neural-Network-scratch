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


# ---------------------------------------------------------------------------
# Sequential model
# ---------------------------------------------------------------------------

class Sequential:
    """A linear stack of layers forming a feedforward neural network.

    Parameters
    ----------
    layers : list, optional
        Pre-existing list of layer objects. You can also add layers
        incrementally with `.add()`.

    Examples
    --------
    >>> from neuralnet import Sequential, DenseLayer
    >>> model = Sequential([
    ...     DenseLayer(64, activation="relu"),
    ...     DenseLayer(1,  activation="sigmoid"),
    ... ])
    >>> model.compile(loss="bce", optimizer="adam")
    >>> history = model.fit(X_train, y_train, epochs=100)
    """

    def __init__(self, layers: list | None = None) -> None:
        self.layers: list = layers if layers is not None else []
        self._loss: Loss | None = None
        self._optimizer: Optimizer | None = None
        self._compiled = False
        self._history: dict = {
            "train_loss": [],
            "val_loss":   [],
            "train_acc":  [],
            "val_acc":    [],
        }

    # ------------------------------------------------------------------
    # Building the model
    # ------------------------------------------------------------------

    def add(self, layer) -> "Sequential":
        """Append a layer to the network.

        Returns self so calls can be chained:
            model.add(DenseLayer(64)).add(DenseLayer(1))
        """
        self.layers.append(layer)
        return self

    def compile(
        self,
        loss: str | Loss,
        optimizer: str | Optimizer = "adam",
        learning_rate: float | None = None,
    ) -> None:
        """Set the loss function and optimizer.

        Parameters
        ----------
        loss : str or Loss instance
            E.g. "mse", "bce", "cce", "softmax_cce".
        optimizer : str or Optimizer instance
            E.g. "sgd", "momentum", "rmsprop", "adam".
        learning_rate : float, optional
            If provided, overrides the optimizer's default lr.
        """
        self._loss = get_loss(loss)

        if isinstance(optimizer, str):
            kwargs = {}
            if learning_rate is not None:
                kwargs["lr"] = learning_rate
            self._optimizer = get_optimizer(optimizer, **kwargs)
        else:
            self._optimizer = optimizer
            if learning_rate is not None:
                self._optimizer.lr = learning_rate

        self._compiled = True

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Chain-call forward() through all layers.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
        training : bool
            Passed to layers that behave differently during training (Dropout).

        Returns
        -------
        output : np.ndarray, shape (m, n_output)
        """
        A = to_array(X)
        for layer in self.layers:
            A = layer.forward(A, training=training)
        return A

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Backpropagate loss gradient through all layers (reversed order).

        Computes and stores .dW and .db on every layer.

        Parameters
        ----------
