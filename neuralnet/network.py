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
        y_true : np.ndarray
        y_pred : np.ndarray  — output of forward pass
        """
        # Compute initial gradient dL/dŷ from the loss function
        dA = self._loss.gradient(y_true, y_pred)

        # Special case: SoftmaxCCE returns dL/dZ (pre-activation gradient)
        # so the last layer must skip its activation backward.
        # We signal this by temporarily replacing the last Dense layer's
        # activation backward with identity during this specific backward.
        if isinstance(self._loss, SoftmaxCCE):
            # dA is already dL/dZ for the output layer's Linear activation
            # Iterate layers in reverse, but skip the last layer's activation
            dA = self._backward_layers(dA, skip_last_activation=True)
        else:
            dA = self._backward_layers(dA, skip_last_activation=False)

    def _backward_layers(
        self,
        dA: np.ndarray,
        skip_last_activation: bool,
    ) -> np.ndarray:
        """Internal: propagate dA backwards through the layer list."""
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(reversed(self.layers)):
            actual_idx = last_idx - i

            if skip_last_activation and actual_idx == last_idx:
                # SoftmaxCCE already computed dL/dZ so we skip activation step.
                # Temporarily inject dZ directly into the Dense layer's backward
                # by calling _backward_skip_activation.
                if isinstance(layer, DenseLayer):
                    dA = self._dense_backward_no_activation(layer, dA)
                else:
                    dA = layer.backward(dA)
            else:
                dA = layer.backward(dA)

        return dA

    def _dense_backward_no_activation(
        self, layer: DenseLayer, dZ: np.ndarray
    ) -> np.ndarray:
        """Backward through a Dense layer when dZ is provided directly.

        Used when SoftmaxCCE already computed dL/dZ so we skip f'(Z).
        SoftmaxCCE.gradient() already divides by m, so dZ carries the 1/m
        factor. Do NOT divide by m again here.
        """
        A_prev = layer._cache["A_prev"]

        layer.dW = A_prev.T @ dZ                    # dZ already has 1/m
        layer.db = dZ.sum(axis=0, keepdims=True)    # sum, not mean (1/m in dZ)
        dA_prev = dZ @ layer.W.T

        return dA_prev.astype(DTYPE)

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> None:
        """Apply the optimizer update to every trainable layer's parameters."""
        for layer_id, layer in enumerate(self.layers):
            if not layer.params:
                continue  # skip non-trainable layers (Dropout)
            updated = self._optimizer.update(layer_id, layer.params, layer.grads)
            layer.set_params(updated)
        self._optimizer.step()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple | None = None,
        verbose: int = 1,
        verbose_every: int = 10,
    ) -> dict:
        """Train the network.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
        y : np.ndarray, shape (m,) or (m, n_classes)
        epochs : int
            Number of full passes through the training data.
        batch_size : int
            Samples per mini-batch. Use -1 for full-batch gradient descent.
        validation_data : tuple (X_val, y_val), optional
            If provided, compute val_loss and val_acc after each epoch.
        verbose : int
            0 = silent, 1 = progress every `verbose_every` epochs.
        verbose_every : int
            Print interval when verbose=1. Default every 10 epochs.

        Returns
        -------
        history : dict
            Keys: "train_loss", "val_loss", "train_acc", "val_acc".
            Each value is a list with one entry per epoch.

        WHAT HAPPENS EACH EPOCH?
        -------------------------
        1. Shuffle the training data (so mini-batches are different each time).
        2. Split into mini-batches.
        3. For each mini-batch:
            a. Forward pass → ŷ
            b. Loss → L
            c. Backward pass → dW, db for every layer
            d. Optimizer step → update W, b
        4. Compute epoch-level train/val metrics.

        WHY SHUFFLE?
        ------------
        If we always feed data in the same order, the model might overfit
        to the order. Shuffling ensures every mini-batch has a different
        composition, making the gradient estimate more representative.
        """
        assert self._compiled, "Call model.compile() before model.fit()."

        X = to_array(X)
        y = to_array(y)
        m = X.shape[0]

        if batch_size == -1:
            batch_size = m   # full-batch GD

        history = {k: [] for k in self._history}

        t0 = time.time()

        for epoch in range(1, epochs + 1):

            # 1. Shuffle
            idx = np.random.permutation(m)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            # 2. Mini-batch loop
            epoch_losses = []
            for start in range(0, m, batch_size):
                end     = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # a. Forward
                y_pred = self.forward(X_batch, training=True)

                # b. Loss
                loss_val = self._loss.compute(y_batch, y_pred)
                epoch_losses.append(loss_val)

                # c. Backward
                self.backward(y_batch, y_pred)

                # d. Optimizer update
                self._optimizer_step()

            # 3. Epoch metrics
            train_loss = float(np.mean(epoch_losses))
            history["train_loss"].append(train_loss)

            # Training accuracy
            train_acc = self._compute_accuracy(X, y)
            history["train_acc"].append(train_acc)
