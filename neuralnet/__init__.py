"""
neuralnet/__init__.py
=====================
Public API for the neuralnet package.

Usage
-----
    from neuralnet import Sequential, DenseLayer, DropoutLayer
    from neuralnet import Adam, SGD
    from neuralnet.utils import plot_history, plot_decision_boundary

Design philosophy
-----------------
Every public class and function in this package is implemented from scratch
using NumPy. No high-level ML frameworks are used internally.
The goal is complete transparency into how neural networks work.
"""

from .network      import Sequential
from .layers       import DenseLayer, DropoutLayer
from .activations  import (
    Linear, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax,
    get_activation,
)
from .losses       import (
    MSE, BinaryCrossEntropy, CategoricalCrossEntropy, SoftmaxCCE,
    get_loss,
)
from .optimizers   import (
    SGD, SGDMomentum, RMSProp, Adam,
    get_optimizer,
)
from .initializers import (
    ZeroInit, RandomNormal, XavierUniform, XavierNormal, HeNormal, HeUniform,
    get_initializer,
)
from . import utils

__version__ = "1.0.0"
__all__ = [
    # Model
    "Sequential",
    # Layers
    "DenseLayer", "DropoutLayer",
    # Activations
    "Linear", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax",
    "get_activation",
    # Losses
    "MSE", "BinaryCrossEntropy", "CategoricalCrossEntropy", "SoftmaxCCE",
    "get_loss",
    # Optimizers
    "SGD", "SGDMomentum", "RMSProp", "Adam",
    "get_optimizer",
    # Initialisers
    "ZeroInit", "RandomNormal", "XavierUniform", "XavierNormal",
    "HeNormal", "HeUniform",
    "get_initializer",
    # Utils
    "utils",
]
