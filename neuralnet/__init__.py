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
