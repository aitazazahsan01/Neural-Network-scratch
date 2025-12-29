"""
examples/02_binary_classification.py
=====================================
Demonstration: binary classification on a synthetic 2-class dataset.

WHAT THIS EXAMPLE TEACHES
--------------------------
1. How to preprocess real (synthetic) data: standardise features.
2. How Sigmoid + BCE work together for binary classification.
3. How to visualise a 2-D decision boundary evolving during training.
4. Overfitting: how a model with too much capacity fits noise.
5. Dropout's effect on overfitting.
6. Comparison of all four optimizers: SGD → Momentum → RMSProp → Adam.

DATASET
-------
sklearn's make_classification with 2 features (so we can plot it):
    • 1000 samples, 2 informative features, 2 classes
    • Some overlap between classes (not perfectly separable)

ARCHITECTURE
------------
Input (2) → Dense(16, ReLU) → Dropout(0.3) → Dense(8, ReLU) → Dense(1, Sigmoid)
Loss: Binary Cross-Entropy
Optimizer: Adam (lr=0.01)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from neuralnet import Sequential, DenseLayer, DropoutLayer
from neuralnet.utils.data_utils import train_val_split, standardize
from neuralnet.utils.metrics import classification_report_nn
