import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.datasets import make_classification

from neuralnet import Sequential, DenseLayer, DropoutLayer
from neuralnet.utils.data_utils import train_val_split, standardize
from neuralnet.utils.metrics import classification_report_nn

np.random.seed(0)

X_raw, y_raw = make_classification(
    n_samples=1000, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, class_sep=1.2, random_state=42
)
y_raw = y_raw.reshape(-1, 1).astype(float)
