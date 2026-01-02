import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import numpy as np

from neuralnet import Sequential, DenseLayer, DropoutLayer
from neuralnet.utils.data_utils import train_val_split, normalize, one_hot_encode
from neuralnet.utils.metrics import classification_report_nn

np.random.seed(42)

print("Loading MNIST...")
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_full = mnist.data.astype(float)
y_full = mnist.target.astype(int)
print(f"Loaded {X_full.shape[0]} samples")

# Use 20k subset
idx   = np.random.permutation(len(X_full))[:20000]
X_sub = X_full[idx]
y_sub = y_full[idx]

X_train_raw, X_val_raw, y_train_int, y_val_int = train_val_split(X_sub, y_sub, val_fraction=0.15, seed=42)
X_train, x_min, x_max = normalize(X_train_raw)
