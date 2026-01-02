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
X_val,   _,     _     = normalize(X_val_raw, X_ref=X_train_raw)
Y_train = one_hot_encode(y_train_int, n_classes=10)
Y_val   = one_hot_encode(y_val_int,   n_classes=10)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

np.random.seed(42)
model = Sequential([
    DenseLayer(256, activation='relu', name='Dense-1'),
    DropoutLayer(rate=0.3, name='Dropout-1'),
    DenseLayer(128, activation='relu', name='Dense-2'),
    DropoutLayer(rate=0.2, name='Dropout-2'),
    DenseLayer(10,  activation='linear', name='Output'),
])
