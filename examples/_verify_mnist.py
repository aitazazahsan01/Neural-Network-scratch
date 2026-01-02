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

