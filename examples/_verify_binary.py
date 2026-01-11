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

X_train_raw, X_val_raw, y_train, y_val = train_val_split(X_raw, y_raw, val_fraction=0.2, seed=42)
X_train, mean, std = standardize(X_train_raw)
X_val, _, _        = standardize(X_val_raw, X_ref=X_train_raw)

model = Sequential([
    DenseLayer(16, activation='relu', name='Dense-1'),
    DropoutLayer(rate=0.3, name='Dropout'),
    DenseLayer(8,  activation='relu', name='Dense-2'),
    DenseLayer(1,  activation='sigmoid', name='Output'),
])
model.compile(loss='bce', optimizer='adam', learning_rate=0.01)
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1, verbose_every=50,
)

result = model.evaluate(X_val, y_val)
y_val_pred = model.predict(X_val)
classification_report_nn(y_val, y_val_pred, class_names=['Class 0', 'Class 1'])
print(f"\nFinal val accuracy: {result['accuracy']:.4f}")
