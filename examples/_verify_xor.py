import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import numpy as np

from neuralnet import Sequential, DenseLayer

np.random.seed(42)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# Linear model - should stall
model_lin = Sequential([DenseLayer(1, activation='sigmoid', name='Output')])
model_lin.compile(loss='bce', optimizer='adam', learning_rate=0.05)
h_lin = model_lin.fit(X, y, epochs=2000, batch_size=-1, verbose=0)
print("[Linear] Final loss:", round(h_lin['train_loss'][-1], 4),
      " acc:", round(h_lin['train_acc'][-1], 4),
      " (should be stuck near 0.5 acc)")

# 2-layer network - should solve XOR
np.random.seed(42)
model = Sequential([
    DenseLayer(4, activation='tanh', name='Hidden'),
    DenseLayer(1, activation='sigmoid', name='Output'),
])
model.compile(loss='bce', optimizer='adam', learning_rate=0.05)
h = model.fit(X, y, epochs=3000, batch_size=-1, verbose=1, verbose_every=500)

print("[2-Layer] Final loss:", round(h['train_loss'][-1], 6),
      " acc:", h['train_acc'][-1])

preds = model.predict(X)
