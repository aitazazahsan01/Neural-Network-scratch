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
