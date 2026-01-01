import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.datasets import make_classification

from neuralnet import Sequential, DenseLayer, DropoutLayer
