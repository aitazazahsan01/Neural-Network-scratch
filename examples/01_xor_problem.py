"""
examples/01_xor_problem.py
==========================
Demonstration: solving the XOR problem with a neural network.

THE XOR PROBLEM — WHY IT MATTERS
----------------------------------
XOR (exclusive-OR) is historically significant in neural network research.
In 1969, Minsky & Papert proved that a *single-layer* (linear) perceptron
cannot solve XOR because XOR is not linearly separable.

The truth table:
    x1   x2   XOR
    ──   ──   ───
     0    0    0
     0    1    1
     1    0    1
     1    1    0

On a 2-D plot, no single straight line can separate the 1s from the 0s.
Only by adding a hidden layer (a non-linear transformation) can a network
learn the XOR function.

This example proves that backpropagation with non-linear activations
can solve problems that linear models fundamentally cannot.

WHAT TO OBSERVE
