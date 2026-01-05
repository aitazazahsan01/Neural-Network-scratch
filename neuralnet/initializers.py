"""
neuralnet/initializers.py
=========================
Weight and bias initialisation strategies.

WHY INITIALISATION MATTERS
---------------------------
Before training begins every weight W must be assigned a starting value.
This choice has a dramatic effect on whether the network can learn at all.

Consider what happens during the forward pass:

    Z = X @ W + b          (pre-activation, shape [m, n_out])
    A = activation(Z)

If W is *too large*, |Z| grows exponentially through each layer →
activations saturate (sigmoid/tanh near 0 or 1, gradient ≈ 0) →
gradients vanish in the lower layers → training stalls.

If W is *too small*, |Z| shrinks exponentially → activations near 0 →
gradients also collapse → same problem.

The goal is to keep the variance of activations *roughly constant* across
every layer so that gradients flow without exploding or vanishing.

VARIANCE ANALYSIS (Xavier initialisation)
