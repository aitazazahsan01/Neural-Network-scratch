"""
neuralnet/tensor.py
===================
Thin NumPy wrapper providing consistent dtype management and shape utilities
used throughout the library.

WHY THIS MODULE EXISTS
----------------------
NumPy operations sometimes silently change dtypes (e.g., integer division
producing int64 arrays, or mixing float32/float64 mid-computation). All
neural-network math here runs in float64 for numerical precision. This
