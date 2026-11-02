"""Activation functions, on the whole batch: forward a = f(z), backward dL/dz from dL/da."""

import numpy as np

ACTIVATIONS = ("relu", "leaky_relu", "elu", "sigmoid", "tanh", "linear", "softmax")
# activations with a parameter, and its default
ALPHA = {"leaky_relu": 0.01, "elu": 1.0}


def forward(name, z, alpha=0.0):
    if name == "relu":
        return np.maximum(z, 0)
    if name == "leaky_relu":
        return np.where(z > 0, z, alpha * z)
    if name == "elu":
        # expm1 of min(z, 0) only: exp(z) would overflow for large z
        return np.where(z > 0, z, alpha * np.expm1(np.minimum(z, 0)))
    if name == "sigmoid":
        # = 1 / (1 + exp(-z)), without overflow for very negative z
        return 0.5 * (1 + np.tanh(0.5 * z))
    if name == "tanh":
        return np.tanh(z)
    if name == "softmax":
        # minus the row max: same result, and exp() stays finite
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return z


def backward(name, z, a, grad, alpha=0.0):
    """Gradient wrt z, from z, a = f(z) and the gradient wrt a."""
    if name == "relu":
        return grad * (z > 0)
    if name == "leaky_relu":
        return grad * np.where(z > 0, 1.0, alpha).astype(z.dtype)
    if name == "elu":
        return grad * np.where(z > 0, 1.0, a + alpha).astype(z.dtype)
    if name == "sigmoid":
        return grad * a * (1 - a)
    if name == "tanh":
        return grad * (1 - a * a)
    if name == "softmax":
        # Jacobian-vector product, row by row, without building the Jacobian
        return a * (grad - (grad * a).sum(axis=1, keepdims=True))
    return grad
