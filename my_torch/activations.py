"""Activation functions, on the whole batch: forward a = f(z), backward dL/dz from dL/da.

A registry name -> (forward, backward) instead of an if/elif chain: adding an
activation is one entry, and the configuration is validated against the same
dict. The backward functions take a = f(z) as well as z, so the forward value is
reused instead of recomputed.
"""

import numpy as np

from .errors import lookup


def relu(z, alpha):
    return np.maximum(z, 0)


def relu_backward(z, a, grad, alpha):
    return grad * (z > 0)


def leaky_relu(z, alpha):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_backward(z, a, grad, alpha):
    return grad * np.where(z > 0, 1.0, alpha).astype(z.dtype)


def elu(z, alpha):
    # expm1 of min(z, 0) only: exp(z) would overflow for large z
    return np.where(z > 0, z, alpha * np.expm1(np.minimum(z, 0)))


def elu_backward(z, a, grad, alpha):
    return grad * np.where(z > 0, 1.0, a + alpha).astype(z.dtype)


def sigmoid(z, alpha):
    # = 1 / (1 + exp(-z)), without overflow for very negative z
    return 0.5 * (1 + np.tanh(0.5 * z))


def sigmoid_backward(z, a, grad, alpha):
    return grad * a * (1 - a)


def tanh(z, alpha):
    return np.tanh(z)


def tanh_backward(z, a, grad, alpha):
    return grad * (1 - a * a)


def linear(z, alpha):
    return z


def linear_backward(z, a, grad, alpha):
    return grad


def softmax(z, alpha):
    # minus the row max: same result, and exp() stays finite
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def softmax_backward(z, a, grad, alpha):
    # Jacobian-vector product, row by row, without building the Jacobian
    return a * (grad - (grad * a).sum(axis=1, keepdims=True))


ACTIVATIONS = {
    "relu": (relu, relu_backward),
    "leaky_relu": (leaky_relu, leaky_relu_backward),
    "elu": (elu, elu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh": (tanh, tanh_backward),
    "linear": (linear, linear_backward),
    "softmax": (softmax, softmax_backward),
}
# activations with a parameter, and its default
ALPHA = {"leaky_relu": 0.01, "elu": 1.0}


def forward(name, z, alpha=0.0):
    return lookup(ACTIVATIONS, name, "activation")[0](z, alpha)


def backward(name, z, a, grad, alpha=0.0):
    """Gradient wrt z, from z, a = f(z) and the gradient wrt a."""
    return lookup(ACTIVATIONS, name, "activation")[1](z, a, grad, alpha)
