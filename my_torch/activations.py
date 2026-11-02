"""Activation functions, vectorised over the whole batch.

Each activation gives its output `a = f(z)` and, for backpropagation, the
gradient of the loss with respect to `z` from the gradient with respect to `a`.
"""

import numpy as np


class Activation:
    name = ""

    def forward(self, z):
        raise NotImplementedError

    def backward(self, z, a, grad):
        """Gradient wrt z, given z, a = f(z) and the gradient wrt a."""
        raise NotImplementedError

    def params(self):
        """Settings saved in the network file (empty for most activations)."""
        return {}


class ReLU(Activation):
    name = "relu"

    def forward(self, z):
        return np.maximum(z, 0)

    def backward(self, z, a, grad):
        return grad * (z > 0)


class LeakyReLU(Activation):
    name = "leaky_relu"

    def __init__(self, alpha=0.01):
        self.alpha = float(alpha)

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, z, a, grad):
        return grad * np.where(z > 0, 1.0, self.alpha).astype(z.dtype)

    def params(self):
        return {"alpha": self.alpha}


class ELU(Activation):
    name = "elu"

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def forward(self, z):
        # expm1 on min(z, 0) only: exp(z) would overflow for large positive z
        return np.where(z > 0, z, self.alpha * np.expm1(np.minimum(z, 0)))

    def backward(self, z, a, grad):
        return grad * np.where(z > 0, 1.0, a + self.alpha).astype(z.dtype)

    def params(self):
        return {"alpha": self.alpha}


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, z):
        # same function as 1 / (1 + exp(-z)), without overflow for very negative z
        return 0.5 * (1 + np.tanh(0.5 * z))

    def backward(self, z, a, grad):
        return grad * a * (1 - a)


class Tanh(Activation):
    name = "tanh"

    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a, grad):
        return grad * (1 - a * a)


class Linear(Activation):
    name = "linear"

    def forward(self, z):
        return z

    def backward(self, z, a, grad):
        return grad


class Softmax(Activation):
    name = "softmax"

    def forward(self, z):
        # subtracting the row max changes nothing mathematically but keeps exp() finite
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def backward(self, z, a, grad):
        # Jacobian-vector product of softmax, row by row, without building the Jacobian
        return a * (grad - (grad * a).sum(axis=1, keepdims=True))


ACTIVATIONS = {cls.name: cls for cls in (ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Linear, Softmax)}


def get_activation(name, **params):
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation '{name}' (choices: {', '.join(ACTIVATIONS)})")
    return ACTIVATIONS[name](**params)
