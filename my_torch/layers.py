"""Fully connected layer a = f(x @ W + b), computed on the whole batch, with inverted dropout."""

import numpy as np

from . import activations
from .errors import lookup
from .initializers import INITS


class Dense:
    def __init__(self, n_in, n_out, spec, rng, dtype):
        """spec: normalized layer config {"size", "activation", ["alpha"], "dropout", "init"}."""
        self.activation, self.alpha, self.dropout = spec["activation"], spec.get("alpha", 0.0), spec["dropout"]
        self.W = lookup(INITS, spec["init"], "init")(n_in, n_out, rng).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.x = self.z = self.a = self.mask = self.dW = self.db = None

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, x, training=False, rng=None):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = activations.forward(self.activation, self.z, self.alpha)
        self.mask = None
        if training and self.dropout > 0:
            # inverted dropout: drop with probability p, scale the others by 1/(1-p),
            # so the expected output is unchanged and nothing needs fixing at inference
            keep = 1.0 - self.dropout
            self.mask = (rng.random(self.a.shape, dtype=self.a.dtype) < keep).astype(self.a.dtype) / keep
            return self.a * self.mask
        return self.a

    def backward(self, grad, input_grad=True):
        """Gradient wrt the output -> gradient wrt the input (None if not needed). Stores dW, db."""
        if self.mask is not None:
            grad = grad * self.mask
        return self.backward_from_z(activations.backward(self.activation, self.z, self.a, grad, self.alpha), input_grad)

    def backward_from_z(self, dz, input_grad=True):
        self.dW = self.x.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T if input_grad else None
