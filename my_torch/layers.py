"""Fully connected (dense) layer: a = f(x @ W + b), with optional inverted dropout.

Everything is computed on the whole batch at once (one matrix product per
layer), never sample by sample.
"""

import numpy as np

from .activations import get_activation
from .init import init_weights


class Dense:
    def __init__(self, n_in, n_out, activation, rng, init="he", dropout=0.0, dtype=np.float32,
                 activation_params=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = get_activation(activation, **(activation_params or {}))
        self.init = init
        self.dropout = float(dropout)
        self.W = init_weights(init, n_in, n_out, rng).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.dW = None
        self.db = None
        # values kept from the forward pass for backward()
        self.x = self.z = self.a = self.mask = None

    def forward(self, x, training=False, rng=None):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.activation.forward(self.z)
        self.mask = None
        if training and self.dropout > 0:
            # inverted dropout: drop with probability p, scale the kept ones by 1/(1-p)
            # so that the expected output is the same with and without dropout
            keep = 1.0 - self.dropout
            self.mask = (rng.random(self.a.shape, dtype=self.a.dtype) < keep).astype(self.a.dtype) / keep
            return self.a * self.mask
        return self.a

    def backward(self, grad, input_grad=True):
        """Gradient wrt this layer's output -> gradient wrt its input. Stores dW, db.

        `input_grad=False` skips the gradient wrt the input (useless for the first layer).
        """
        if self.mask is not None:
            grad = grad * self.mask
        return self.backward_from_z(self.activation.backward(self.z, self.a, grad), input_grad)

    def backward_from_z(self, dz, input_grad=True):
        """Same as backward(), starting from the gradient wrt z (fused loss + activation)."""
        self.dW = self.x.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T if input_grad else None

    @property
    def n_params(self):
        return self.W.size + self.b.size
