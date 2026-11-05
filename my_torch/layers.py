"""Fully connected layer a = f(x @ W + b), computed on the whole batch, with inverted dropout.

For an input of 0/1 values (one-hot encodings, such as a chessboard: 15 active
inputs out of 773 on average), x @ W is the sum of the rows of W whose input is
1: forward_binary() adds those rows instead of computing the matrix product.
Same values, added in the same order as the reference BLAS (exactly equal there),
and 2.4x faster for a whole prediction on such a machine (docs/benchmarks.md §7).
It is only used to predict: the training keeps the matrix product.
"""

import numpy as np

from . import activations
from .errors import lookup
from .initializers import INITS


def sum_active_rows(X, W):
    """X @ W for a 0/1 matrix X: each output row is the sum of the rows of W whose input is 1."""
    counts = X.sum(axis=1, dtype=np.int64)
    order = np.argsort(-counts, kind="stable")  # the boards with the most active inputs first
    counts = counts[order]
    _, cols = np.nonzero(X[order])  # active inputs, row by row, in increasing order
    first = np.cumsum(counts) - counts  # index in cols of the first active input of each row
    out = np.zeros((len(X), W.shape[1]), dtype=W.dtype)
    for j in range(int(counts[0]) if len(X) else 0):
        n = int(np.count_nonzero(counts > j))  # rows with a j-th active input: the first n, they are sorted
        out[:n] += W[cols[first[:n] + j]]
    result = np.empty_like(out)
    result[order] = out
    return result


class Dense:
    def __init__(self, n_in, n_out, spec, rng, dtype):
        """spec: normalized layer config {"size", "activation", ["alpha"], "dropout", "init"}."""
        self.activation = spec["activation"]
        self.alpha = spec.get("alpha", 0.0)  # parameter of leaky_relu and elu
        self.dropout = spec["dropout"]
        self.W = lookup(INITS, spec["init"], "init")(n_in, n_out, rng).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        # kept by forward() for backward(): input, pre-activation, output, dropout mask; gradients
        self.x = self.z = self.a = self.mask = None
        self.dW = self.db = None

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

    def forward_binary(self, X):
        """forward(X) at inference, for a matrix X of 0/1 values: the rows of W are added up."""
        self.x = None
        self.z = sum_active_rows(X, self.W) + self.b
        self.a = activations.forward(self.activation, self.z, self.alpha)
        self.mask = None
        return self.a

    def backward(self, grad, input_grad=True):
        """Gradient wrt the output -> gradient wrt the input (None if not needed). Stores dW, db."""
        if self.mask is not None:
            grad = grad * self.mask
        dz = activations.backward(self.activation, self.z, self.a, grad, self.alpha)
        return self.backward_from_z(dz, input_grad)

    def backward_from_z(self, dz, input_grad=True):
        """Gradient wrt the pre-activation z -> gradient wrt the input. Stores dW, db."""
        self.dW = self.x.T @ dz
        self.db = dz.sum(axis=0)
        return dz @ self.W.T if input_grad else None
