"""Inputs and targets as the network expects them.

Inputs: a 2-D array with input_size columns, any dtype (uint8 chess boards are
converted batch by batch, 4x less memory than float32). Targets: class indices
(N,) turned into one-hot rows, or 2-D targets kept as they are (regression, or
already one-hot). Checked once here, so that every error names the shape.
"""

import numpy as np


def inputs(net, X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != net.input_size:
        raise ValueError(f"expected inputs of shape (N, {net.input_size}), got {X.shape}")
    return X


def targets(net, Y, n):
    """Class indices (N,) -> one-hot (N, n_out); 2-D targets are kept as they are."""
    Y = np.asarray(Y)
    if len(Y) != n:
        raise ValueError(f"{n} inputs but {len(Y)} targets")
    if Y.ndim == 2 and Y.shape[1] == net.output_size:
        return Y.astype(net.dtype, copy=False)
    if Y.ndim != 1:
        raise ValueError(f"expected targets of shape (N,) or (N, {net.output_size}), got {Y.shape}")
    if net.output_size == 1:
        return Y.reshape(-1, 1).astype(net.dtype)
    if len(Y) and (Y.min() < 0 or Y.max() >= net.output_size):
        raise ValueError(f"class indices must be in [0, {net.output_size})")
    onehot = np.zeros((n, net.output_size), dtype=net.dtype)
    onehot[np.arange(n), Y.astype(np.int64)] = 1
    return onehot
