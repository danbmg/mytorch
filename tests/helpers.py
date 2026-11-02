"""Shared test helpers."""

import numpy as np


def numeric_grad(f, x, eps=1e-6):
    """Central finite differences of the scalar function f at x (float64 array, modified in place)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        plus = f()
        x[i] = old - eps
        minus = f()
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def rel_error(a, b):
    """max |a - b| / max(|a|, |b|): the usual gradient check measure."""
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12))


XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])


def small_config(**overrides):
    config = {
        "input_size": 2,
        "layers": [{"size": 8, "activation": "tanh"}, {"size": 2, "activation": "softmax"}],
        "optimizer": {"name": "adam", "lr": 0.05},
        "training": {"epochs": 300, "batch_size": 4},
        "seed": 0,
    }
    config.update(overrides)
    return config
