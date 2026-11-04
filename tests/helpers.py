"""Helpers shared by the test files: the XOR problem (small enough for finite
differences and a few hundred epochs in milliseconds), a numeric gradient, and a
capture of a command line entry point. Imported explicitly (no pytest fixtures),
so the tests also run with plain unittest."""

import contextlib
import io
import os

import numpy as np

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "boards.txt")
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])


def xor_config(**overrides):
    return {"input_size": 2, "layers": [{"size": 8, "activation": "tanh"}, {"size": 2, "activation": "softmax"}],
            "optimizer": {"name": "adam", "lr": 0.05}, "training": {"epochs": 300, "batch_size": 4}, "seed": 0,
            **overrides}


def numeric_grad(f, x, eps=1e-6):
    """Central finite differences of the scalar function f() with respect to the array x."""
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        plus = f()
        x[i] = old - eps
        grad[i] = (plus - f()) / (2 * eps)
        x[i] = old
    return grad


def rel_error(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12))


def run_main(main, *args):
    """main(argv) with stdout and stderr captured -> (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        return main(list(args)), out.getvalue(), err.getvalue()
