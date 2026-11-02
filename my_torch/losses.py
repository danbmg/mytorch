"""Losses, averaged over the batch; y is one-hot (N, n_out).

cross_entropy always follows a softmax output (checked by config.py): its
gradient is then taken directly on the logits, (out - y) / N, which is simple
and never divides by a probability close to 0 (network.py).
"""

import numpy as np

LOSSES = ("cross_entropy", "mse")


def value(name, out, y, logits):
    if name == "cross_entropy":
        # log-softmax computed from the logits: exact, no log(0) possible
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-(y * log_probs).sum() / len(y))
    return float(np.mean((out - y) ** 2))


def mse_grad(out, y):
    return 2 * (out - y) / out.size
