"""Losses, averaged over the batch; y is one-hot (N, n_out).

One class per loss in a registry, each with value(out, y, logits) and
backward(layer, out, y): the gradient through the output layer. cross_entropy
always follows a softmax output (checked by the configuration): its gradient is
then taken directly on the logits, (out - y) / N, which is simple and never
divides by a probability close to 0.
"""

import numpy as np

from .errors import lookup


class CrossEntropy:
    NAME = "cross_entropy"

    @staticmethod
    def value(out, y, logits):
        # log-softmax computed from the logits: exact, no log(0) possible
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-(y * log_probs).sum() / len(y))

    @staticmethod
    def backward(layer, out, y, input_grad=True):
        # softmax + cross-entropy: gradient on the logits = (p - y) / N, the softmax Jacobian is skipped
        return layer.backward_from_z((out - y) / len(y), input_grad)


class MSE:
    NAME = "mse"

    @staticmethod
    def value(out, y, logits):
        return float(np.mean((out - y) ** 2))

    @staticmethod
    def backward(layer, out, y, input_grad=True):
        return layer.backward(2 * (out - y) / out.size, input_grad)


LOSSES = {cls.NAME: cls for cls in (CrossEntropy, MSE)}


def loss(name):
    """The loss called `name` (a class with value() and backward())."""
    return lookup(LOSSES, name, "loss")
