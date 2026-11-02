"""Loss functions.

`value()` is the loss averaged over the batch. `grad()` is its gradient with
respect to the network output. When the output activation is the one the loss
is designed for (softmax for cross-entropy, sigmoid for binary cross-entropy),
`fused_grad()` gives the gradient with respect to the logits directly: it is
simpler (`out - y`) and avoids dividing by probabilities close to 0.
"""

import numpy as np

# floor for probabilities before taking their log, when no logits are available
EPS = 1e-12


class Loss:
    name = ""
    fused_activation = None

    def value(self, out, y, logits=None):
        raise NotImplementedError

    def grad(self, out, y):
        raise NotImplementedError

    def fused_grad(self, out, y):
        raise NotImplementedError


class CrossEntropy(Loss):
    """Categorical cross-entropy, y one-hot of shape (N, n_classes)."""

    name = "cross_entropy"
    fused_activation = "softmax"

    def value(self, out, y, logits=None):
        if logits is not None:
            # log-softmax computed from the logits: exact, no log(0) possible
            shifted = logits - logits.max(axis=1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        else:
            log_probs = np.log(np.clip(out, EPS, 1.0))
        return float(-(y * log_probs).sum() / len(y))

    def grad(self, out, y):
        return -y / np.clip(out, EPS, 1.0) / len(y)

    def fused_grad(self, out, y):
        return (out - y) / len(y)


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy for independent 0/1 outputs (summed over outputs)."""

    name = "binary_cross_entropy"
    fused_activation = "sigmoid"

    def value(self, out, y, logits=None):
        if logits is not None:
            # log(1 + exp(-|z|)) form: never overflows
            z = logits
            per_item = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        else:
            p = np.clip(out, EPS, 1 - EPS)
            per_item = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(per_item.sum() / len(y))

    def grad(self, out, y):
        p = np.clip(out, EPS, 1 - EPS)
        return (p - y) / (p * (1 - p)) / len(y)

    def fused_grad(self, out, y):
        return (out - y) / len(y)


class MSE(Loss):
    """Mean squared error over every output of every sample."""

    name = "mse"

    def value(self, out, y, logits=None):
        return float(np.mean((out - y) ** 2))

    def grad(self, out, y):
        return 2 * (out - y) / out.size


LOSSES = {cls.name: cls for cls in (CrossEntropy, BinaryCrossEntropy, MSE)}


def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"unknown loss '{name}' (choices: {', '.join(LOSSES)})")
    return LOSSES[name]()
