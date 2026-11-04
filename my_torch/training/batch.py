"""One optimisation step on a mini-batch, and what keeps it fast.

The gradients of every layer are gathered in one list, so L2 (added to the
gradient, not to the loss: same result, no extra pass) and gradient clipping
(one scale for all the gradients: the direction of the step is kept) are a few
lines each, and the optimiser sees plain arrays.
"""

import math

import numpy as np

FLUSH_EVERY = 100  # optimisation steps between two flush_subnormals()


def flush_subnormals(arrays):
    """Set to 0 the values too small to be normal floats (< 1.2e-38 in float32).

    With L2, weights that never receive a gradient (an input always 0, such as a
    pawn on rank 8) shrink towards 0 forever; once subnormal, every operation on
    them is 10-100x slower on the CPU. 0 is their optimum anyway.
    """
    for a in arrays:
        np.putmask(a, np.abs(a) < np.finfo(a.dtype).tiny, 0)


def gradients(net, l2):
    """dW (+ l2 W) and db of every layer, in the order of net's parameters."""
    return [g for layer in net.layers for g in (layer.dW + l2 * layer.W if l2 else layer.dW, layer.db)]


def clip(grads, max_norm):
    """Scale every gradient down, in place, if their global norm exceeds max_norm."""
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm > max_norm:
        for g in grads:
            g *= max_norm / (norm + 1e-12)


def train_batch(net, opt, params, xb, yb, cfg, rng):
    """Forward with dropout, loss, backward, one optimiser step -> (summed loss, correct predictions)."""
    out = net._forward(xb, training=True, rng=rng)
    loss, correct = net._loss(out, yb) * len(xb), int((net._decide(out) == net._decide(yb)).sum())
    net._backward(out, yb)
    grads = gradients(net, cfg["l2"])
    if cfg["grad_clip"]:
        clip(grads, cfg["grad_clip"])
    opt.step(params, grads)
    return loss, correct
