"""Gradient descent optimisers. step() updates every parameter array in place;
the schedule changes `lr` between epochs.

One class per optimiser, each with its settings and their defaults (read by the
configuration), in a registry name -> class; Optimizer(name, lr, **settings) is
the factory. The update formulas are the ones of the papers, written in place
so that no copy of the parameters is made.
"""

import numpy as np

from .errors import lookup


class BaseOptimizer:
    """What every optimiser shares: the learning rate, the settings, the step counter
    and the arrays kept between steps (m: first moment or velocity, v: second moment)."""

    NAME = ""
    DEFAULTS = {}  # settings and their default (the learning rate is apart)
    LR = 0.01  # default learning rate
    MOMENTS = 0  # arrays kept per parameter between steps: 0, 1 (m) or 2 (m and v)

    def __init__(self, lr, **settings):
        self.lr, self.settings = lr, {**self.DEFAULTS, **settings}
        self.t = 0
        self.m = self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params] if self.MOMENTS == 2 else self.m
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            self.update(p, g, m, v)

    def update(self, p, g, m, v):
        """One parameter array p, its gradient g and its moment arrays: updates p in place."""
        raise NotImplementedError

    def state(self):
        """Arrays kept between steps (flushed of subnormal floats along with the weights)."""
        if self.m is None or self.MOMENTS == 0:
            return []
        return self.m + (self.v if self.v is not self.m else [])


class SGD(BaseOptimizer):
    NAME = "sgd"

    def update(self, p, g, m, v):
        p -= self.lr * g


class Momentum(BaseOptimizer):
    NAME, DEFAULTS, MOMENTS = "momentum", {"momentum": 0.9}, 1

    def update(self, p, g, m, v):
        # velocity: running sum of the past steps, damped by `momentum`
        m *= self.settings["momentum"]
        m -= self.lr * g
        p += m


class RMSprop(BaseOptimizer):
    NAME, DEFAULTS, LR, MOMENTS = "rmsprop", {"rho": 0.9, "eps": 1e-8}, 0.001, 2

    def update(self, p, g, m, v):
        # each step divided by the running RMS of its gradient
        s = self.settings
        v *= s["rho"]
        v += (1 - s["rho"]) * g * g
        p -= self.lr * g / (np.sqrt(v) + s["eps"])


class Adam(BaseOptimizer):
    NAME, DEFAULTS, LR, MOMENTS = "adam", {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}, 0.001, 2

    def update(self, p, g, m, v):
        # adam = momentum + rmsprop; both moments start at 0, hence the (1 - beta^t) correction
        s = self.settings
        m *= s["beta1"]
        m += (1 - s["beta1"]) * g
        v *= s["beta2"]
        v += (1 - s["beta2"]) * g * g
        p -= self.lr * (m / (1 - s["beta1"] ** self.t)) / (np.sqrt(v / (1 - s["beta2"] ** self.t)) + s["eps"])


OPTIMIZERS = {cls.NAME: cls for cls in (SGD, Momentum, RMSprop, Adam)}


def Optimizer(name, lr, **settings):  # noqa: N802  (factory kept under the former class name)
    """The optimiser called `name`: Optimizer("adam", 0.001, beta1=0.9)."""
    return lookup(OPTIMIZERS, name, "optimizer")(lr, **settings)
