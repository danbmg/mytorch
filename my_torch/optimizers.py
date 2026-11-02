"""Gradient descent optimisers. step() updates every parameter array in place;
the schedule changes `lr` between epochs."""

import numpy as np

# settings of each optimiser and their default (the learning rate is apart)
OPTIMIZERS = {
    "sgd": {},
    "momentum": {"momentum": 0.9},
    "rmsprop": {"rho": 0.9, "eps": 1e-8},
    "adam": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
}
DEFAULT_LR = {"sgd": 0.01, "momentum": 0.01, "rmsprop": 0.001, "adam": 0.001}


class Optimizer:
    def __init__(self, name, lr, **settings):
        self.name, self.lr, self.s = name, lr, settings
        self.t = 0
        self.m = self.v = None  # first moment (velocity), second moment (squared gradients)

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params] if self.name in ("rmsprop", "adam") else self.m
        self.t += 1
        s = self.s
        for p, g, m, v in zip(params, grads, self.m, self.v):
            if self.name == "sgd":
                p -= self.lr * g
            elif self.name == "momentum":
                # velocity: running sum of the past steps, damped by `momentum`
                m *= s["momentum"]
                m -= self.lr * g
                p += m
            elif self.name == "rmsprop":
                # each step divided by the running RMS of its gradient
                v *= s["rho"]
                v += (1 - s["rho"]) * g * g
                p -= self.lr * g / (np.sqrt(v) + s["eps"])
            else:
                # adam = momentum + rmsprop; both moments start at 0, hence the (1 - beta^t) correction
                m *= s["beta1"]
                m += (1 - s["beta1"]) * g
                v *= s["beta2"]
                v += (1 - s["beta2"]) * g * g
                p -= self.lr * (m / (1 - s["beta1"] ** self.t)) / (np.sqrt(v / (1 - s["beta2"] ** self.t)) + s["eps"])

    def state(self):
        """Arrays kept between steps (flushed of subnormal floats along with the weights)."""
        if self.m is None or self.name == "sgd":
            return []
        return self.m + (self.v if self.v is not self.m else [])
