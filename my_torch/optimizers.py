"""Gradient descent optimisers.

`step(params, grads)` updates every parameter array in place. `lr` is a plain
attribute so that the learning rate schedule can change it between epochs.
"""

import numpy as np


class Optimizer:
    name = ""

    def __init__(self, lr):
        self.lr = float(lr)

    def step(self, params, grads):
        raise NotImplementedError

    def state(self):
        """Arrays the optimiser keeps between steps (moments, velocity...)."""
        return []


class SGD(Optimizer):
    name = "sgd"

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum(Optimizer):
    """SGD with a velocity: a running sum of past gradients damped by `momentum`."""

    name = "momentum"

    def __init__(self, lr, momentum=0.9, nesterov=False):
        super().__init__(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.momentum
            v -= self.lr * g
            if self.nesterov:
                # look-ahead: apply the step the velocity is about to take
                p += self.momentum * v - self.lr * g
            else:
                p += v

    def state(self):
        return self.velocity or []


class RMSprop(Optimizer):
    """Divides each step by a running average of the squared gradient."""

    name = "rmsprop"

    def __init__(self, lr, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self.square = None

    def step(self, params, grads):
        if self.square is None:
            self.square = [np.zeros_like(p) for p in params]
        for p, g, s in zip(params, grads, self.square):
            s *= self.rho
            s += (1 - self.rho) * g * g
            p -= self.lr * g / (np.sqrt(s) + self.eps)

    def state(self):
        return self.square or []


class Adam(Optimizer):
    """Momentum (first moment) + RMSprop (second moment), both bias-corrected."""

    name = "adam"

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        # both moments start at 0: dividing by (1 - beta^t) removes that bias early on
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)

    def state(self):
        return (self.m or []) + (self.v or [])


OPTIMIZERS = {cls.name: cls for cls in (SGD, Momentum, RMSprop, Adam)}


def make_optimizer(spec):
    """Build an optimiser from its config, e.g. {"name": "adam", "lr": 0.001}."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{name}' (choices: {', '.join(OPTIMIZERS)})")
    return OPTIMIZERS[name](**spec)
