"""Gradient descent optimizers. step() updates every parameter array in place;
the learning rate schedule changes `lr` between epochs.

One class per optimizer, in a registry name -> class. Each class lists its
settings with their default (read by the configuration) and the arrays it keeps
for each parameter from one step to the next (STATE: the moments). The update
formulas are those of the papers, written in place so that no copy of the
parameters is made.
"""

import numpy as np

from .errors import lookup


class Optimizer:
    NAME = ""
    DEFAULTS = {}  # settings and their default (the learning rate apart)
    LR = 0.01  # default learning rate
    STATE = ()  # arrays kept for each parameter between two steps

    def __init__(self, lr, **settings):
        self.lr = lr
        self.settings = {**self.DEFAULTS, **settings}
        self.t = 0  # steps done
        self.state = None  # one dict {name: array} per parameter, created at the first step

    def step(self, params, grads):
        if self.state is None:
            self.state = [{name: np.zeros_like(p) for name in self.STATE} for p in params]
        self.t += 1
        for p, g, state in zip(params, grads, self.state):
            self.update(p, g, state)

    def update(self, p, g, state):
        """One parameter array p and its gradient g: updates p (and its state) in place."""
        raise NotImplementedError

    def moments(self):
        """The STATE arrays of every parameter (flushed of subnormal floats along with the weights)."""
        return [state[name] for state in self.state or [] for name in self.STATE]


class SGD(Optimizer):
    NAME = "sgd"

    def update(self, p, g, state):
        p -= self.lr * g


class Momentum(Optimizer):
    NAME = "momentum"
    DEFAULTS = {"momentum": 0.9}
    STATE = ("velocity",)

    def update(self, p, g, state):
        # velocity: running sum of the past steps, damped by `momentum`
        velocity = state["velocity"]
        velocity *= self.settings["momentum"]
        velocity -= self.lr * g
        p += velocity


class RMSprop(Optimizer):
    NAME = "rmsprop"
    DEFAULTS = {"rho": 0.9, "eps": 1e-8}
    LR = 0.001
    STATE = ("v",)

    def update(self, p, g, state):
        # each step divided by the running RMS of its gradient
        rho = self.settings["rho"]
        v = state["v"]
        v *= rho
        v += (1 - rho) * g * g
        p -= self.lr * g / (np.sqrt(v) + self.settings["eps"])


class Adam(Optimizer):
    NAME = "adam"
    DEFAULTS = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
    LR = 0.001
    STATE = ("m", "v")

    def update(self, p, g, state):
        # Adam = momentum (m) + RMSprop (v); both start at 0, hence the division by (1 - beta^t)
        beta1 = self.settings["beta1"]
        beta2 = self.settings["beta2"]
        m = state["m"]
        v = state["v"]
        m *= beta1
        m += (1 - beta1) * g
        v *= beta2
        v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.settings["eps"])


OPTIMIZERS = {cls.NAME: cls for cls in (SGD, Momentum, RMSprop, Adam)}


def make_optimizer(name, lr, **settings):
    """The optimizer called `name`: make_optimizer("adam", 0.001, beta1=0.9)."""
    return lookup(OPTIMIZERS, name, "optimizer")(lr, **settings)
