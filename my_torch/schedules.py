"""Learning rate schedules, evaluated at the start of every epoch.

A large step moves fast at the beginning; a smaller one at the end lets the
weights settle in a minimum instead of bouncing around it. One class per
schedule, each with its settings and their defaults (read by the configuration),
in a registry name -> class; make_schedule(spec, lr0, epochs) builds one.
"""

import math

from .errors import lookup


class Schedule:
    NAME = ""
    DEFAULTS = {}  # settings and their default

    def __init__(self, spec, lr0, epochs):
        self.settings = spec
        self.lr0 = lr0
        self.epochs = epochs

    def lr(self, epoch, last_loss=None):
        """Learning rate of `epoch` (0-based); last_loss: validation loss of the epoch before."""
        raise NotImplementedError


class Constant(Schedule):
    NAME = "constant"

    def lr(self, epoch, last_loss=None):
        return self.lr0


class Step(Schedule):
    NAME = "step"
    DEFAULTS = {"step_size": 10, "gamma": 0.5}  # x gamma every step_size epochs

    def lr(self, epoch, last_loss=None):
        return self.lr0 * self.settings["gamma"] ** (epoch // self.settings["step_size"])


class Exponential(Schedule):
    NAME = "exponential"
    DEFAULTS = {"gamma": 0.95}  # x gamma every epoch

    def lr(self, epoch, last_loss=None):
        return self.lr0 * self.settings["gamma"] ** epoch


class Cosine(Schedule):
    NAME = "cosine"
    DEFAULTS = {"t_max": None, "lr_min": 0.0}  # half a cosine down to lr_min in t_max epochs (default: all)

    def __init__(self, spec, lr0, epochs):
        super().__init__(spec, lr0, epochs)
        self.t_max = spec.get("t_max") or max(epochs, 1)

    def lr(self, epoch, last_loss=None):
        lr_min = self.settings["lr_min"]
        progress = min(epoch, self.t_max) / self.t_max
        return lr_min + 0.5 * (self.lr0 - lr_min) * (1 + math.cos(math.pi * progress))


class Plateau(Schedule):
    NAME = "plateau"
    DEFAULTS = {"factor": 0.5, "patience": 2, "min_lr": 1e-6, "min_delta": 0.0}  # x factor when val_loss stalls

    def __init__(self, spec, lr0, epochs):
        super().__init__(spec, lr0, epochs)
        self.current = lr0
        self.best = math.inf
        self.wait = 0  # epochs since the last improvement

    def lr(self, epoch, last_loss=None):
        s = self.settings
        if last_loss is None:
            return self.current
        if last_loss < self.best - s["min_delta"]:
            self.best = last_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= s["patience"]:
                self.current = max(self.current * s["factor"], s["min_lr"])
                self.wait = 0
        return self.current


SCHEDULES = {cls.NAME: cls for cls in (Constant, Step, Exponential, Cosine, Plateau)}


def make_schedule(spec, lr0, epochs):
    """The schedule described by spec ({"name": ..., settings}), starting at lr0 for `epochs` epochs."""
    return lookup(SCHEDULES, spec["name"], "lr_schedule")(spec, lr0, epochs)
