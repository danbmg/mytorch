"""Learning rate schedules, evaluated at the start of every epoch.

A large step moves fast at the beginning; a smaller one at the end lets the
weights settle in a minimum instead of bouncing around it. One class per
schedule, each with its settings and their defaults (read by the configuration),
in a registry name -> class; Schedule(spec, lr0, epochs) is the factory.
"""

import math

from .errors import lookup


class BaseSchedule:
    """A constant learning rate; the subclasses compute lr(epoch) from lr0."""

    NAME = "constant"
    DEFAULTS = {}  # settings and their default

    def __init__(self, spec, lr0, epochs):
        self.settings, self.lr0, self.current, self.epochs = spec, lr0, lr0, epochs

    def lr(self, epoch, last_loss=None):
        """Learning rate of `epoch` (0-based); last_loss: validation loss of the previous epoch."""
        return self.current


class Constant(BaseSchedule):
    NAME = "constant"


class Step(BaseSchedule):
    NAME, DEFAULTS = "step", {"step_size": 10, "gamma": 0.5}  # x gamma every step_size epochs

    def lr(self, epoch, last_loss=None):
        return self.lr0 * self.settings["gamma"] ** (epoch // self.settings["step_size"])


class Exponential(BaseSchedule):
    NAME, DEFAULTS = "exponential", {"gamma": 0.95}  # x gamma every epoch

    def lr(self, epoch, last_loss=None):
        return self.lr0 * self.settings["gamma"] ** epoch


class Cosine(BaseSchedule):
    NAME, DEFAULTS = "cosine", {"t_max": None, "lr_min": 0.0}  # half a cosine down to lr_min in t_max epochs (default: all)

    def __init__(self, spec, lr0, epochs):
        super().__init__(spec, lr0, epochs)
        self.t_max = spec.get("t_max") or max(epochs, 1)

    def lr(self, epoch, last_loss=None):
        s, progress = self.settings, min(epoch, self.t_max) / self.t_max
        return s["lr_min"] + 0.5 * (self.lr0 - s["lr_min"]) * (1 + math.cos(math.pi * progress))


class Plateau(BaseSchedule):
    NAME, DEFAULTS = "plateau", {"factor": 0.5, "patience": 2, "min_lr": 1e-6, "min_delta": 0.0}  # x factor when val_loss stalls

    def __init__(self, spec, lr0, epochs):
        super().__init__(spec, lr0, epochs)
        self.best, self.wait = math.inf, 0

    def lr(self, epoch, last_loss=None):
        s = self.settings
        if last_loss is not None:
            if last_loss < self.best - s["min_delta"]:
                self.best, self.wait = last_loss, 0
            else:
                self.wait += 1
                if self.wait >= s["patience"]:
                    self.current, self.wait = max(self.current * s["factor"], s["min_lr"]), 0
        return self.current


SCHEDULES = {cls.NAME: cls for cls in (Constant, Step, Exponential, Cosine, Plateau)}


def Schedule(spec, lr0, epochs):  # noqa: N802  (factory kept under the former class name)
    """The schedule described by spec ({"name": ..., settings}), starting at lr0 for `epochs` epochs."""
    return lookup(SCHEDULES, spec["name"], "lr_schedule")(spec, lr0, epochs)
