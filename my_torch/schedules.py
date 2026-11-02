"""Learning rate schedules, evaluated once at the start of every epoch.

A large learning rate moves fast at the beginning; a smaller one at the end
lets the weights settle in a minimum instead of bouncing around it.
"""

import math


class Schedule:
    name = ""

    def __init__(self, lr0):
        self.lr0 = float(lr0)

    def lr(self, epoch, last_loss=None):
        """Learning rate for `epoch` (0-based). `last_loss`: previous epoch's validation loss."""
        raise NotImplementedError


class Constant(Schedule):
    name = "constant"

    def lr(self, epoch, last_loss=None):
        return self.lr0


class Step(Schedule):
    """Multiply the learning rate by `gamma` every `step_size` epochs."""

    name = "step"

    def __init__(self, lr0, step_size=10, gamma=0.5):
        super().__init__(lr0)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def lr(self, epoch, last_loss=None):
        return self.lr0 * self.gamma ** (epoch // self.step_size)


class Exponential(Schedule):
    """Multiply the learning rate by `gamma` every epoch."""

    name = "exponential"

    def __init__(self, lr0, gamma=0.95):
        super().__init__(lr0)
        self.gamma = float(gamma)

    def lr(self, epoch, last_loss=None):
        return self.lr0 * self.gamma ** epoch


class Cosine(Schedule):
    """Half a cosine from lr0 down to lr_min over t_max epochs (default: the whole training)."""

    name = "cosine"

    def __init__(self, lr0, t_max=None, lr_min=0.0, epochs=None):
        super().__init__(lr0)
        self.t_max = int(t_max if t_max is not None else max(epochs or 1, 1))
        self.lr_min = float(lr_min)

    def lr(self, epoch, last_loss=None):
        progress = min(epoch, self.t_max) / self.t_max
        return self.lr_min + 0.5 * (self.lr0 - self.lr_min) * (1 + math.cos(math.pi * progress))


class Plateau(Schedule):
    """Multiply the learning rate by `factor` after `patience` epochs without improvement of the validation loss."""

    name = "plateau"

    def __init__(self, lr0, factor=0.5, patience=2, min_lr=1e-6, min_delta=0.0):
        super().__init__(lr0)
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.min_delta = float(min_delta)
        self.current = self.lr0
        self.best = math.inf
        self.wait = 0

    def lr(self, epoch, last_loss=None):
        if last_loss is not None:
            if last_loss < self.best - self.min_delta:
                self.best = last_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.current = max(self.current * self.factor, self.min_lr)
                    self.wait = 0
        return self.current


SCHEDULES = {cls.name: cls for cls in (Constant, Step, Exponential, Cosine, Plateau)}


def make_schedule(spec, lr0, epochs=None):
    """Build a schedule from its config, e.g. {"name": "step", "step_size": 10, "gamma": 0.5}."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in SCHEDULES:
        raise ValueError(f"unknown lr_schedule '{name}' (choices: {', '.join(SCHEDULES)})")
    if name == "cosine":
        spec["epochs"] = epochs
    return SCHEDULES[name](lr0, **spec)
