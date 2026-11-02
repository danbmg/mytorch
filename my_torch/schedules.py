"""Learning rate schedules, evaluated at the start of every epoch.

A large step moves fast at the beginning; a smaller one at the end lets the
weights settle in a minimum instead of bouncing around it.
"""

import math

# settings of each schedule and their default
SCHEDULES = {
    "constant": {},
    "step": {"step_size": 10, "gamma": 0.5},  # x gamma every step_size epochs
    "exponential": {"gamma": 0.95},  # x gamma every epoch
    "cosine": {"t_max": None, "lr_min": 0.0},  # half a cosine down to lr_min in t_max epochs (default: all)
    "plateau": {"factor": 0.5, "patience": 2, "min_lr": 1e-6, "min_delta": 0.0},  # x factor when val_loss stalls
}


class Schedule:
    def __init__(self, spec, lr0, epochs):
        self.s, self.lr0, self.current = spec, lr0, lr0
        self.t_max = spec.get("t_max") or max(epochs, 1)
        self.best, self.wait = math.inf, 0

    def lr(self, epoch, last_loss=None):
        """Learning rate of `epoch` (0-based); last_loss: validation loss of the previous epoch."""
        s, name = self.s, self.s["name"]
        if name == "step":
            return self.lr0 * s["gamma"] ** (epoch // s["step_size"])
        if name == "exponential":
            return self.lr0 * s["gamma"] ** epoch
        if name == "cosine":
            progress = min(epoch, self.t_max) / self.t_max
            return s["lr_min"] + 0.5 * (self.lr0 - s["lr_min"]) * (1 + math.cos(math.pi * progress))
        if name == "plateau" and last_loss is not None:
            if last_loss < self.best - s["min_delta"]:
                self.best, self.wait = last_loss, 0
            else:
                self.wait += 1
                if self.wait >= s["patience"]:
                    self.current, self.wait = max(self.current * s["factor"], s["min_lr"]), 0
        return self.current
