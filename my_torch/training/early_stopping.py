"""Early stopping: the validation loss decides when a training stops and which
weights are kept.

The best validation epoch is tracked even without early stopping (it goes in
the network's metadata); with it, the training stops after `patience` epochs
without improvement and the weights of the best epoch are restored, so the
saved network is the one that generalised best, not the last one.
"""

import math


class EarlyStopping:
    def __init__(self, net, spec):
        """spec: None (track the best epoch only) or {"patience", "min_delta", "restore_best"}."""
        self.net = net
        self.spec = spec
        self.best_loss = math.inf
        self.best_epoch = None
        self.best_weights = None
        self.wait = 0  # epochs since the last improvement

    @property
    def patience(self):
        return self.spec["patience"]

    def update(self, epoch, val_loss):
        """Record this epoch's validation loss -> True when the training must stop."""
        min_delta = self.spec["min_delta"] if self.spec else 0.0
        if val_loss < self.best_loss - min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            if self.spec and self.spec["restore_best"]:
                self.best_weights = self.net.get_weights()
        else:
            self.wait += 1
        return bool(self.spec) and self.wait >= self.spec["patience"]

    def restore(self, last_epoch):
        """Put back the weights of the best epoch, unless it is the last one -> True if restored."""
        if self.best_weights is None or self.best_epoch == last_epoch:
            return False
        self.net.set_weights(self.best_weights)
        return True
