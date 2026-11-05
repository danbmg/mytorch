"""Mini-batch training of a Network: fit() checks the settings and the data,
then a Training object runs the epochs (shuffle, mini-batches, evaluation,
learning rate schedule, early stopping) and writes the history and metadata
into the network.

Everything random is drawn from one generator seeded by (network seed, epochs
already trained): the same call on the same network gives the same weights, and
a resumed training is reproducible too.
"""

import math
import sys
import time

import numpy as np

from ..config import TRAINING, normalize_training
from ..data import inputs, targets
from ..errors import TrainingError
from ..optimizers import make_optimizer
from ..schedules import make_schedule
from . import batch
from .early_stopping import EarlyStopping
from .progress import Progress

HISTORY = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time")


def fit(net, X, Y, X_val=None, Y_val=None, verbose=False, log=None, callback=None, **hp):
    """Train net on (X, Y), validating on (X_val, Y_val) after every epoch -> history of this call.

    hp overrides config["training"] for this call only (epochs, batch_size, l2...)."""
    unknown = sorted(set(hp) - set(TRAINING))
    if unknown:
        raise ValueError(f"unknown training setting(s): {', '.join(unknown)}")
    cfg = normalize_training({**net.config["training"], **hp})
    progress = Progress(log or sys.stderr, verbose)
    X = inputs(net, X)
    Y = targets(net, Y, len(X))
    if len(X) == 0:
        raise ValueError("no training data")
    if X_val is None or len(X_val) == 0:
        X_val = Y_val = None
    else:
        X_val = inputs(net, X_val)
        Y_val = targets(net, Y_val, len(X_val))
    if cfg["early_stopping"] and X_val is None:
        progress.message("early stopping needs validation data: disabled")
    training = Training(net, cfg, X, Y, X_val, Y_val, progress, callback)
    start = time.perf_counter()
    # a diverging training overflows: reported as a TrainingError, not as numpy warnings
    with np.errstate(over="ignore", invalid="ignore"):
        stopped = training.run()
    training.finish(stopped, time.perf_counter() - start)
    return training.history


class Training:
    """The state of one fit() call: data, optimizer, schedule, random generator, history."""

    def __init__(self, net, cfg, X, Y, X_val, Y_val, progress, callback):
        self.net = net
        self.cfg = cfg
        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
        self.has_val = X_val is not None
        self.progress = progress
        self.callback = callback
        self.optimizer = make_optimizer(**net.config["optimizer"])
        self.schedule = make_schedule(net.config["lr_schedule"], self.optimizer.lr, cfg["epochs"])
        self.params = [p for layer in net.layers for p in (layer.W, layer.b)]
        self.first = net.meta["epochs_trained"]  # epochs done by earlier fit() calls
        # seeded by (seed, epochs already trained): resuming is reproducible too
        self.rng = np.random.default_rng([net.seed, self.first])
        self.X_eval, self.Y_eval = self._eval_subset()
        self.history = {key: [] for key in HISTORY}
        self.stopping = EarlyStopping(net, cfg["early_stopping"] if self.has_val else None)
        self.steps = 0
        self.last_loss = None  # loss of the previous epoch, read by the plateau schedule

    def _eval_subset(self):
        """The training rows re-evaluated after each epoch (all of them, or a random subset)."""
        n = len(self.X)
        size = self.cfg["eval_train_samples"]
        if 0 < size < n:
            idx = self.rng.choice(n, size=size, replace=False)
            return self.X[idx], self.Y[idx]
        return self.X, self.Y

    def run(self):
        """Every epoch -> True if early stopping ended the training."""
        for epoch in range(self.first + 1, self.first + self.cfg["epochs"] + 1):
            if self._epoch(epoch):
                return True
        return False

    def _batches(self, epoch):
        """One pass over the training data -> running loss and accuracy of the epoch."""
        n = len(self.X)
        size = self.cfg["batch_size"]
        order = self.rng.permutation(n) if self.cfg["shuffle"] else np.arange(n)
        n_batches = math.ceil(n / size)
        total_loss = 0.0
        total_correct = 0
        for b, start in enumerate(range(0, n, size)):
            idx = order[start:start + size]
            loss, correct = batch.train_batch(self.net, self.optimizer, self.params,
                                              self.X[idx], self.Y[idx], self.cfg, self.rng)
            total_loss += loss
            total_correct += correct
            self.steps += 1
            if self.steps % batch.FLUSH_EVERY == 0:
                batch.flush_subnormals(self.params + self.optimizer.moments())
            self.progress.batch(epoch, b, n_batches)
        if not math.isfinite(total_loss):
            raise TrainingError(f"the loss diverged (NaN/inf) at epoch {epoch}:"
                                " lower the learning rate or enable grad_clip")
        return {"loss": total_loss / n, "accuracy": total_correct / n}

    def _epoch(self, epoch):
        """Train one epoch, evaluate, record -> True when early stopping says stop."""
        self.optimizer.lr = self.schedule.lr(epoch - self.first - 1, self.last_loss)
        start = time.perf_counter()
        running = self._batches(epoch)
        if self.cfg["eval_train_samples"]:
            train = self.net.evaluate(self.X_eval, self.Y_eval)  # dropout off, unlike the running values
        else:
            train = running
        if self.has_val:
            val = self.net.evaluate(self.X_val, self.Y_val)
        else:
            val = {"loss": None, "accuracy": None}
        seconds = time.perf_counter() - start
        values = (epoch, train["loss"], train["accuracy"], val["loss"], val["accuracy"],
                  self.optimizer.lr, seconds)
        for key, value in zip(HISTORY, values):
            self.history[key].append(value)
        self.progress.epoch(epoch, self.first + self.cfg["epochs"], train, val, self.optimizer.lr, seconds)
        if self.callback:
            self.callback(epoch, self.history)
        self.last_loss = val["loss"] if self.has_val else train["loss"]
        return self.has_val and self.stopping.update(epoch, val["loss"])

    def finish(self, stopped, seconds):
        """Best weights restored, metadata and history of the network updated."""
        net = self.net
        stopping = self.stopping
        if stopped:
            self.progress.message(f"early stopping: no val_loss improvement for {stopping.patience} epochs")
        # keep the weights of the best validation epoch, not those of the last one
        if stopping.restore(self.history["epoch"][-1]):
            self.progress.message(f"weights of epoch {stopping.best_epoch} restored"
                                  f" (val_loss {stopping.best_loss:.4f})")
        net.meta.update(epochs_trained=self.first + len(self.history["epoch"]), train_samples=len(self.X),
                        stopped_early=stopped, train_time=net.meta["train_time"] + seconds)
        if stopping.best_epoch is not None:
            net.meta.update(best_epoch=stopping.best_epoch, best_val_loss=stopping.best_loss)
        for key, values in self.history.items():
            net.history.setdefault(key, []).extend(values)
