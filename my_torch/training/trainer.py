"""The epoch loop of a training: shuffle, mini-batches, evaluation, learning
rate schedule, early stopping, history and metadata.

A Training object holds the state of one fit() call (optimiser, schedule,
random generator, history) so that each stage is a short method. Everything
random is drawn from one generator seeded by (network seed, epochs already
trained): the same call on the same network gives the same weights, and a
resumed training is reproducible too.
"""

import math
import time

import numpy as np

from ..errors import TrainingError
from ..optimizers import Optimizer
from ..schedules import Schedule
from . import batch
from .early_stopping import EarlyStopping

HISTORY = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time")


class Training:
    def __init__(self, net, cfg, X, Y, X_val, Y_val, progress, callback):
        self.net, self.cfg, self.X, self.Y, self.X_val, self.Y_val = net, cfg, X, Y, X_val, Y_val
        self.progress, self.callback, self.has_val = progress, callback, X_val is not None
        self.opt = Optimizer(**net.config["optimizer"])
        self.sched = Schedule(net.config["lr_schedule"], self.opt.lr, cfg["epochs"])
        self.params = [p for layer in net.layers for p in (layer.W, layer.b)]
        self.first = net.meta["epochs_trained"]
        # seeded by (seed, epochs already trained): resuming is reproducible too
        self.rng = np.random.default_rng([net.seed, self.first])
        self.X_eval, self.Y_eval = self._eval_subset()
        self.history = {key: [] for key in HISTORY}
        self.stopping = EarlyStopping(net, cfg["early_stopping"] if self.has_val else None)
        self.steps, self.last_loss = 0, None

    def _eval_subset(self):
        """The training rows re-evaluated after each epoch (all of them, or a random subset)."""
        n, size = len(self.X), self.cfg["eval_train_samples"]
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
        n, bs, cfg = len(self.X), self.cfg["batch_size"], self.cfg
        order = self.rng.permutation(n) if cfg["shuffle"] else np.arange(n)
        n_batches, run_loss, run_correct = math.ceil(n / bs), 0.0, 0
        for b, start in enumerate(range(0, n, bs)):
            idx = order[start:start + bs]
            loss, correct = batch.train_batch(self.net, self.opt, self.params, self.X[idx], self.Y[idx], cfg, self.rng)
            run_loss, run_correct = run_loss + loss, run_correct + correct
            self.steps += 1
            if self.steps % batch.FLUSH_EVERY == 0:
                batch.flush_subnormals(self.params + self.opt.state())
            self.progress.batch(epoch, b, n_batches)
        if not math.isfinite(run_loss):
            raise TrainingError(f"the loss diverged (NaN/inf) at epoch {epoch}: lower the learning rate or enable grad_clip")
        return {"loss": run_loss / n, "accuracy": run_correct / n}

    def _epoch(self, epoch):
        self.opt.lr = self.sched.lr(epoch - self.first - 1, self.last_loss)
        t0 = time.perf_counter()
        running = self._batches(epoch)
        train = self.net.evaluate(self.X_eval, self.Y_eval) if self.cfg["eval_train_samples"] else running
        val = self.net.evaluate(self.X_val, self.Y_val) if self.has_val else {"loss": None, "accuracy": None}
        for key, value in zip(HISTORY, (epoch, train["loss"], train["accuracy"], val["loss"], val["accuracy"],
                                        self.opt.lr, time.perf_counter() - t0)):
            self.history[key].append(value)
        self.progress.epoch(epoch, self.first + self.cfg["epochs"], train, val, self.opt.lr, self.history["time"][-1])
        if self.callback:
            self.callback(epoch, self.history)
        self.last_loss = val["loss"] if self.has_val else train["loss"]
        return self.has_val and self.stopping.update(epoch, val["loss"])

    def finish(self, stopped, seconds):
        """Best weights restored, metadata and history of the network updated."""
        net, stopping, history = self.net, self.stopping, self.history
        if stopped:
            self.progress.message(f"early stopping: no val_loss improvement for {stopping.patience} epochs")
        # keep the weights of the best validation epoch, not those of the last one
        if stopping.restore(history["epoch"][-1]):
            self.progress.message(f"weights of epoch {stopping.best_epoch} restored (val_loss {stopping.best_loss:.4f})")
        net.meta.update(epochs_trained=self.first + len(history["epoch"]), train_samples=len(self.X),
                        stopped_early=stopped, train_time=net.meta["train_time"] + seconds)
        if stopping.best_epoch is not None:
            net.meta.update(best_epoch=stopping.best_epoch, best_val_loss=stopping.best_loss)
        for key, values in history.items():
            net.history.setdefault(key, []).extend(values)

