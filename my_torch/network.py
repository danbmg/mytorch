"""Neural network: a stack of Dense layers trained by mini-batch gradient descent."""

import math
import sys
import time

import numpy as np

from . import losses
from .config import TRAINING, normalize_config, normalize_training
from .errors import TrainingError
from .layers import Dense
from .optimizers import Optimizer
from .schedules import Schedule
from .training import batch
from .training.early_stopping import EarlyStopping
from .training.progress import Progress

PREDICT_BATCH = 8192  # rows per chunk when predicting (bounds memory, not results)


class Network:
    def __init__(self, config, seed=None):
        config = dict(config, **({"seed": seed} if seed is not None else {}))
        self.config = normalize_config(config)
        if self.config["seed"] is None:
            # drawn once and kept in the config: the network can be rebuilt identically
            self.config["seed"] = int(np.random.SeedSequence().entropy % 2**32)
        self.seed, self.labels = self.config["seed"], self.config["labels"]
        self.dtype, self.loss = np.dtype(self.config["dtype"]), losses.loss(self.config["loss"])
        rng = np.random.default_rng(self.seed)
        sizes = [self.config["input_size"]] + [spec["size"] for spec in self.config["layers"]]
        self.layers = [Dense(n_in, n_out, spec, rng, self.dtype)
                       for n_in, n_out, spec in zip(sizes, sizes[1:], self.config["layers"])]
        self.meta = {"epochs_trained": 0, "train_samples": 0, "train_time": 0.0,
                     "best_epoch": None, "best_val_loss": None, "stopped_early": False}
        self.history = {}

    @property
    def input_size(self):
        return self.config["input_size"]

    @property
    def output_size(self):
        return self.layers[-1].W.shape[1]

    @property
    def n_params(self):
        return sum(layer.n_params for layer in self.layers)

    def get_weights(self):
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_weights(self, weights):
        for layer, (W, b) in zip(self.layers, weights):
            if W.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError(f"weights of shape {W.shape} for a layer of shape {layer.W.shape}")
            layer.W, layer.b = np.array(W, dtype=self.dtype), np.array(b, dtype=self.dtype)

    # -------------------------------------------------------------- inference
    def _inputs(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(f"expected inputs of shape (N, {self.input_size}), got {X.shape}")
        return X

    def _forward(self, X, training=False, rng=None):
        # uint8 inputs (4x less memory) are converted batch by batch
        out = X.astype(self.dtype, copy=False)
        for layer in self.layers:
            out = layer.forward(out, training, rng)
        return out

    def predict_proba(self, X):
        """Outputs for every row of X, shape (N, n_out), computed by chunks."""
        X = self._inputs(X)
        if len(X) == 0:
            return np.zeros((0, self.output_size), dtype=self.dtype)
        return np.concatenate([self._forward(X[s:s + PREDICT_BATCH]) for s in range(0, len(X), PREDICT_BATCH)])

    def predict(self, X):
        """Index of the predicted class of every row of X, shape (N,)."""
        return self._decide(self.predict_proba(X))

    def _decide(self, out):
        return (out[:, 0] > 0.5).astype(np.int64) if out.shape[1] == 1 else out.argmax(axis=1)

    # --------------------------------------------------------------- training
    def _targets(self, Y, n):
        """Class indices (N,) -> one-hot (N, n_out); 2-D targets are kept as they are."""
        Y = np.asarray(Y)
        if len(Y) != n:
            raise ValueError(f"{n} inputs but {len(Y)} targets")
        if Y.ndim == 2 and Y.shape[1] == self.output_size:
            return Y.astype(self.dtype, copy=False)
        if Y.ndim != 1:
            raise ValueError(f"expected targets of shape (N,) or (N, {self.output_size}), got {Y.shape}")
        if self.output_size == 1:
            return Y.reshape(-1, 1).astype(self.dtype)
        if len(Y) and (Y.min() < 0 or Y.max() >= self.output_size):
            raise ValueError(f"class indices must be in [0, {self.output_size})")
        onehot = np.zeros((n, self.output_size), dtype=self.dtype)
        onehot[np.arange(n), Y.astype(np.int64)] = 1
        return onehot

    def _loss(self, out, y):
        return self.loss.value(out, y, self.layers[-1].z)

    def _backward(self, out, y):
        n = len(self.layers)
        grad = self.loss.backward(self.layers[-1], out, y, input_grad=n > 1)
        for i in range(n - 2, -1, -1):
            grad = self.layers[i].backward(grad, input_grad=i > 0)

    def evaluate(self, X, Y):
        """Loss (without the L2 term) and accuracy, dropout off."""
        X = self._inputs(X)
        Y = self._targets(Y, len(X))
        if len(X) == 0:
            return {"loss": float("nan"), "accuracy": float("nan")}
        total, correct = 0.0, 0
        for s in range(0, len(X), PREDICT_BATCH):
            out, y = self._forward(X[s:s + PREDICT_BATCH]), Y[s:s + PREDICT_BATCH]
            total += self._loss(out, y) * len(y)
            correct += int((self._decide(out) == self._decide(y)).sum())
        return {"loss": total / len(X), "accuracy": correct / len(X)}

    def fit(self, X, Y, X_val=None, Y_val=None, verbose=False, log=None, callback=None, **hp):
        """Train on (X, Y), validating on (X_val, Y_val) after every epoch.

        hp overrides config["training"] for this call (epochs, batch_size, l2...).
        verbose: one line per epoch on `log` (stderr); callback(epoch, history) after
        each epoch. Returns the history of this call (one value per epoch per metric).
        A new call on a trained network resumes the training.
        """
        progress = Progress(log or sys.stderr, verbose)
        unknown = sorted(set(hp) - set(TRAINING))
        if unknown:
            raise ValueError(f"unknown training setting(s): {', '.join(unknown)}")
        cfg = normalize_training({**self.config["training"], **hp})
        X = self._inputs(X)
        Y = self._targets(Y, len(X))
        if len(X) == 0:
            raise ValueError("no training data")
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            X_val = self._inputs(X_val)
            Y_val = self._targets(Y_val, len(X_val))
        es = cfg["early_stopping"] if has_val else None
        if cfg["early_stopping"] and not has_val:
            progress.message("early stopping needs validation data: disabled")

        n, bs, epochs = len(X), cfg["batch_size"], cfg["epochs"]
        opt = Optimizer(**self.config["optimizer"])
        sched = Schedule(self.config["lr_schedule"], opt.lr, epochs)
        params = [p for layer in self.layers for p in (layer.W, layer.b)]
        # seeded by (seed, epochs already trained): resuming is reproducible too
        rng = np.random.default_rng([self.seed, self.meta["epochs_trained"]])
        X_eval, Y_eval = X, Y
        if 0 < cfg["eval_train_samples"] < n:
            idx = rng.choice(n, size=cfg["eval_train_samples"], replace=False)
            X_eval, Y_eval = X[idx], Y[idx]

        history = {key: [] for key in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time")}
        first = self.meta["epochs_trained"]
        stopping, last_loss, stopped = EarlyStopping(self, es), None, False
        n_batches, steps, t_fit = math.ceil(n / bs), 0, time.perf_counter()
        # a diverging training overflows: reported below as a TrainingError, not as numpy warnings
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(first + 1, first + epochs + 1):
                opt.lr = sched.lr(epoch - first - 1, last_loss)
                t0 = time.perf_counter()
                order = rng.permutation(n) if cfg["shuffle"] else np.arange(n)
                run_loss, run_correct = 0.0, 0
                for b, start in enumerate(range(0, n, bs)):
                    idx = order[start:start + bs]
                    loss, correct = batch.train_batch(self, opt, params, X[idx], Y[idx], cfg, rng)
                    run_loss, run_correct = run_loss + loss, run_correct + correct
                    steps += 1
                    if steps % batch.FLUSH_EVERY == 0:
                        batch.flush_subnormals(params + opt.state())
                    progress.batch(epoch, b, n_batches)
                if not math.isfinite(run_loss):
                    raise TrainingError(f"the loss diverged (NaN/inf) at epoch {epoch}:"
                                        " lower the learning rate or enable grad_clip")
                train = (self.evaluate(X_eval, Y_eval) if cfg["eval_train_samples"]
                         else {"loss": run_loss / n, "accuracy": run_correct / n})
                val = self.evaluate(X_val, Y_val) if has_val else {"loss": None, "accuracy": None}
                for key, value in (("epoch", epoch), ("train_loss", train["loss"]), ("train_acc", train["accuracy"]),
                                   ("val_loss", val["loss"]), ("val_acc", val["accuracy"]), ("lr", opt.lr),
                                   ("time", time.perf_counter() - t0)):
                    history[key].append(value)
                progress.epoch(epoch, first + epochs, train, val, opt.lr, history["time"][-1])
                if callback:
                    callback(epoch, history)
                last_loss = val["loss"] if has_val else train["loss"]
                if has_val and stopping.update(epoch, val["loss"]):
                    stopped = True
                    break

        if stopped:
            progress.message(f"early stopping: no val_loss improvement for {stopping.patience} epochs")
        # keep the weights of the best validation epoch, not those of the last one
        if stopping.restore(history["epoch"][-1]):
            progress.message(f"weights of epoch {stopping.best_epoch} restored (val_loss {stopping.best_loss:.4f})")
        self.meta.update(epochs_trained=first + len(history["epoch"]), train_samples=n, stopped_early=stopped,
                         train_time=self.meta["train_time"] + time.perf_counter() - t_fit)
        if stopping.best_epoch is not None:
            self.meta.update(best_epoch=stopping.best_epoch, best_val_loss=stopping.best_loss)
        for key, values in history.items():
            self.history.setdefault(key, []).extend(values)
        return history

    def save(self, path, backup=False):
        from .serialization import save_network
        save_network(self, path, backup=backup)

    @staticmethod
    def load(path):
        from .serialization import load_network
        return load_network(path)
