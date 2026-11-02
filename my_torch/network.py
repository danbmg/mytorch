"""Neural network: a stack of Dense layers trained by mini-batch gradient descent."""

import math
import sys
import time

import numpy as np

from .config import DEFAULT_TRAINING, _normalize_training, normalize_config
from .layers import Dense
from .losses import get_loss
from .optimizers import make_optimizer
from .schedules import make_schedule

# samples per chunk when predicting / evaluating (bounds memory, not results)
PREDICT_BATCH = 8192
# optimisation steps between two flush_subnormals()
FLUSH_EVERY = 100


class TrainingError(RuntimeError):
    pass


def _clip_global_norm(grads, max_norm):
    """Scale all gradients down together when their global L2 norm exceeds max_norm."""
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for g in grads:
            g *= scale
    return norm


def flush_subnormals(arrays):
    """Set to exactly 0 the values too small to be normal floats (|x| < 1.2e-38 in float32).

    Weights that only L2 moves (inputs that are always 0, dead ReLU units) shrink
    towards 0 forever; once "subnormal", every operation on them is 10-100x slower
    on the CPU. 0 is their optimum anyway.
    """
    for a in arrays:
        tiny = np.finfo(a.dtype).tiny
        np.putmask(a, np.abs(a) < tiny, 0)


class Network:
    def __init__(self, config, seed=None):
        config = dict(config)
        if seed is not None:
            config["seed"] = seed
        self.config = normalize_config(config)
        if self.config["seed"] is None:
            # draw a seed and keep it in the config: the network can then be rebuilt identically
            self.config["seed"] = int(np.random.SeedSequence().entropy % 2**32)
        self.seed = self.config["seed"]
        self.dtype = np.dtype(self.config["dtype"])
        rng = np.random.default_rng(self.seed)
        self.layers = []
        n_in = self.config["input_size"]
        for spec in self.config["layers"]:
            act_params = {k: v for k, v in spec.items() if k not in ("size", "activation", "dropout", "init")}
            self.layers.append(Dense(n_in, spec["size"], spec["activation"], rng, init=spec["init"],
                                     dropout=spec["dropout"], dtype=self.dtype,
                                     activation_params=act_params))
            n_in = spec["size"]
        self.loss = get_loss(self.config["loss"])
        # softmax + cross-entropy (or sigmoid + binary cross-entropy): gradient taken on the logits
        self.fused = self.loss.fused_activation == self.config["layers"][-1]["activation"]
        self.labels = self.config["labels"]
        self.meta = {
            "epochs_trained": 0,
            "train_samples": 0,
            "train_time": 0.0,
            "best_epoch": None,
            "best_val_loss": None,
            "stopped_early": False,
        }
        self.history = {}

    # ------------------------------------------------------------------ shape
    @property
    def input_size(self):
        return self.config["input_size"]

    @property
    def output_size(self):
        return self.layers[-1].n_out

    @property
    def n_params(self):
        return sum(layer.n_params for layer in self.layers)

    def summary(self):
        lines = [f"input {self.input_size}"]
        for i, (layer, spec) in enumerate(zip(self.layers, self.config["layers"])):
            extra = f", dropout {spec['dropout']}" if spec["dropout"] else ""
            lines.append(f"layer {i}: {layer.n_in} -> {layer.n_out} {spec['activation']}"
                         f" (init {spec['init']}{extra}), {layer.n_params} params")
        lines.append(f"total {self.n_params} params")
        return "\n".join(lines)

    # ---------------------------------------------------------------- weights
    def get_weights(self):
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_weights(self, weights):
        for layer, (W, b) in zip(self.layers, weights):
            if W.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError(f"weights of shape {W.shape} for a layer of shape {layer.W.shape}")
            layer.W = np.array(W, dtype=self.dtype)
            layer.b = np.array(b, dtype=self.dtype)

    def _params(self):
        return [p for layer in self.layers for p in (layer.W, layer.b)]

    # ---------------------------------------------------------------- forward
    def _check_inputs(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(f"expected inputs of shape (N, {self.input_size}), got {X.shape}")
        return X

    def _forward(self, X, training=False, rng=None):
        out = X.astype(self.dtype, copy=False)
        for layer in self.layers:
            out = layer.forward(out, training, rng)
        return out

    def forward(self, X, training=False, rng=None):
        """Outputs of the network for every row of X, shape (N, n_out)."""
        X = self._check_inputs(X)
        if training and rng is None:
            rng = np.random.default_rng()
        return self._forward(X, training, rng)

    def predict_proba(self, X, mc_samples=0, batch_size=PREDICT_BATCH):
        """Outputs for X, computed by chunks. mc_samples > 0: MC dropout average."""
        if mc_samples:
            return self.predict_mc(X, mc_samples, batch_size)[0]
        X = self._check_inputs(X)
        if len(X) == 0:
            return np.zeros((0, self.output_size), dtype=self.dtype)
        return np.concatenate([self._forward(X[s:s + batch_size]) for s in range(0, len(X), batch_size)])

    def predict_mc(self, X, n_samples=30, batch_size=PREDICT_BATCH):
        """MC dropout: keep dropout on at inference and average n_samples passes.

        Returns (mean, std): the std measures how unsure the network is about each output.
        """
        X = self._check_inputs(X)
        rng = np.random.default_rng([self.seed, 1])
        # Welford's running mean / variance: exact 0 when every pass gives the same output
        mean = np.zeros((len(X), self.output_size))
        m2 = np.zeros_like(mean)
        for t in range(1, n_samples + 1):
            for s in range(0, len(X), batch_size):
                out = self._forward(X[s:s + batch_size], training=True, rng=rng).astype(np.float64)
                delta = out - mean[s:s + batch_size]
                mean[s:s + batch_size] += delta / t
                m2[s:s + batch_size] += delta * (out - mean[s:s + batch_size])
        std = np.sqrt(m2 / n_samples)
        return mean.astype(self.dtype), std.astype(self.dtype)

    def predict(self, X, mc_samples=0):
        """Index of the predicted class for every row of X, shape (N,)."""
        return self._decide(self.predict_proba(X, mc_samples))

    def _decide(self, out):
        if out.shape[1] == 1:
            return (out[:, 0] > 0.5).astype(np.int64)
        return out.argmax(axis=1)

    # --------------------------------------------------------------- training
    def _targets(self, Y, n):
        """Class indices (N,) -> one-hot (N, n_out); 2-D targets are kept as they are."""
        Y = np.asarray(Y)
        if len(Y) != n:
            raise ValueError(f"{n} inputs but {len(Y)} targets")
        if Y.ndim == 1:
            if self.output_size == 1:
                return Y.reshape(-1, 1).astype(self.dtype)
            if len(Y) and (Y.min() < 0 or Y.max() >= self.output_size):
                raise ValueError(f"class indices must be in [0, {self.output_size})")
            onehot = np.zeros((n, self.output_size), dtype=self.dtype)
            onehot[np.arange(n), Y.astype(np.int64)] = 1
            return onehot
        if Y.ndim != 2 or Y.shape[1] != self.output_size:
            raise ValueError(f"expected targets of shape (N,) or (N, {self.output_size}), got {Y.shape}")
        return Y.astype(self.dtype, copy=False)

    def _loss(self, out, y):
        return self.loss.value(out, y, self.layers[-1].z if self.fused else None)

    def _n_correct(self, out, y):
        return int((self._decide(out) == self._decide(y)).sum())

    def _backward(self, out, y):
        last = self.layers[-1]
        only = len(self.layers) == 1
        if self.fused:
            grad = last.backward_from_z(self.loss.fused_grad(out, y), input_grad=not only)
        else:
            grad = last.backward(self.loss.grad(out, y), input_grad=not only)
        for i in range(len(self.layers) - 2, -1, -1):
            grad = self.layers[i].backward(grad, input_grad=i > 0)

    def _grads(self, l2):
        grads = []
        for layer in self.layers:
            # L2: gradient of (l2 / 2) * ||W||^2, biases are not penalised
            grads.append(layer.dW + l2 * layer.W if l2 else layer.dW)
            grads.append(layer.db)
        return grads

    def evaluate(self, X, Y, batch_size=PREDICT_BATCH):
        """Loss (without the L2 term) and accuracy on (X, Y), dropout off."""
        X = self._check_inputs(X)
        Y = self._targets(Y, len(X))
        if len(X) == 0:
            return {"loss": float("nan"), "accuracy": float("nan")}
        total, correct = 0.0, 0
        for s in range(0, len(X), batch_size):
            out = self._forward(X[s:s + batch_size])
            yb = Y[s:s + batch_size]
            total += self._loss(out, yb) * len(yb)
            correct += self._n_correct(out, yb)
        return {"loss": total / len(X), "accuracy": correct / len(X)}

    def fit(self, X, Y, X_val=None, Y_val=None, **hp):
        """Train on (X, Y), validating on (X_val, Y_val) after every epoch.

        hp overrides config["training"] for this call (epochs, batch_size, l2,
        grad_clip, early_stopping, shuffle, eval_train_samples), plus:
        verbose (progress on `log`, default stderr), callback(epoch, history).
        Returns the history of this call: one value per epoch for each metric.
        """
        verbose = hp.pop("verbose", False)
        log = hp.pop("log", None) or sys.stderr
        callback = hp.pop("callback", None)
        unknown = sorted(set(hp) - set(DEFAULT_TRAINING))
        if unknown:
            raise ValueError(f"unknown training setting(s): {', '.join(unknown)}")
        cfg = _normalize_training({**self.config["training"], **hp})

        X = self._check_inputs(X)
        Y = self._targets(Y, len(X))
        if len(X) == 0:
            raise ValueError("no training data")
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            X_val = self._check_inputs(X_val)
            Y_val = self._targets(Y_val, len(X_val))

        n, bs, l2 = len(X), cfg["batch_size"], cfg["l2"]
        epochs = cfg["epochs"]
        es = cfg["early_stopping"]
        if es and not has_val:
            if verbose:
                print("early stopping needs validation data: disabled", file=log)
            es = None
        opt = make_optimizer(self.config["optimizer"])
        sched = make_schedule(self.config["lr_schedule"], opt.lr, epochs)
        params = self._params()
        # seeded by (seed, epochs already trained): resuming a training is reproducible too
        rng = np.random.default_rng([self.seed, self.meta["epochs_trained"]])
        X_eval = Y_eval = None
        if 0 < cfg["eval_train_samples"] < n:
            idx = rng.choice(n, size=cfg["eval_train_samples"], replace=False)
            X_eval, Y_eval = X[idx], Y[idx]
        elif cfg["eval_train_samples"]:
            X_eval, Y_eval = X, Y

        history = {key: [] for key in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time")}
        first_epoch = self.meta["epochs_trained"]
        best_loss, best_weights, best_epoch, wait = math.inf, None, None, 0
        last_val = None
        stopped = False
        progress = verbose and hasattr(log, "isatty") and log.isatty()
        n_batches = math.ceil(n / bs)
        steps = 0
        t_fit = time.perf_counter()

        # a diverging training overflows: reported as a TrainingError below, not as numpy warnings
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(epochs):
                opt.lr = sched.lr(epoch, last_val)
                t0 = time.perf_counter()
                order = rng.permutation(n) if cfg["shuffle"] else np.arange(n)
                run_loss, run_correct = 0.0, 0
                for b, start in enumerate(range(0, n, bs)):
                    idx = order[start:start + bs]
                    xb, yb = X[idx], Y[idx]
                    out = self._forward(xb, training=True, rng=rng)
                    run_loss += self._loss(out, yb) * len(idx)
                    run_correct += self._n_correct(out, yb)
                    self._backward(out, yb)
                    grads = self._grads(l2)
                    if cfg["grad_clip"]:
                        _clip_global_norm(grads, cfg["grad_clip"])
                    opt.step(params, grads)
                    steps += 1
                    if steps % FLUSH_EVERY == 0:
                        flush_subnormals(params + opt.state())
                    if progress and b % max(1, n_batches // 50) == 0:
                        print(f"\r\033[Kepoch {first_epoch + epoch + 1}: {100 * b // n_batches}%",
                              end="", file=log, flush=True)
                if not math.isfinite(run_loss):
                    raise TrainingError(f"the loss diverged (NaN/inf) at epoch {first_epoch + epoch + 1}:"
                                        " lower the learning rate or enable grad_clip")
                if X_eval is not None:
                    train = self.evaluate(X_eval, Y_eval)
                else:
                    train = {"loss": run_loss / n, "accuracy": run_correct / n}
                val = self.evaluate(X_val, Y_val) if has_val else None
                elapsed = time.perf_counter() - t0

                history["epoch"].append(first_epoch + epoch + 1)
                history["train_loss"].append(train["loss"])
                history["train_acc"].append(train["accuracy"])
                history["val_loss"].append(val["loss"] if val else None)
                history["val_acc"].append(val["accuracy"] if val else None)
                history["lr"].append(opt.lr)
                history["time"].append(elapsed)
                if verbose:
                    line = (f"epoch {first_epoch + epoch + 1:>3}/{first_epoch + epochs}"
                            f"  loss {train['loss']:.4f}  acc {100 * train['accuracy']:6.2f}%")
                    if val:
                        line += f"  val_loss {val['loss']:.4f}  val_acc {100 * val['accuracy']:6.2f}%"
                    line += f"  lr {opt.lr:.2e}  {elapsed:.1f}s"
                    print(("\r\033[K" if progress else "") + line, file=log, flush=True)
                if callback:
                    callback(first_epoch + epoch + 1, history)

                last_val = val["loss"] if val else train["loss"]
                if val:
                    if val["loss"] < best_loss - (es["min_delta"] if es else 0.0):
                        best_loss, best_epoch, wait = val["loss"], first_epoch + epoch + 1, 0
                        if es and es["restore_best"]:
                            best_weights = self.get_weights()
                    else:
                        wait += 1
                    if es and wait >= es["patience"]:
                        stopped = True
                        break

        if stopped and verbose:
            print(f"early stopping: no val_loss improvement for {es['patience']} epochs", file=log)
        # keep the weights of the best validation epoch, not those of the last one
        if best_weights is not None and best_epoch != history["epoch"][-1]:
            self.set_weights(best_weights)
            if verbose:
                print(f"weights of epoch {best_epoch} restored (val_loss {best_loss:.4f})", file=log)

        self.meta["epochs_trained"] += len(history["epoch"])
        self.meta["train_samples"] = n
        self.meta["train_time"] += time.perf_counter() - t_fit
        self.meta["stopped_early"] = stopped
        if best_epoch is not None:
            self.meta["best_epoch"] = best_epoch
            self.meta["best_val_loss"] = best_loss
        for key, values in history.items():
            self.history.setdefault(key, []).extend(values)
        return history

    # ------------------------------------------------------------ persistence
    def save(self, path, backup=False):
        from .serialization import save_network
        save_network(self, path, backup=backup)

    @staticmethod
    def load(path):
        from .serialization import load_network
        return load_network(path)
