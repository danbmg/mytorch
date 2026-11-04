"""Neural network: a stack of Dense layers built from a configuration.

The class owns the model (layers, weights, forward pass, prediction and
evaluation) and the identity of a training (seed, metadata, history); the
mini-batch training itself is in training/, so that the model stays readable
and the training loop can be tested apart.
"""

import numpy as np

from . import losses
from .config import normalize_config
from .data import inputs, targets
from .layers import Dense
from .training.fit import fit as train

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
    def _forward(self, X, training=False, rng=None):
        # uint8 inputs (4x less memory) are converted batch by batch
        out = X.astype(self.dtype, copy=False)
        for layer in self.layers:
            out = layer.forward(out, training, rng)
        return out

    def predict_proba(self, X):
        """Outputs for every row of X, shape (N, n_out), computed by chunks."""
        X = inputs(self, X)
        if len(X) == 0:
            return np.zeros((0, self.output_size), dtype=self.dtype)
        return np.concatenate([self._forward(X[s:s + PREDICT_BATCH]) for s in range(0, len(X), PREDICT_BATCH)])

    def predict(self, X):
        """Index of the predicted class of every row of X, shape (N,)."""
        return self._decide(self.predict_proba(X))

    def _decide(self, out):
        return (out[:, 0] > 0.5).astype(np.int64) if out.shape[1] == 1 else out.argmax(axis=1)

    # --------------------------------------------------------------- training
    def _loss(self, out, y):
        return self.loss.value(out, y, self.layers[-1].z)

    def _backward(self, out, y):
        n = len(self.layers)
        grad = self.loss.backward(self.layers[-1], out, y, input_grad=n > 1)
        for i in range(n - 2, -1, -1):
            grad = self.layers[i].backward(grad, input_grad=i > 0)

    def evaluate(self, X, Y):
        """Loss (without the L2 term) and accuracy, dropout off."""
        X = inputs(self, X)
        Y = targets(self, Y, len(X))
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
        return train(self, X, Y, X_val, Y_val, verbose, log, callback, **hp)

    def save(self, path, backup=False):
        from .serialization import save_network
        save_network(self, path, backup=backup)

    @staticmethod
    def load(path):
        from .serialization import load_network
        return load_network(path)
