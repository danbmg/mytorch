"""Neural network: a stack of Dense layers built from a configuration.

The class owns the model: layers and weights, forward and backward passes,
prediction, evaluation, saving and loading. The mini-batch training loop is in
training/ and only uses the public methods below (forward, compute_loss,
backward, decide), so that the model stays readable on its own.
"""

import numpy as np

from . import losses
from .config import normalize_config
from .data import inputs, targets
from .errors import ConfigError, NetworkFormatError
from .layers import Dense
from .serialization import read_network_file, save_network
from .training.trainer import fit

PREDICT_BATCH = 8192  # rows per chunk when predicting (bounds memory, not results)


class Network:
    def __init__(self, config, seed=None):
        if not isinstance(config, dict):
            raise ConfigError("config: must be an object")
        config = dict(config)
        if seed is not None:
            config["seed"] = seed
        self.config = normalize_config(config)
        if self.config["seed"] is None:
            # drawn once and kept in the config: the network can be rebuilt identically
            self.config["seed"] = int(np.random.SeedSequence().entropy % 2**32)
        self.seed = self.config["seed"]
        self.labels = self.config["labels"]
        self.dtype = np.dtype(self.config["dtype"])
        self.loss = losses.loss(self.config["loss"])
        rng = np.random.default_rng(self.seed)
        sizes = [self.config["input_size"]] + [spec["size"] for spec in self.config["layers"]]
        self.layers = []
        for n_in, n_out, spec in zip(sizes, sizes[1:], self.config["layers"]):
            self.layers.append(Dense(n_in, n_out, spec, rng, self.dtype))
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
            layer.W = np.array(W, dtype=self.dtype)
            layer.b = np.array(b, dtype=self.dtype)

    # ------------------------------------------------------------ passes
    def forward(self, X, training=False, rng=None):
        """Outputs for the rows of X, shape (N, n_out). training: dropout on, drawn from rng."""
        out = X.astype(self.dtype, copy=False)  # uint8 inputs (4x less memory) are converted batch by batch
        for layer in self.layers:
            out = layer.forward(out, training, rng)
        return out

    def compute_loss(self, out, Y):
        """Loss of `out`, the outputs of the last forward(), for the one-hot targets Y."""
        return self.loss.value(out, Y, self.layers[-1].z)

    def backward(self, out, Y):
        """Gradients of the loss after forward(): every layer stores its dW and db."""
        last = len(self.layers) - 1
        grad = self.loss.backward(self.layers[last], out, Y, input_grad=last > 0)
        for i in range(last - 1, -1, -1):
            grad = self.layers[i].backward(grad, input_grad=i > 0)

    def decide(self, out):
        """Class of each row of outputs (or of one-hot targets): the largest one, or > 0.5 for a single output."""
        if out.shape[1] == 1:
            return (out[:, 0] > 0.5).astype(np.int64)
        return out.argmax(axis=1)

    # --------------------------------------------------------- inference
    def predict_proba(self, X):
        """Outputs for every row of X, shape (N, n_out), computed by chunks.

        0/1 inputs (uint8 or bool) go through the first layer by adding the rows of
        its weights (layers.py): the same outputs, without the matrix product."""
        X = inputs(self, X)
        if len(X) == 0:
            return np.zeros((0, self.output_size), dtype=self.dtype)
        chunks = [self._predict_chunk(X[s:s + PREDICT_BATCH]) for s in range(0, len(X), PREDICT_BATCH)]
        return np.concatenate(chunks)

    def _predict_chunk(self, X):
        if X.dtype in (np.uint8, np.bool_) and X.max() <= 1:
            out = self.layers[0].forward_binary(X)
        else:
            out = self.layers[0].forward(X.astype(self.dtype, copy=False))
        for layer in self.layers[1:]:
            out = layer.forward(out)
        return out

    def predict(self, X):
        """Index of the predicted class of every row of X, shape (N,)."""
        return self.decide(self.predict_proba(X))

    def evaluate(self, X, Y):
        """Loss (without the L2 term) and accuracy, dropout off."""
        X = inputs(self, X)
        Y = targets(self, Y, len(X))
        if len(X) == 0:
            return {"loss": float("nan"), "accuracy": float("nan")}
        total = 0.0
        correct = 0
        for s in range(0, len(X), PREDICT_BATCH):
            out = self.forward(X[s:s + PREDICT_BATCH])
            y = Y[s:s + PREDICT_BATCH]
            total += self.compute_loss(out, y) * len(y)
            correct += int((self.decide(out) == self.decide(y)).sum())
        return {"loss": total / len(X), "accuracy": correct / len(X)}

    # ---------------------------------------------------------- training
    def fit(self, X, Y, X_val=None, Y_val=None, verbose=False, log=None, callback=None, **hp):
        """Train on (X, Y), validating on (X_val, Y_val) after every epoch.

        hp overrides config["training"] for this call (epochs, batch_size, l2...).
        verbose: one line per epoch on `log` (stderr); callback(epoch, history) after
        each epoch. Returns the history of this call (one value per epoch per metric).
        A new call on a trained network resumes the training.
        """
        return fit(self, X, Y, X_val, Y_val, verbose, log, callback, **hp)

    # ------------------------------------------------------------- files
    def save(self, path, backup=False):
        """Write the network to `path` atomically; backup=True keeps the replaced file as path.bak."""
        save_network(self, path, backup=backup)

    @staticmethod
    def load(path):
        """The network saved in a .nn file. Raises NetworkFormatError if the file is unusable."""
        header, arrays = read_network_file(path)
        try:
            net = Network(header["config"])
        except (KeyError, ConfigError) as e:
            raise NetworkFormatError(f"{path}: invalid network config ({e})") from None
        try:
            net.set_weights([(arrays[f"W{i}"], arrays[f"b{i}"]) for i in range(len(net.layers))])
        except KeyError as e:
            raise NetworkFormatError(f"{path}: missing weights {e}") from None
        except ValueError as e:
            raise NetworkFormatError(f"{path}: corrupted weights ({e})") from None
        net.meta.update(header.get("meta") or {})
        net.history = header.get("history") or {}
        return net


def load_network(path):
    """Same as Network.load(path)."""
    return Network.load(path)
