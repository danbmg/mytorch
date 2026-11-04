"""The two modes of my_torch_analyzer.

--predict: one label per line on stdout, in the order of the file, nothing
else (the summary for a person goes to stderr, and only at a terminal).
--train: 10% of the file held out for validation (early stopping, report), the
destination checked before the training so that a bad path fails at once, and
the replaced network kept as .bak.
"""

import os
import sys

import my_torch  # noqa: F401  (before numpy: sets the BLAS threads)
from my_torch import load_network
from my_torch.metrics import confusion_matrix, report

import numpy as np  # noqa: E402

from .boards import read_boards
from .errors import log
from .fen import INPUT_SIZE
from .labels import network_labels
from .split import stratified_split

VAL_FRACTION = 0.1  # --train: share of CHESSFILE held out for validation (early stopping, report)
MIN_BOARDS_FOR_VAL = 100  # below, everything is used for training


def load(path):
    net = load_network(path)
    if net.input_size != INPUT_SIZE:
        raise ValueError(f"{path}: the network takes {net.input_size} inputs, but a chessboard is encoded with {INPUT_SIZE}")
    try:
        return net, network_labels(net)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


def _scores(title, y_true, y_pred, labels):
    return f"\n{title}:\n{report(confusion_matrix(y_true, y_pred, len(labels)), labels)}"


def predict(opts):
    net, labels = load(opts.loadfile)
    X, y = read_boards(opts.chessfile, labels)
    pred = net.predict(X)
    sys.stdout.write("".join(labels[i] + "\n" for i in pred))
    sys.stdout.flush()
    # summary only for a person at a terminal: a script reading the output sees nothing more
    if (y >= 0).any() and sys.stderr.isatty():
        log(_scores(f"{int((y >= 0).sum())} boards with an expected output", y[y >= 0], pred[y >= 0], labels))


def _writable(target):
    """Fail now rather than after the training if the network cannot be saved there."""
    directory = os.path.dirname(os.path.abspath(target))
    if not os.path.isdir(directory) or os.path.isdir(target) or not os.access(directory, os.W_OK):
        raise ValueError(f"cannot save the network in {target}: no such directory or not writable")


def _holdout(y, seed):
    """(training indices, validation indices): 10% held out when there are enough boards."""
    if len(y) >= MIN_BOARDS_FOR_VAL:
        return stratified_split(y, (1 - VAL_FRACTION, VAL_FRACTION), seed)
    return np.arange(len(y)), np.arange(0)


def train(opts):
    net, labels = load(opts.loadfile)
    target = opts.save or opts.loadfile
    _writable(target)
    X, y = read_boards(opts.chessfile, labels, require_labels=True)
    train_idx, val_idx = _holdout(y, net.seed)
    counts = ", ".join(f"{label} {c}" for label, c in zip(labels, np.bincount(y, minlength=len(labels))))
    log(f"{opts.chessfile}: {len(X)} boards ({counts}); training on {len(train_idx)}, validating on {len(val_idx)}")
    log(f"network {opts.loadfile}: {' > '.join(str(s['size']) + ' ' + s['activation'] for s in net.config['layers'])},"
        f" {net.n_params} parameters, {net.meta['epochs_trained']} epochs already trained")
    net.fit(X[train_idx], y[train_idx], X[val_idx], y[val_idx], verbose=True,
            **({"epochs": opts.epochs} if opts.epochs else {}))
    if len(val_idx):
        log(_scores(f"validation ({len(val_idx)} boards)", y[val_idx], net.predict(X[val_idx]), labels) + "\n")
    existed = os.path.exists(target)
    net.save(target, backup=existed)  # never destroy a network silently: the replaced file is kept as .bak
    log(f"network saved in {target}" + (f" (previous version kept as {target}.bak)" if existed else ""))
