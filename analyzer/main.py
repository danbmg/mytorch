"""my_torch_analyzer: state of chessboards (check, checkmate...) predicted by a my_torch network."""

import os
import sys

# one training at a time: 2 BLAS threads suits both our machines (docs/benchmarks.md); MY_TORCH_THREADS overrides
os.environ.setdefault("MY_TORCH_THREADS", "2")

import my_torch  # noqa: E402,F401  (before numpy: sets the BLAS threads)
from my_torch import ConfigError, NetworkFormatError, TrainingError, load_network  # noqa: E402
from my_torch.metrics import confusion_matrix, report  # noqa: E402

import numpy as np  # noqa: E402

from .cli import USAGE, UsageError, parse_args  # noqa: E402
from .dataset import read_boards, stratified_split  # noqa: E402
from .fen import INPUT_SIZE  # noqa: E402
from .labels import network_labels  # noqa: E402

VAL_FRACTION = 0.1  # --train: share of CHESSFILE held out for validation (early stopping, report)
MIN_BOARDS_FOR_VAL = 100  # below, everything is used for training


def log(message):
    print(message, file=sys.stderr, flush=True)


def load(path):
    net = load_network(path)
    if net.input_size != INPUT_SIZE:
        raise ValueError(f"{path}: the network takes {net.input_size} inputs, but a chessboard is encoded with {INPUT_SIZE}")
    try:
        return net, network_labels(net)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


def predict(opts):
    net, labels = load(opts.loadfile)
    X, y = read_boards(opts.chessfile, labels)
    pred = net.predict(X)
    sys.stdout.write("".join(labels[i] + "\n" for i in pred))
    sys.stdout.flush()
    # summary only for a person at a terminal: a script reading the output sees nothing more
    if (y >= 0).any() and sys.stderr.isatty():
        log(f"\n{int((y >= 0).sum())} boards with an expected output:\n"
            f"{report(confusion_matrix(y[y >= 0], pred[y >= 0], len(labels)), labels)}")


def train(opts):
    net, labels = load(opts.loadfile)
    target = opts.save or opts.loadfile
    # fail now rather than after the training
    directory = os.path.dirname(os.path.abspath(target))
    if not os.path.isdir(directory) or os.path.isdir(target) or not os.access(directory, os.W_OK):
        raise ValueError(f"cannot save the network in {target}: no such directory or not writable")
    X, y = read_boards(opts.chessfile, labels, require_labels=True)
    if len(X) >= MIN_BOARDS_FOR_VAL:
        train_idx, val_idx = stratified_split(y, (1 - VAL_FRACTION, VAL_FRACTION), net.seed)
    else:
        train_idx, val_idx = np.arange(len(X)), np.arange(0)
    counts = ", ".join(f"{label} {c}" for label, c in zip(labels, np.bincount(y, minlength=len(labels))))
    log(f"{opts.chessfile}: {len(X)} boards ({counts}); training on {len(train_idx)}, validating on {len(val_idx)}")
    log(f"network {opts.loadfile}: {' > '.join(str(s['size']) + ' ' + s['activation'] for s in net.config['layers'])},"
        f" {net.n_params} parameters, {net.meta['epochs_trained']} epochs already trained")
    net.fit(X[train_idx], y[train_idx], X[val_idx], y[val_idx], verbose=True,
            **({"epochs": opts.epochs} if opts.epochs else {}))
    if len(val_idx):
        log(f"\nvalidation ({len(val_idx)} boards):\n"
            f"{report(confusion_matrix(y[val_idx], net.predict(X[val_idx]), len(labels)), labels)}\n")
    existed = os.path.exists(target)
    net.save(target, backup=existed)  # never destroy a network silently: the replaced file is kept as .bak
    log(f"network saved in {target}" + (f" (previous version kept as {target}.bak)" if existed else ""))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if "-h" in args or "--help" in args:
        print(USAGE)
        return 0
    try:
        opts = parse_args(args)
        if opts.mode == "predict":
            predict(opts)
        else:
            train(opts)
        return 0
    except BrokenPipeError:
        # output piped into a command that stopped reading (| head): not an error
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (UsageError, ValueError, NetworkFormatError, ConfigError, TrainingError) as e:
        log(f"my_torch_analyzer: {e}")
    except OSError as e:
        log(f"my_torch_analyzer: {e.filename + ': ' if e.filename else ''}{e.strerror or e}")
    except MemoryError:
        log("my_torch_analyzer: not enough memory for this file")
    except KeyboardInterrupt:
        log("\nmy_torch_analyzer: interrupted, no network written")
    return 84
