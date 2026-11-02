"""Helpers shared by the scripts: encoded dataset files, config files, CSV."""

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import my_torch  # noqa: E402,F401  (before numpy: sets the BLAS threads)
from my_torch.hpo import set_path  # noqa: E402

import numpy as np  # noqa: E402


def fail(message):
    print(f"{os.path.basename(sys.argv[0])}: {message}", file=sys.stderr)
    sys.exit(84)


def _subset(X, Y, n, rng):
    if n is None or n >= len(X):
        return X, Y
    idx = np.sort(rng.choice(len(X), size=n, replace=False))
    return X[idx], Y[idx]


def load_dataset(path, train_samples=None, val_samples=None, seed=0):
    """Encoded dataset: .npz with X_train, Y_train, X_val, Y_val (X: (N, input_size),
    any numeric dtype, uint8 recommended; Y: class indices), optionally X_test, Y_test,
    labels. Returns a dict of arrays; train/val optionally cut to random subsets."""
    try:
        with np.load(path, allow_pickle=False) as data:
            # arrays are read one by one: the test set is not loaded when unused
            arrays = {key: data[key] for key in data.files if key not in ("X_test", "Y_test")}
    except (OSError, ValueError) as e:
        fail(f"{path}: cannot read dataset ({e})")
    missing = [k for k in ("X_train", "Y_train", "X_val", "Y_val") if k not in arrays]
    if missing:
        fail(f"{path}: missing array(s) {', '.join(missing)}")
    rng = np.random.default_rng(seed)
    arrays["X_train"], arrays["Y_train"] = _subset(arrays["X_train"], arrays["Y_train"], train_samples, rng)
    arrays["X_val"], arrays["Y_val"] = _subset(arrays["X_val"], arrays["Y_val"], val_samples, rng)
    return arrays


def load_json(path, what="config"):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        fail(f"{path}: cannot read {what} ({e.strerror})")
    except json.JSONDecodeError as e:
        fail(f"{path}: invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")


def parse_value(text):
    """'0.1' -> 0.1, 'null' -> None, '[1, 2]' -> [1, 2], 'relu' -> 'relu'."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_assignments(config, assignments):
    """Apply ["training.l2=0", "layers.hidden.dropout=0.3", ...] to a config."""
    for item in assignments:
        if "=" not in item:
            fail(f"'{item}': expected key=value")
        key, value = item.split("=", 1)
        set_path(config, key.strip(), parse_value(value.strip()))
    return config


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def describe_layers(config):
    """'256 relu > 128 relu > 5 softmax'"""
    return " > ".join(f"{layer['size']} {layer['activation']}" for layer in config["layers"])
