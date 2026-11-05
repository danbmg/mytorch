"""Helpers shared by the scripts: encoded dataset, JSON files, CSV, graph style."""

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402  (after sys.path: the repository's packages come first)

from my_torch.hpo import set_path  # noqa: E402

# graph style: series colors in a fixed order (color-blind safe neighbours); train dashed, validation solid
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def fail(message):
    print(f"{os.path.basename(sys.argv[0])}: {message}", file=sys.stderr)
    sys.exit(84)


def load_dataset(path, train_samples=None, val_samples=None, seed=0, test=False):
    """Encoded dataset (scripts/dedup_split.py): X_* (N, 773) uint8, Y_* class indices, S_* subset
    of each board, labels, subsets. Train / validation optionally cut to random subsets."""
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if test or not k.endswith("_test")}
    except (OSError, ValueError) as e:
        fail(f"{path}: cannot read dataset ({e})")
    rng = np.random.default_rng(seed)
    for part, n in (("train", train_samples), ("val", val_samples)):
        if f"X_{part}" not in arrays:
            fail(f"{path}: no X_{part}")
        if n and n < len(arrays[f"X_{part}"]):
            idx = np.sort(rng.choice(len(arrays[f"X_{part}"]), size=n, replace=False))
            arrays[f"X_{part}"] = arrays[f"X_{part}"][idx]
            arrays[f"Y_{part}"] = arrays[f"Y_{part}"][idx]
    return arrays


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        fail(f"{path}: {e.strerror}")
    except json.JSONDecodeError as e:
        fail(f"{path}: invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")


def apply_assignments(config, assignments):
    """["training.l2=0", "layers.hidden.dropout=0.3", 'lr_schedule={"name":"step"}'] applied to a config."""
    for item in assignments:
        key, sep, text = item.partition("=")
        if not sep:
            fail(f"'{item}': expected key=value")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text  # a bare word: relu
        set_path(config, key.strip(), value)
    return config


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def describe_layers(config):
    """'256 relu > 128 relu > 5 softmax'"""
    return " > ".join(f"{layer['size']} {layer['activation']}" for layer in config["layers"])


def figure(n_panels=1, width=6.0, height=4.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.8, "axes.grid": True, "grid.color": "#e4e3df"})
    fig, axes = plt.subplots(1, n_panels, figsize=(width * n_panels, height), squeeze=False)
    return fig, list(axes[0])


def log_count_axis(ax):
    """Log x axis labelled 20k, 50k, 100k... instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.set_xscale("log")
    label = FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k")
    ax.xaxis.set_major_formatter(label)
    ax.xaxis.set_minor_locator(LogLocator(subs=(2, 5)))  # 2 and 5 only: every one of 2..9 would overlap
    ax.xaxis.set_minor_formatter(label)


def save_figure(fig, path, title=None):
    import matplotlib.pyplot as plt
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
