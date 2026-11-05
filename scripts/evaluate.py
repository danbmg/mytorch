#!/usr/bin/env python3
"""Accuracy of trained networks on the test set, in 5 classes and in 3 classes (color ignored).

usage: python3 scripts/evaluate.py NET.nn [NET.nn ...] --data data/chess.npz

For each network: the per-class report and confusion matrix of the 5 labels, then
the same with the color of the check ignored (Nothing / Check / Checkmate). The
predictions are those of my_torch_analyzer --predict (averaged over the 4
symmetries of each board); the accuracy of the network on the boards alone is
given for comparison.
"""

import argparse
import sys

from common import fail, load_dataset

from analyzer.augment import symmetric_proba
from my_torch import NetworkFormatError, load_network
from my_torch.metrics import confusion_matrix, report

import numpy as np


def color_blind(labels):
    """The 3 color-free labels and the index of each of the 5 labels among them."""
    names = list(dict.fromkeys(label.replace("White", "").replace("Black", "").strip() for label in labels))
    index = np.array([names.index(label.replace("White", "").replace("Black", "").strip()) for label in labels])
    return names, index


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("networks", nargs="+")
    p.add_argument("--data", required=True)
    args = p.parse_args(argv)
    data = load_dataset(args.data, test=True)
    X = data["X_test"]
    y = data["Y_test"]
    for path in args.networks:
        try:
            net = load_network(path)
        except NetworkFormatError as e:
            fail(str(e))
        labels = net.labels
        plain = net.predict(X)
        pred = symmetric_proba(net, X, labels).argmax(axis=1)
        names, index = color_blind(labels)
        print(f"== {path}: {len(X)} test boards")
        print(f"accuracy: {(pred == y).mean():.4f} averaged over the symmetries,"
              f" {(plain == y).mean():.4f} on the boards alone\n")
        print(report(confusion_matrix(y, pred, len(labels)), labels))
        print(f"\n-- color ignored ({' / '.join(names)})\n")
        print(report(confusion_matrix(index[y], index[pred], len(names)), names))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
