#!/usr/bin/env python3
"""Accuracy of trained networks on the test set, in 5 classes and in 3 classes (colour ignored).

usage: python3 scripts/evaluate.py NET.nn [NET.nn ...] --data data/chess.npz

For each network: the per-class report and confusion matrix of the 5 labels, then
the same with the colour of the check ignored (Nothing / Check / Checkmate).
"""

import argparse
import sys

from common import fail, load_dataset

from my_torch import NetworkFormatError, load_network
from my_torch.metrics import confusion_matrix, report

import numpy as np


def colour_blind(labels):
    """The 3 colour-free labels and the index of each of the 5 labels among them."""
    names = list(dict.fromkeys(label.replace("White", "").replace("Black", "").strip() for label in labels))
    index = np.array([names.index(label.replace("White", "").replace("Black", "").strip()) for label in labels])
    return names, index


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("networks", nargs="+")
    p.add_argument("--data", required=True)
    args = p.parse_args(argv)
    data = load_dataset(args.data, test=True)
    X, y = data["X_test"], data["Y_test"]
    for path in args.networks:
        try:
            net = load_network(path)
        except NetworkFormatError as e:
            fail(str(e))
        labels, pred = net.labels, net.predict(X)
        names, index = colour_blind(labels)
        print(f"== {path}: {len(X)} test boards\n")
        print(report(confusion_matrix(y, pred, len(labels)), labels))
        print(f"\n-- colour ignored ({' / '.join(names)})\n")
        print(report(confusion_matrix(index[y], index[pred], len(names)), names))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
