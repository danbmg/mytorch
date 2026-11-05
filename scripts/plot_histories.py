#!/usr/bin/env python3
"""Validation curves of several trained networks on one graph, against the number of
training boards seen (T2.5): networks trained on more or less data compare at equal
computation, epoch by epoch.

usage: python3 scripts/plot_histories.py NET.nn LABEL [NET.nn LABEL ...] --out OUT.png [--title TITLE]

Writes OUT.png (validation accuracy and validation loss, one panel each, the best
epoch of each network marked by a dot) and OUT.csv (every point of every curve).
The curves come from the history saved in each .nn file: nothing is trained again.
"""

import argparse
import os
import sys

from common import COLORS, fail, figure, save_figure, write_csv

from my_torch import NetworkFormatError, read_header

TEXT = "#52514e"  # labels are written in text ink, the line next to them carries the color


def read_run(path, label):
    """(label, history, boards per epoch, best epoch) of a trained network."""
    try:
        header = read_header(path)
    except NetworkFormatError as e:
        fail(str(e))
    history = header.get("history") or {}
    meta = header.get("meta") or {}
    if not history.get("val_acc") or not meta.get("train_samples"):
        fail(f"{path}: no validation history to plot")
    return label, history, meta["train_samples"], meta.get("best_epoch")


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("pairs", nargs="+", metavar="NET.nn LABEL")
    p.add_argument("--out", required=True)
    p.add_argument("--title")
    args = p.parse_args(argv)
    if len(args.pairs) % 2:
        fail("expected pairs of arguments: NET.nn LABEL")
    if len(args.pairs) // 2 > len(COLORS):
        fail(f"at most {len(COLORS)} networks on one graph")
    runs = [read_run(path, label) for path, label in zip(args.pairs[::2], args.pairs[1::2])]

    fig, axes = figure(2, width=5.4, height=4.0)
    panels = (("val_acc", "validation accuracy (%)", 100), ("val_loss", "validation loss", 1))
    rows = []
    for (label, history, per_epoch, best), color in zip(runs, COLORS):
        seen = [epoch * per_epoch / 1e6 for epoch in history["epoch"]]
        for ax, (key, title, scale) in zip(axes, panels):
            values = [v * scale for v in history[key]]
            ax.plot(seen, values, color=color, linewidth=2, label=label)
            if best in history["epoch"]:
                i = history["epoch"].index(best)
                ax.plot(seen[i], values[i], "o", color=color, markersize=6,
                        markeredgecolor="white", markeredgewidth=1.5)  # surface ring on the line
            ax.annotate(label, (seen[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        for epoch, x, acc, loss in zip(history["epoch"], seen, history["val_acc"], history["val_loss"]):
            rows.append({"network": label, "epoch": epoch, "boards_seen_millions": round(x, 3),
                         "val_acc": round(acc, 5), "val_loss": round(loss, 5)})
    for ax, (_, title, _) in zip(axes, panels):
        ax.set(xlabel="training boards seen (millions)", title=title)
        ax.margins(x=0.18)  # room for the direct labels on the right
        ax.set_xlim(left=0)
    axes[0].legend(loc="lower right")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_figure(fig, args.out, args.title)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    write_csv(csv_path, rows)
    print(f"written {args.out} and {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
