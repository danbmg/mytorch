#!/usr/bin/env python3
"""Learning curve: the same network trained on more and more data (T2.5). Is more data worth it?

usage: python3 scripts/learning_curve.py --data data/chess.npz [--config configs/basic.conf] [--epochs 20] [--out OUT.png]

Writes OUT.png (accuracy and training time vs number of training boards) and OUT.csv.
"""

import argparse
import os
import sys

from common import COLORS, ROOT, figure, load_dataset, load_json, log_count_axis, save_figure, write_csv

from my_torch import Network

import numpy as np


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", required=True)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "basic.conf"))
    p.add_argument("--sizes", type=int, nargs="+", default=[25000, 50000, 100000, 200000, 400000, 0], help="0 = all")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default="results/learning_curve.png")
    args = p.parse_args(argv)
    config = load_json(args.config)
    data = load_dataset(args.data, val_samples=30000)
    config["training"] = {**config.get("training", {}), "early_stopping": None}
    total = len(data["X_train"])
    order = np.random.default_rng(0).permutation(total)
    rows = []
    for size in sorted(s if 0 < s < total else total for s in args.sizes):
        idx = np.sort(order[:size])
        h = Network(config).fit(data["X_train"][idx], data["Y_train"][idx], data["X_val"], data["Y_val"], epochs=args.epochs)
        j = int(np.argmin(h["val_loss"]))
        rows.append({"train_boards": size, "best_epoch": h["epoch"][j], "train_acc": round(h["train_acc"][j], 4),
                     "val_acc": round(h["val_acc"][j], 4), "acc_gap": round(h["train_acc"][j] - h["val_acc"][j], 4),
                     "time_s": round(sum(h["time"]), 1)})
        print(f"  {size:>7} boards: val_acc {100 * rows[-1]['val_acc']:.2f}%, gap {100 * rows[-1]['acc_gap']:.1f}"
              f" points, {rows[-1]['time_s']:.0f}s", file=sys.stderr)
    out = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_csv(out + ".csv", rows)

    n = [r["train_boards"] for r in rows]
    fig, (ax, ax_time) = figure(2, width=5.6)
    ax.plot(n, [r["train_acc"] for r in rows], color=COLORS[0], linestyle="--", marker="o", label="train")
    ax.plot(n, [r["val_acc"] for r in rows], color=COLORS[0], marker="o", label="validation")
    for x, r in zip(n, rows):
        ax.annotate(f"{100 * r['val_acc']:.1f}%", (x, r["val_acc"]), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8)
    ax.set(xlabel="training boards", ylabel="accuracy at the best epoch", title="accuracy vs amount of data")
    ax.legend()
    ax_time.plot(n, [r["time_s"] for r in rows], color=COLORS[1], marker="o")
    ax_time.set(xlabel="training boards", ylabel="training time (s)", title="cost: training time")
    for a in (ax, ax_time):
        log_count_axis(a)
    save_figure(fig, out + ".png", "Learning curve: what more data is worth")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
