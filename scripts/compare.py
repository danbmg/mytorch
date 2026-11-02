#!/usr/bin/env python3
"""Train 2 to 8 configs at the same time on the same data, and draw their learning curves on one graph.

usage: python3 scripts/compare.py --data DATA.npz CONFIG... [options]
       python3 scripts/compare.py --data DATA.npz --base CONFIG --variant "LABEL: key=value ..." [--variant ...]

A variant is the base config with some settings changed (dotted paths):
  --variant "none: layers.hidden.dropout=0 training.l2=0 training.early_stopping=null"
Writes OUT.png (loss and accuracy per epoch, train dashed / validation solid,
plus the learning rate when it differs) and OUT.csv (metrics at the best epoch).
"""

import argparse
import copy
import os
import sys

from common import COLORS, apply_assignments, describe_layers, fail, figure, load_dataset, load_json, save_figure, write_csv

from my_torch.config import ConfigError, normalize_config
from my_torch.parallel import train_many


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("configs", nargs="*")
    p.add_argument("--data", required=True)
    p.add_argument("--base")
    p.add_argument("--variant", action="append", default=[], help='"label: key=value key=value"')
    p.add_argument("--epochs", type=int)
    p.add_argument("--train-samples", type=int, dest="train_samples")
    p.add_argument("--val-samples", type=int, dest="val_samples")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--title")
    p.add_argument("--out", default="results/compare.png")
    args = p.parse_args(argv)

    runs = [(os.path.splitext(os.path.basename(path))[0], load_json(path)) for path in args.configs]
    if args.variant and not args.base:
        fail("--variant needs --base")
    for text in args.variant:
        label, _, changes = text.partition(":")
        runs.append((label.strip(), apply_assignments(copy.deepcopy(load_json(args.base)), changes.split())))
    if not 2 <= len(runs) <= len(COLORS):
        fail(f"{len(runs)} configs: compare between 2 and {len(COLORS)}")
    for label, config in runs:
        try:
            normalize_config(config)
        except ConfigError as e:
            fail(f"{label}: {e}")
    data = load_dataset(args.data, args.train_samples, args.val_samples)
    print(f"{len(runs)} configs on {len(data['X_train'])} training / {len(data['X_val'])} validation boards",
          file=sys.stderr)

    def show(r):
        label = runs[r["index"]][0]
        print(f"  {label}: " + (f"error: {r['error']}" if r.get("error") else
                                f"val_loss {r['val_loss']:.4f} val_acc {100 * r['val_acc']:.2f}% ({r['epochs']} epochs,"
                                f" {r['time']:.0f}s)"), file=sys.stderr)

    results = train_many([c for _, c in runs], data["X_train"], data["Y_train"], data["X_val"], data["Y_val"],
                         workers=args.workers, fit_kwargs={"epochs": args.epochs} if args.epochs else {}, on_result=show)

    rows = []
    for (label, _), r in zip(runs, results):
        if r.get("error"):
            rows.append({"config": label, "error": r["error"]})
            continue
        h, j = r["history"], r["history"]["epoch"].index(r["best_epoch"])
        rows.append({"config": label, "layers": describe_layers(r["config"]), "best_epoch": r["best_epoch"],
                     "epochs": r["epochs"], "train_loss": round(h["train_loss"][j], 4), "val_loss": round(h["val_loss"][j], 4),
                     "train_acc": round(h["train_acc"][j], 4), "val_acc": round(h["val_acc"][j], 4),
                     "last_acc_gap": round(h["train_acc"][-1] - h["val_acc"][-1], 4),
                     "val_macro_f1": round(r["val_macro_f1"], 4), "time_s": round(r["time"], 1)})
    out = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_csv(out + ".csv", rows)

    ok = [(label, r, COLORS[i]) for i, ((label, _), r) in enumerate(zip(runs, results)) if not r.get("error")]
    show_lr = len({tuple(r["history"]["lr"]) for _, r, _ in ok}) > 1
    fig, axes = figure(3 if show_lr else 2, width=5.2)
    for ax, name in zip(axes, ("loss", "acc")):
        for label, r, color in ok:
            h, j = r["history"], r["history"]["epoch"].index(r["best_epoch"])
            ax.plot(h["epoch"], h[f"train_{name}"], color=color, linestyle="--", alpha=0.8)
            ax.plot(h["epoch"], h[f"val_{name}"], color=color, label=label)
            ax.plot(h["epoch"][j], h[f"val_{name}"][j], "o", color=color, markersize=7)
        ax.set(xlabel="epoch", title=f"{'loss' if name == 'loss' else 'accuracy'}: validation (solid) vs train (dashed)")
    axes[0].legend(title="config (dot = best epoch)")
    if show_lr:
        for label, r, color in ok:
            axes[2].plot(r["history"]["epoch"], r["history"]["lr"], color=color, label=label)
        axes[2].set(yscale="log", xlabel="epoch", title="learning rate")
    save_figure(fig, out + ".png", args.title or " vs ".join(label for label, _ in runs))
    print(f"written {out}.png and {out}.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
