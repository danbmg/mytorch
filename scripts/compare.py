#!/usr/bin/env python3
"""Train several configs on the same data and draw their learning curves on one graph.

usage:
  python3 scripts/compare.py --data DATA.npz CONFIG [CONFIG ...] [options]
  python3 scripts/compare.py --data DATA.npz --base CONFIG --variant "LABEL: key=value ..." [--variant ...]

A variant is the base config with some settings changed (dotted paths, see
docs/nn_module.md). Example, effect of the regularisation:

  python3 scripts/compare.py --data data/chess.npz --base configs/basic.conf --train-samples 30000 \\
    --variant "none: layers.hidden.dropout=0 training.l2=0 training.early_stopping=null" \\
    --variant "l2: layers.hidden.dropout=0 training.early_stopping=null" \\
    --variant "dropout: training.l2=0 training.early_stopping=null" \\
    --variant "all: " --out docs/img/regularization.png

Writes OUT.png (loss, accuracy and learning rate per epoch; train dashed,
validation solid), OUT.md and OUT.csv (train/validation gap at the best epoch).
"""

import argparse
import copy
import os
import sys

from common import apply_assignments, describe_layers, fail, load_dataset, load_json, write_csv

from my_torch.config import ConfigError, normalize_config
from my_torch.parallel import train_many


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="*", help="config files to compare")
    p.add_argument("--data", required=True, help="encoded dataset (.npz)")
    p.add_argument("--base", help="base config for --variant")
    p.add_argument("--variant", action="append", default=[], help='"label: key=value key=value"')
    p.add_argument("--epochs", type=int, help="max epochs (default: each config's)")
    p.add_argument("--train-samples", type=int, dest="train_samples")
    p.add_argument("--val-samples", type=int, dest="val_samples")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0, help="seed of the data subsets")
    p.add_argument("--title")
    p.add_argument("--out", default="results/compare.png", help="output image (.png)")
    return p.parse_args(argv)


def build_runs(args):
    runs = []
    for path in args.configs:
        runs.append((os.path.splitext(os.path.basename(path))[0], load_json(path)))
    if args.variant:
        if not args.base:
            fail("--variant needs --base")
        base = load_json(args.base)
        for text in args.variant:
            label, _, changes = text.partition(":")
            runs.append((label.strip(), apply_assignments(copy.deepcopy(base), changes.split())))
    if not 2 <= len(runs) <= 8:
        fail(f"{len(runs)} configs: compare between 2 and 8")
    for label, config in runs:
        try:
            normalize_config(config)
        except ConfigError as e:
            fail(f"{label}: {e}")
    return runs


def best_index(result):
    return result["history"]["epoch"].index(result["best_epoch"])


def draw(runs, results, path, title):
    import plotstyle

    show_lr = len({tuple(r["history"]["lr"]) for r in results if "history" in r}) > 1
    fig, axes = plotstyle.figure(3 if show_lr else 2, width=5.2, height=4.2)
    panels = [("loss", "train_loss", "val_loss"), ("accuracy", "train_acc", "val_acc")]
    for ax, (name, train_key, val_key) in zip(axes, panels):
        for i, ((label, _), result) in enumerate(zip(runs, results)):
            if "history" not in result:
                continue
            h = result["history"]
            c = plotstyle.color(i)
            ax.plot(h["epoch"], h[train_key], color=c, linestyle="--", alpha=0.8)
            ax.plot(h["epoch"], h[val_key], color=c, label=label)
            j = best_index(result)
            ax.plot(h["epoch"][j], h[val_key][j], "o", color=c, markersize=7,
                    markeredgecolor=plotstyle.SURFACE, markeredgewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.set_title(f"{name}: validation (solid) vs train (dashed)", fontsize=10)
    axes[0].legend(title="config (dot = best epoch)")
    if show_lr:
        ax = axes[2]
        for i, ((label, _), result) in enumerate(zip(runs, results)):
            if "history" in result:
                ax.plot(result["history"]["epoch"], result["history"]["lr"], color=plotstyle.color(i), label=label)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_title("learning rate", fontsize=10)
    plotstyle.save(fig, path, title)


def main(argv):
    args = parse_args(argv)
    runs = build_runs(args)
    data = load_dataset(args.data, args.train_samples, args.val_samples, args.seed)
    fit_kwargs = {"epochs": args.epochs} if args.epochs else {}
    print(f"{len(runs)} configs on {len(data['X_train'])} training / {len(data['X_val'])} validation samples",
          file=sys.stderr)

    def report(result):
        label = runs[result["index"]][0]
        if result.get("error"):
            print(f"  {label}: error: {result['error']}", file=sys.stderr)
        else:
            print(f"  {label}: val_loss {result['val_loss']:.4f} val_acc {100 * result['val_acc']:.2f}%"
                  f" ({result['epochs']} epochs, {result['time']:.0f}s)", file=sys.stderr)

    results = train_many([config for _, config in runs], data["X_train"], data["Y_train"],
                         data["X_val"], data["Y_val"], workers=args.workers, fit_kwargs=fit_kwargs,
                         on_result=report)

    rows = []
    for (label, _), result in zip(runs, results):
        if result.get("error"):
            rows.append({"config": label, "error": result["error"]})
            continue
        h, j = result["history"], best_index(result)
        rows.append({
            "config": label,
            "layers": describe_layers(result["config"]),
            "best_epoch": result["best_epoch"],
            "epochs": result["epochs"],
            "train_loss": round(h["train_loss"][j], 4),
            "val_loss": round(h["val_loss"][j], 4),
            "loss_gap": round(h["val_loss"][j] - h["train_loss"][j], 4),
            "train_acc": round(h["train_acc"][j], 4),
            "val_acc": round(h["val_acc"][j], 4),
            "acc_gap": round(h["train_acc"][j] - h["val_acc"][j], 4),
            "last_acc_gap": round(h["train_acc"][-1] - h["val_acc"][-1], 4),
            "val_macro_f1": round(result["val_macro_f1"], 4),
            "time_s": round(result["time"], 1),
        })

    out = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_csv(out + ".csv", rows)
    keys = ["config", "best_epoch", "epochs", "train_loss", "val_loss", "loss_gap",
            "train_acc", "val_acc", "acc_gap", "last_acc_gap", "val_macro_f1", "time_s"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, r.get("error", ""))) for k in keys) + " |")
    lines += ["", f"{len(data['X_train'])} training / {len(data['X_val'])} validation samples."
              " Gaps measured at the best epoch (lowest val_loss); last_acc_gap at the last epoch."]
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    title = args.title or " vs ".join(label for label, _ in runs)
    draw(runs, results, out + ".png", title)
    print(f"written {out}.png, {out}.md, {out}.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
