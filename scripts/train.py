#!/usr/bin/env python3
"""Train one network on the encoded dataset, save it and write its training report (T1.12).

usage: python3 scripts/train.py CONFIG --data data/chess.npz --out NET.nn [--report DIR] [--epochs N] [--flip-colors] [--report-only]

The report (default results/<name>/): curves.png (loss, accuracy, learning rate per
epoch, train dashed / validation solid, redrawn after every epoch: it can be
watched live), confusion.png and report.md (per-class precision / recall / F1 and
per-subset results on the test set). An existing NET.nn is kept as NET.nn.bak.
--flip-colors doubles the training boards with their colour-inverted mirror (analyzer/augment.py);
validation and test are never augmented.
--report-only trains nothing: it rewrites the report of NET.nn (its history is in the file).
"""

import argparse
import os
import sys

os.environ.setdefault("MY_TORCH_THREADS", "2")  # one training: 2 BLAS threads (docs/benchmarks.md)

from common import COLORS, describe_layers, fail, figure, load_dataset, load_json, save_figure  # noqa: E402

from analyzer.augment import flip_colors  # noqa: E402
from my_torch import ConfigError, Network, NetworkFormatError, TrainingError, load_network  # noqa: E402
from my_torch.metrics import confusion_matrix, per_class, report  # noqa: E402

import numpy as np  # noqa: E402


def draw_curves(h, path, title):
    fig, axes = figure(3, width=4.6, height=3.8)
    for ax, name in zip(axes, ("loss", "acc")):
        ax.plot(h["epoch"], h[f"train_{name}"], color=COLORS[0], linestyle="--", label="train")
        ax.plot(h["epoch"], h[f"val_{name}"], color=COLORS[0], label="validation")
        ax.set(xlabel="epoch", title="loss" if name == "loss" else "accuracy")
    axes[0].legend()
    axes[2].plot(h["epoch"], h["lr"], color=COLORS[1])
    axes[2].set(yscale="log", xlabel="epoch", title="learning rate")
    save_figure(fig, path, title)


def draw_confusion(cm, labels, path, title):
    fig, (ax,) = figure(width=6.4, height=5.2)
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # diagonal = recall of each class
    ax.grid(False)
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{100 * rates[i, j]:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if rates[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="predicted", ylabel="true")
    save_figure(fig, path, title)


def subset_rows(data, pred, labels):
    """Accuracy, macro F1 and recall of each class, per subset of the test set (absent class: -)."""
    lines = ["| subset | boards | accuracy | macro F1 | " + " | ".join(f"recall {label}" for label in labels) + " |",
             "|" + "---|" * (len(labels) + 4)]
    for s, name in enumerate(data.get("subsets", [])):
        mask = data["S_test"] == s
        if mask.any():
            stats = per_class(confusion_matrix(data["Y_test"][mask], pred[mask], len(labels)))
            present = stats["support"] > 0
            recalls = [f"{r:.4f}" if ok else "-" for r, ok in zip(stats["recall"], present)]
            lines.append(f"| {name} | {mask.sum()} | {(data['Y_test'][mask] == pred[mask]).mean():.4f} | "
                         f"{stats['f1'][present].mean():.4f} | " + " | ".join(recalls) + " |")
    return lines if len(lines) > 2 else []


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("config")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report")
    p.add_argument("--epochs", type=int)
    p.add_argument("--flip-colors", action="store_true", dest="flip_colors")
    p.add_argument("--report-only", action="store_true", dest="report_only")
    args = p.parse_args(argv)
    config = load_json(args.config)
    name = config.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.report or os.path.join("results", name)
    os.makedirs(out_dir, exist_ok=True)
    data = load_dataset(args.data, test=True)

    if args.report_only:
        try:
            net = load_network(args.out)
        except NetworkFormatError as e:
            fail(str(e))
    else:
        config.setdefault("labels", [str(x) for x in data["labels"]] if "labels" in data else None)
        try:
            net = Network(config)
        except ConfigError as e:
            fail(f"{args.config}: {e}")
        if args.flip_colors:
            # training boards only: validation and test stay as they are
            X_flip, Y_flip = flip_colors(data["X_train"], data["Y_train"], net.labels)
            data["X_train"], data["Y_train"] = np.concatenate([data["X_train"], X_flip]), np.concatenate([data["Y_train"], Y_flip])
            print(f"colour flip augmentation: {len(X_flip)} mirrored boards added to the training set", file=sys.stderr)
        print(f"{name}: {describe_layers(net.config)}, {net.n_params} parameters,"
              f" {len(data['X_train'])} training / {len(data['X_val'])} validation boards", file=sys.stderr)
        live = lambda epoch, h: draw_curves(h, os.path.join(out_dir, "curves.png"), f"{name}: epoch {epoch}")  # noqa: E731
        try:
            net.fit(data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], verbose=True, callback=live,
                    **({"epochs": args.epochs} if args.epochs else {}))
        except TrainingError as e:
            fail(str(e))
        net.save(args.out, backup=os.path.exists(args.out))
        print(f"network saved in {args.out}", file=sys.stderr)

    labels, h, meta = net.labels, net.history, net.meta
    draw_curves(h, os.path.join(out_dir, "curves.png"), f"{name}: learning curves")
    pred = net.predict(data["X_test"])
    cm = confusion_matrix(data["Y_test"], pred, len(labels))
    draw_confusion(cm, labels, os.path.join(out_dir, "confusion.png"), f"{name}: test set")
    lines = [f"# Training report: {name}", "",
             f"- `{args.out}`: {describe_layers(net.config)}, {net.n_params} parameters, config `{args.config}`",
             f"- {len(h['epoch'])} epochs in {meta['train_time']:.0f}s on {meta['train_samples']} boards,"
             f" best epoch {meta['best_epoch']} (val_loss {meta['best_val_loss']:.4f})"
             + (", early stopping" if meta["stopped_early"] else ""), "",
             "![learning curves](curves.png)", "",
             f"## Test set ({len(pred)} boards never used for training or model selection)", "",
             "```", report(cm, labels), "```", "", "![confusion matrix](confusion.png)", ""]
    subsets = subset_rows(data, pred, labels)
    lines += ["## Per subset", "", *subsets, ""] if subsets else []
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n{report(cm, labels)}\nreport written in {out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
