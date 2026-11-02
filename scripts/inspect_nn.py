#!/usr/bin/env python3
"""Show what a .nn network file contains, without training or predicting anything.

usage: python3 scripts/inspect_nn.py FILE.nn [--json] [--history]
  --json      print the raw JSON header
  --history   print the training history, epoch by epoch
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from my_torch import NetworkFormatError, read_header  # noqa: E402

import numpy as np  # noqa: E402


def describe(path, header):
    cfg = header["config"]
    meta = header.get("meta") or {}
    lines = [
        f"file          {path} ({os.path.getsize(path) / 1024:.1f} KiB)",
        f"format        {header['format']} v{header['format_version']}, created {header.get('created', '?')}",
    ]
    if cfg.get("name") or cfg.get("description"):
        lines.append(f"name          {cfg.get('name', '')}  {cfg.get('description', '')}".rstrip())
    lines.append(f"input size    {cfg['input_size']}")
    lines.append("layers")
    for i, layer in enumerate(cfg["layers"]):
        extra = "".join(f", {k} {v}" for k, v in layer.items() if k not in ("size", "activation", "init", "dropout"))
        drop = f", dropout {layer['dropout']}" if layer["dropout"] else ""
        lines.append(f"  {i}  {layer['size']:>5} neurons  {layer['activation']:<10} init {layer['init']}{drop}{extra}")
    opt = cfg["optimizer"]
    lines.append(f"loss          {cfg['loss']}   dtype {cfg['dtype']}   seed {cfg['seed']}")
    lines.append("optimizer     " + ", ".join(f"{k} {v}" for k, v in opt.items()))
    lines.append("lr schedule   " + ", ".join(f"{k} {v}" for k, v in cfg["lr_schedule"].items()))
    lines.append("training      " + ", ".join(f"{k} {v}" for k, v in cfg["training"].items()))
    if header.get("labels"):
        lines.append("labels        " + ", ".join(header["labels"]))
    if meta.get("epochs_trained"):
        best = ""
        if meta.get("best_val_loss") is not None:
            best = f", best val_loss {meta['best_val_loss']:.4f} at epoch {meta['best_epoch']}"
        lines.append(f"trained       {meta['epochs_trained']} epochs on {meta.get('train_samples')} samples"
                     f" in {meta.get('train_time', 0):.1f}s{best}"
                     + (" (early stopping)" if meta.get("stopped_early") else ""))
    else:
        lines.append("trained       no (blank network)")
    return "\n".join(lines)


def weights_table(path):
    lines = ["arrays (binary)"]
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            if name == "header":
                continue
            a = data[name]
            lines.append(f"  {name:<4} {str(a.shape):<12} {a.dtype}  mean {a.mean():+.4f}  std {a.std():.4f}"
                         f"  min {a.min():+.4f}  max {a.max():+.4f}")
    return "\n".join(lines)


def history_table(header):
    h = header.get("history") or {}
    if not h.get("epoch"):
        return "no training history"
    lines = ["epoch  train_loss  train_acc  val_loss  val_acc        lr"]
    for i, epoch in enumerate(h["epoch"]):
        vl, va = h["val_loss"][i], h["val_acc"][i]
        val = f"{vl:9.4f}  {va:7.4f}" if vl is not None else f"{'-':>9}  {'-':>7}"
        lines.append(f"{epoch:5d}  {h['train_loss'][i]:10.4f}  {h['train_acc'][i]:9.4f}  {val}  {h['lr'][i]:.2e}")
    return "\n".join(lines)


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1 or flags - {"--json", "--history"}:
        print(__doc__.strip(), file=sys.stderr)
        return 84
    path = paths[0]
    try:
        header = read_header(path)
    except NetworkFormatError as e:
        print(f"inspect_nn: {e}", file=sys.stderr)
        return 84
    if "--json" in flags:
        print(json.dumps(header, indent=2))
    elif "--history" in flags:
        print(history_table(header))
    else:
        print(describe(path, header))
        print(weights_table(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
