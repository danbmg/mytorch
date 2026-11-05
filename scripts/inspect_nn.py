#!/usr/bin/env python3
"""Show what a .nn network file contains (layers, activations, hyperparameters, training, weights).

usage: python3 scripts/inspect_nn.py FILE.nn [--json]     (--json: the raw header, history included)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from my_torch import NetworkFormatError, read_header  # noqa: E402

import numpy as np  # noqa: E402


def main(argv):
    if "-h" in argv or "--help" in argv:
        print(__doc__.strip())
        return 0
    paths = [a for a in argv if a != "--json"]
    if len(paths) != 1 or [a for a in argv if a.startswith("-") and a != "--json"]:
        print(__doc__.strip(), file=sys.stderr)
        return 84
    path = paths[0]
    try:
        header = read_header(path)
    except NetworkFormatError as e:
        print(f"inspect_nn: {e}", file=sys.stderr)
        return 84
    if "--json" in argv:
        print(json.dumps(header, indent=2))
        return 0
    cfg = header["config"]
    meta = header.get("meta") or {}
    print(f"file          {path} ({os.path.getsize(path) / 1024:.1f} KiB), format v{header['format_version']},"
          f" created {header.get('created', '?')}")
    print(f"name          {cfg.get('name', '')}  {cfg.get('description', '')}")
    print(f"input size    {cfg['input_size']}\nlayers")
    for i, layer in enumerate(cfg["layers"]):
        extra = "".join(f", {k} {v}" for k, v in layer.items() if k not in ("size", "activation", "init") and v)
        print(f"  {i}  {layer['size']:>5} neurons  {layer['activation']:<10} init {layer['init']}{extra}")
    print(f"loss          {cfg['loss']}, dtype {cfg['dtype']}, seed {cfg['seed']}")
    for key in ("optimizer", "lr_schedule", "training"):
        print(f"{key:<14}" + ", ".join(f"{k} {v}" for k, v in cfg[key].items()))
    print(f"labels        {', '.join(header.get('labels') or [])}")
    if meta.get("epochs_trained"):
        print(f"trained       {meta['epochs_trained']} epochs on {meta.get('train_samples')} samples in"
              f" {meta.get('train_time', 0):.0f}s, best val_loss {meta.get('best_val_loss') or float('nan'):.4f}"
              f" at epoch {meta.get('best_epoch')}" + (" (early stopping)" if meta.get("stopped_early") else ""))
    else:
        print("trained       no (blank network)")
    print("arrays (binary)")
    with np.load(path, allow_pickle=False) as data:
        for name in (n for n in data.files if n != "header"):
            a = data[name]
            print(f"  {name:<4} {str(a.shape):<12} {a.dtype}  mean {a.mean():+.4f}  std {a.std():.4f}"
                  f"  min {a.min():+.4f}  max {a.max():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
