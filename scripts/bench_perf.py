#!/usr/bin/env python3
"""Training speed benchmark (T2.3): what each optimisation is worth, on the same network and data.

usage: python3 scripts/bench_perf.py --data DATA.npz [--config configs/basic.conf] [--out results/perf]

  threads        seconds per epoch vs number of BLAS threads (batch 256)
  dtype          float64 vs float32
  vectorization  samples/s, one sample at a time vs mini-batches of 256
  subnormal      L2 training with and without flushing subnormal floats to 0
  speedup        2 trainings per core run by 1, 2, 4... worker processes
Writes perf.csv and perf_speedup.png in --out.
"""

import argparse
import copy
import os
import sys
import time

from common import COLORS, ROOT, fail, figure, load_dataset, load_json, save_figure, write_csv

import my_torch.training.batch as batch_module
from my_torch.config import normalize_config
from my_torch.network import Network
from my_torch.parallel import train_many

import numpy as np


def plain(config, **training):
    """The config without dropout, early stopping, schedule nor evaluation: only the training loop is timed."""
    cfg = normalize_config(copy.deepcopy(config))
    for layer in cfg["layers"]:
        layer["dropout"] = 0.0
    cfg["training"].update(early_stopping=None, eval_train_samples=0, l2=0.0, grad_clip=None, **training)
    cfg["lr_schedule"] = {"name": "constant"}
    return cfg


def epoch_time(config, X, Y, threads=1, **training):
    """Median seconds per epoch of one training in a fresh process (its own number of BLAS threads)."""
    r = train_many([plain(config, **training)], X, Y, workers=1, threads_per_worker=threads)[0]
    if r.get("error"):
        fail(r["error"])
    return float(np.median(r["history"]["time"]))


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", required=True)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "basic.conf"))
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--out", default="results/perf")
    args = p.parse_args(argv)
    config, data = load_json(args.config), load_dataset(args.data, args.samples)
    X, Y, cores, rows = data["X_train"], data["Y_train"], os.cpu_count() or 1, []
    os.makedirs(args.out, exist_ok=True)

    def add(bench, setting, value, unit):
        rows.append({"bench": bench, "setting": setting, "value": round(value, 3), "unit": unit})
        print(f"  {bench:<13} {setting:<14} {value:10.3f} {unit}", file=sys.stderr)

    for threads in sorted({1, 2, 4, cores} & set(range(1, cores + 1))):
        add("threads", f"{threads} threads", epoch_time(config, X, Y, threads, epochs=2), "s/epoch")
    for dtype in ("float64", "float32"):
        add("dtype", dtype, epoch_time({**config, "dtype": dtype}, X, Y, epochs=2), "s/epoch")
    for batch, n in ((1, 2000), (256, len(X))):
        add("vectorization", f"batch {batch}", n / epoch_time(config, X[:n], Y[:n], epochs=1, batch_size=batch), "samples/s")
    # in this process: the flush period is a module constant
    default = batch_module.FLUSH_EVERY
    for label, every in (("without flush", 10**12), ("with flush", default)):
        batch_module.FLUSH_EVERY = every
        history = Network(plain(config, epochs=30, l2=0.001)).fit(X[:20000], Y[:20000])
        add("subnormal", label, float(np.mean(history["time"][-5:])), "s/epoch (last 5 of 30)")
    batch_module.FLUSH_EVERY = default
    configs = [dict(plain(config, epochs=3), seed=i) for i in range(2 * cores)]
    base, workers_list, speedups = None, sorted({1, 2, 4, 8, cores} & set(range(1, cores + 1))), []
    for workers in workers_list:
        t0 = time.perf_counter()
        train_many(configs, X, Y, workers=workers)
        wall = time.perf_counter() - t0
        base = base or wall
        speedups.append(base / wall)
        add("speedup", f"{workers} workers", base / wall, f"x faster than 1 worker ({len(configs)} trainings, {wall:.1f}s)")
    write_csv(os.path.join(args.out, "perf.csv"), rows)

    fig, (ax,) = figure(width=6.0, height=3.8)
    ax.plot(workers_list, workers_list, color="#9a9890", linestyle="--", label="ideal (linear)")
    ax.plot(workers_list, speedups, color=COLORS[0], marker="o", label="measured")
    for w, v in zip(workers_list, speedups):
        ax.annotate(f"{v:.1f}x", (w, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set(xlabel="worker processes", ylabel="speedup", xticks=workers_list)
    ax.legend()
    save_figure(fig, os.path.join(args.out, "perf_speedup.png"), "parallel trainings: speedup vs workers")
    print(f"written {args.out}/perf.csv and perf_speedup.png", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
