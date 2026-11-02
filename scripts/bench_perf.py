#!/usr/bin/env python3
"""Training speed benchmark of my_torch (T2.3): what each optimisation is worth.

usage: python3 scripts/bench_perf.py --data DATA.npz [--config configs/basic.conf] [--out results/perf]

Measures, on the same network and data:
  threads        seconds per epoch vs number of BLAS threads (batch 256)
  dtype          float64 vs float32
  vectorization  samples/s, one sample at a time (batch 1) vs mini-batches of 256
  subnormal      L2 training with and without flushing subnormal floats to 0
  speedup        N independent trainings with 1, 2, 4... worker processes

Writes perf.csv, perf.md and one perf_<name>.png per measurement in --out.
"""

import argparse
import copy
import os
import sys
import time

from common import fail, load_dataset, load_json, write_csv

import my_torch.network as network_module
from my_torch.config import normalize_config
from my_torch.network import Network
from my_torch.parallel import train_many

import numpy as np


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", required=True, help="encoded dataset (.npz)")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "basic.conf"))
    p.add_argument("--samples", type=int, default=100000, help="training samples per timed epoch")
    p.add_argument("--out", default="results/perf")
    p.add_argument("--only", nargs="*", choices=("threads", "dtype", "vectorization", "subnormal", "speedup"))
    return p.parse_args(argv)


def plain(config, **training):
    """Config without dropout / early stopping / evaluation, so that only the training loop is timed."""
    cfg = normalize_config(copy.deepcopy(config))
    for layer in cfg["layers"]:
        layer["dropout"] = 0.0
    cfg["training"].update({"early_stopping": None, "eval_train_samples": 0, "l2": 0.0, "grad_clip": None})
    cfg["training"].update(training)
    cfg["lr_schedule"] = {"name": "constant"}
    return cfg


def epoch_time(result):
    """Median seconds per epoch of a train_many result."""
    if result.get("error"):
        fail(result["error"])
    return float(np.median(result["history"]["time"]))


def bench_threads(config, X, Y, rows):
    cores = os.cpu_count() or 1
    for threads in sorted({1, 2, 4, cores} & set(range(1, cores + 1))):
        cfg = plain(config, epochs=2, batch_size=256)
        t = epoch_time(train_many([cfg], X, Y, workers=1, threads_per_worker=threads)[0])
        rows.append({"bench": "threads", "setting": f"{threads} thread{'s' if threads > 1 else ''}",
                     "value": round(t, 3), "unit": "s/epoch"})
        print(f"  threads {threads}: {t:.2f}s/epoch", file=sys.stderr)


def bench_dtype(config, X, Y, rows):
    for dtype in ("float64", "float32"):
        cfg = plain(config, epochs=2, batch_size=256)
        cfg["dtype"] = dtype
        t = epoch_time(train_many([cfg], X, Y, workers=1)[0])
        rows.append({"bench": "dtype", "setting": dtype, "value": round(t, 3), "unit": "s/epoch"})
        print(f"  {dtype}: {t:.2f}s/epoch", file=sys.stderr)


def bench_vectorization(config, X, Y, rows):
    for batch, n in ((1, 2000), (256, len(X))):
        cfg = plain(config, epochs=1, batch_size=batch)
        t = epoch_time(train_many([cfg], X[:n], Y[:n], workers=1)[0])
        rows.append({"bench": "vectorization", "setting": f"batch {batch}", "value": round(n / t),
                     "unit": "samples/s"})
        print(f"  batch {batch}: {n / t:.0f} samples/s", file=sys.stderr)


def bench_subnormal(config, X, Y, rows):
    # in this process: the flush period is a module constant
    X, Y = X[:20000], Y[:20000]
    default = network_module.FLUSH_EVERY
    for label, every in (("without flush", 10**12), ("with flush", default)):
        network_module.FLUSH_EVERY = every
        cfg = plain(config, epochs=30, batch_size=256, l2=0.001)
        net = Network(cfg)
        history = net.fit(X, Y)
        last = float(np.mean(history["time"][-5:]))
        rows.append({"bench": "subnormal", "setting": label, "value": round(last, 3),
                     "unit": "s/epoch (last 5 of 30)"})
        print(f"  L2 {label}: {last:.2f}s/epoch at the end of training", file=sys.stderr)
    network_module.FLUSH_EVERY = default


def bench_speedup(config, X, Y, rows):
    cores = os.cpu_count() or 1
    n_jobs = cores
    configs = []
    for i in range(n_jobs):
        cfg = plain(config, epochs=3, batch_size=256)
        cfg["seed"] = i
        configs.append(cfg)
    # trainings of a few seconds each, so that starting the processes (~0.5 s) stays negligible
    X, Y = X[:50000], Y[:50000]
    base = None
    for workers in sorted({1, 2, 4, 8, cores} & set(range(1, cores + 1))):
        t0 = time.perf_counter()
        train_many(configs, X, Y, workers=workers)
        wall = time.perf_counter() - t0
        base = base or wall
        rows.append({"bench": "speedup", "setting": f"{workers} workers", "value": round(base / wall, 2),
                     "unit": f"x faster than 1 worker ({n_jobs} trainings, {wall:.1f}s)"})
        print(f"  {workers} workers: {wall:.1f}s for {n_jobs} trainings, speedup {base / wall:.2f}x",
              file=sys.stderr)


def draw(rows, out_dir):
    import plotstyle

    for bench in sorted({r["bench"] for r in rows}):
        pts = [r for r in rows if r["bench"] == bench]
        fig, (ax,) = plotstyle.figure(width=6.0, height=3.8)
        labels = [r["setting"] for r in pts]
        values = [r["value"] for r in pts]
        if bench == "speedup":
            workers = [int(label.split()[0]) for label in labels]
            ax.plot(workers, workers, color=plotstyle.NEUTRAL, linestyle="--", label="ideal (linear)")
            ax.plot(workers, values, color=plotstyle.color(0), marker="o", markersize=7, label="measured")
            for w, v in zip(workers, values):
                ax.annotate(f"{v:.1f}x", (w, v), textcoords="offset points", xytext=(0, 8),
                            ha="center", color=plotstyle.TEXT_SECONDARY, fontsize=9)
            ax.set_xlabel("worker processes")
            ax.set_ylabel("speedup")
            ax.set_xticks(workers)
            ax.legend()
        else:
            bars = ax.bar(labels, values, color=plotstyle.color(0), width=0.6)
            for bar, v in zip(bars, values):
                ax.annotate(f"{v:g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            textcoords="offset points", xytext=(0, 4), ha="center",
                            color=plotstyle.TEXT_SECONDARY, fontsize=9)
            ax.set_ylabel(pts[0]["unit"])
            if bench == "vectorization":
                ax.set_yscale("log")
        titles = {"threads": "seconds per epoch vs BLAS threads (batch 256)",
                  "dtype": "seconds per epoch: float64 vs float32",
                  "vectorization": "throughput: one sample at a time vs mini-batch",
                  "subnormal": "L2 training: flushing subnormal floats",
                  "speedup": "parallel trainings: speedup vs workers"}
        plotstyle.save(fig, os.path.join(out_dir, f"perf_{bench}.png"), titles[bench])


def main(argv):
    args = parse_args(argv)
    config = load_json(args.config)
    data = load_dataset(args.data, args.samples)
    X, Y = data["X_train"], data["Y_train"]
    os.makedirs(args.out, exist_ok=True)
    benches = [("threads", bench_threads), ("dtype", bench_dtype), ("vectorization", bench_vectorization),
               ("subnormal", bench_subnormal), ("speedup", bench_speedup)]
    rows = []
    for name, bench in benches:
        if args.only and name not in args.only:
            continue
        print(f"{name}:", file=sys.stderr)
        bench(config, X, Y, rows)
    write_csv(os.path.join(args.out, "perf.csv"), rows)
    lines = [f"Network `{args.config}`, {len(X)} training samples, {os.cpu_count()} CPU cores.", "",
             "| measurement | setting | value | unit |", "|---|---|---|---|"]
    lines += [f"| {r['bench']} | {r['setting']} | {r['value']} | {r['unit']} |" for r in rows]
    with open(os.path.join(args.out, "perf.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    draw(rows, args.out)
    print(f"written to {args.out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
