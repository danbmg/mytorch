#!/usr/bin/env python3
"""Hyperparameter search (grid, random or genetic), trainings in parallel (T2.4).

usage: python3 scripts/hpo.py SEARCH.json --data DATA.npz [--out DIR] [--workers N]

SEARCH.json: {"base": "configs/basic.conf", "method": "random", "space": {...}, "trials": 24,
"epochs": 15, "train_samples": 150000, ...} (see configs/search/ and my_torch/hpo.py).
Writes in DIR (default results/hpo_<name>): trials.csv (genes and metrics of every
training), best.conf (ready for scripts/train.py or my_torch_generator) and hpo.png.
"""

import argparse
import json
import math
import os
import sys
import time

from common import COLORS, ROOT, describe_layers, fail, figure, load_dataset, load_json, log_count_axis, save_figure, write_csv

from my_torch import hpo
from my_torch.config import ConfigError, normalize_config
from my_torch.parallel import train_many

import numpy as np

SETTINGS = {"method": "random", "objective": "val_loss", "trials": 20, "population": 10, "generations": 5,
            "elite": 2, "mutation_rate": 0.25, "tournament": 3, "epochs": None, "train_samples": None,
            "val_samples": None, "seed": 0}


def plot(results, s, path):
    obj = s["objective"]
    ok = [r for r in results if math.isfinite(hpo.score(r, obj))]
    fig, (ax,) = figure(width=7.5, height=4.6)
    if s["method"] == "genetic":
        gens = sorted({r["generation"] for r in ok})
        best = np.minimum.accumulate([min(hpo.score(r, obj) for r in ok if r["generation"] == g) for g in gens])
        ax.scatter([r["generation"] + 1 for r in ok], [r[obj] for r in ok], color="#9a9890", s=24, label="individual")
        ax.plot([g + 1 for g in gens], hpo.OBJECTIVES[obj] * best, color=COLORS[0], marker="o", label="best so far")
        ax.set(xlabel="generation", xticks=[g + 1 for g in gens])
    else:
        for depth in sorted({len(r["config"]["layers"]) - 1 for r in ok}):
            pts = [r for r in ok if len(r["config"]["layers"]) - 1 == depth]
            ax.scatter([r["n_params"] for r in pts], [r[obj] for r in pts], s=40, color=COLORS[depth - 1],
                       label=f"{depth} hidden layer{'s' if depth > 1 else ''}")
        log_count_axis(ax)
        ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel(obj)
    ax.legend()
    save_figure(fig, path, f"{s['name']}: {s['method']} search, {len(results)} trainings")


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("search")
    p.add_argument("--data", required=True)
    p.add_argument("--out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    search = load_json(args.search)
    if not isinstance(search, dict) or "space" not in search or "base" not in search:
        fail(f"{args.search}: needs at least \"base\" and \"space\"")
    s = {**SETTINGS, **{k: v for k, v in search.items() if k in SETTINGS},
         "name": search.get("name", os.path.splitext(os.path.basename(args.search))[0])}
    try:
        base = normalize_config(load_json(os.path.join(ROOT, search["base"])) if isinstance(search["base"], str)
                                else search["base"])
        hpo.check_space(search["space"])
    except (ConfigError, hpo.SpaceError) as e:
        fail(f"{args.search}: {e}")
    data = load_dataset(args.data, s["train_samples"], s["val_samples"], s["seed"])
    out_dir = args.out or os.path.join("results", f"hpo_{s['name']}")
    os.makedirs(out_dir, exist_ok=True)
    obj = s["objective"]
    print(f"{s['name']}: {s['method']} search on {len(data['X_train'])} training / {len(data['X_val'])} validation"
          f" boards, {args.workers} workers, objective {obj}", file=sys.stderr)

    def evaluate(configs):
        def show(r):
            print("  " + (f"error: {r['error']}" if r.get("error") else
                          f"{obj} {r[obj]:.4f}  val_acc {100 * r['val_acc']:.2f}%  {describe_layers(r['config'])}, "
                          f"{r['config']['optimizer']['name']} lr {r['config']['optimizer']['lr']:.1e}, "
                          f"batch {r['config']['training']['batch_size']}  {r['time']:.0f}s"), file=sys.stderr)
        return train_many(configs, data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], workers=args.workers,
                          fit_kwargs={"epochs": s["epochs"]} if s["epochs"] else {}, on_result=show)

    rng, t0 = np.random.default_rng(s["seed"]), time.perf_counter()
    try:
        if s["method"] == "grid":
            results = hpo.grid_search(search["space"], base, evaluate)
        elif s["method"] == "random":
            results = hpo.random_search(search["space"], base, evaluate, s["trials"], rng)
        else:
            results = hpo.genetic_search(
                search["space"], base, evaluate, rng, population=s["population"], generations=s["generations"],
                elite=s["elite"], mutation_rate=s["mutation_rate"], tournament=s["tournament"], objective=obj,
                on_generation=lambda g, ranked: print(f"generation {g + 1}/{s['generations']}: best {obj}"
                                                      f" {ranked[0].get(obj, float('nan')):.4f}", file=sys.stderr))
    except (hpo.SpaceError, ConfigError) as e:
        fail(str(e))

    write_csv(os.path.join(out_dir, "trials.csv"), [
        {"trial": i + 1, "generation": r.get("generation", ""), **{f"gene:{k}": v for k, v in r["genes"].items()},
         "layers": describe_layers(r["config"]) if "layers" in r["config"] else "",
         **{k: r.get(k) for k in ("val_loss", "val_acc", "val_macro_f1", "train_acc", "epochs", "best_epoch", "n_params")},
         "time_s": round(r["time"], 1), "error": r.get("error", "")} for i, r in enumerate(results)])
    winner = min(results, key=lambda r: hpo.score(r, obj))
    if not math.isfinite(hpo.score(winner, obj)):
        fail("every training failed, see trials.csv")
    with open(os.path.join(out_dir, "best.conf"), "w", encoding="utf-8") as f:
        json.dump({**winner["config"], "name": f"{s['name']}_best",
                   "description": f"best of {len(results)} trainings ({s['method']} search, {obj} {winner[obj]:.4f})"},
                  f, indent=2)
    plot(results, s, os.path.join(out_dir, "hpo.png"))
    print(f"best: {obj} {winner[obj]:.4f} ({describe_layers(winner['config'])}); {len(results)} trainings in"
          f" {time.perf_counter() - t0:.0f}s, written to {out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
