#!/usr/bin/env python3
"""Hyperparameter search: grid, random search or genetic algorithm, trainings in parallel.

usage: python3 scripts/hpo.py SEARCH.json --data DATA.npz [options]

SEARCH.json: {"base": "configs/basic.conf", "method": "random", "space": {...}, ...}
(see configs/search/ and docs/nn_module.md). Command line options override it.

Writes in --out (default results/hpo_<search name>):
  trials.csv    one row per trained config: genes, metrics, epochs, time
  results.json  same, plus the full config and history of every trial
  best.conf     best config found, ready for train / my_torch_generator
  summary.md    the 10 best trials
  hpo.png       with --plot
"""

import argparse
import json
import math
import os
import sys
import time

from common import ROOT, describe_layers, fail, load_dataset, load_json, write_csv

from my_torch import hpo
from my_torch.config import ConfigError, normalize_config
from my_torch.parallel import train_many

import numpy as np

OBJ_SIGN = hpo.OBJECTIVES

DEFAULTS = {"method": "random", "objective": "val_loss", "trials": 20, "population": 10,
            "generations": 5, "elite": 2, "mutation_rate": 0.25, "tournament": 3,
            "epochs": None, "train_samples": None, "val_samples": None, "seed": 0}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("search", help="search file (JSON)")
    p.add_argument("--data", required=True, help="encoded dataset (.npz)")
    p.add_argument("--method", choices=("grid", "random", "genetic"))
    p.add_argument("--objective", choices=tuple(hpo.OBJECTIVES))
    p.add_argument("--trials", type=int, help="random search: number of configs")
    p.add_argument("--population", type=int, help="genetic: individuals per generation")
    p.add_argument("--generations", type=int, help="genetic: number of generations")
    p.add_argument("--elite", type=int, help="genetic: best individuals kept as they are")
    p.add_argument("--mutation-rate", type=float, dest="mutation_rate")
    p.add_argument("--epochs", type=int, help="max epochs of each training (default: base config)")
    p.add_argument("--train-samples", type=int, dest="train_samples", help="random subset of the training set")
    p.add_argument("--val-samples", type=int, dest="val_samples", help="random subset of the validation set")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel trainings (default: one per core)")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", help="output directory")
    p.add_argument("--plot", action="store_true", help="also draw hpo.png")
    return p.parse_args(argv)


def resolve(path, relative_to):
    if os.path.exists(path) or os.path.isabs(path):
        return path
    for base in (os.path.dirname(relative_to), ROOT):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return path


def short(genes):
    """Compact text of the genes for the progress lines."""
    parts = []
    for key, value in genes.items():
        name = key.split(".")[-1]
        parts.append(f"{name}={value:.3g}" if isinstance(value, float) else f"{name}={value}")
    return " ".join(parts)


def brief(config):
    """Architecture and main hyperparameters of a config, on one line."""
    cfg = normalize_config(config)
    hidden = cfg["layers"][:-1]
    drop = hidden[0]["dropout"] if hidden else 0.0
    return (f"{describe_layers(cfg)}, {cfg['optimizer']['name']} lr {cfg['optimizer']['lr']:.1e},"
            f" batch {cfg['training']['batch_size']}, dropout {drop:.2f}, l2 {cfg['training']['l2']:.0e}")


def row(i, result, objective):
    out = {"trial": i + 1, "method": result.get("method"), "generation": result.get("generation", "")}
    out.update({f"gene:{k}": v for k, v in result["genes"].items()})
    out["layers"] = describe_layers(result["config"]) if "layers" in result["config"] else ""
    for key in ("val_loss", "val_acc", "val_macro_f1", "train_loss", "train_acc"):
        value = result.get(key)
        out[key] = round(value, 6) if isinstance(value, float) else value
    if result.get("train_acc") is not None and result.get("val_acc") is not None:
        out["acc_gap"] = round(result["train_acc"] - result["val_acc"], 6)
    for key in ("epochs", "best_epoch", "n_params"):
        out[key] = result.get(key)
    out["time_s"] = round(result.get("time", 0.0), 2)
    out["error"] = result.get("error", "")
    out["score"] = hpo.score(result, objective)
    return out


def plot(results, settings, path):
    import plotstyle

    objective = settings["objective"]
    ok = [r for r in results if math.isfinite(hpo.score(r, objective))]
    fig, (ax,) = plotstyle.figure(width=7.5, height=4.6)
    if settings["method"] == "genetic":
        gens = sorted({r["generation"] for r in ok})
        best_so_far, current = [], math.inf
        for g in gens:
            current = min(current, min(hpo.score(r, objective) for r in ok if r["generation"] == g))
            best_so_far.append(OBJ_SIGN[objective] * current)
        ax.scatter([r["generation"] + 1 for r in ok], [r[objective] for r in ok],
                   color=plotstyle.NEUTRAL, s=24, label="individual")
        ax.plot([g + 1 for g in gens], best_so_far, color=plotstyle.color(0), marker="o",
                markersize=7, label="best so far")
        ax.set_xlabel("generation")
        ax.set_xticks([g + 1 for g in gens])
    else:
        depths = sorted({len(r["config"]["layers"]) - 1 for r in ok})
        for i, depth in enumerate(depths):
            pts = [r for r in ok if len(r["config"]["layers"]) - 1 == depth]
            ax.scatter([r["n_params"] for r in pts], [r[objective] for r in pts], s=40,
                       color=plotstyle.color(i), edgecolor=plotstyle.SURFACE, linewidth=1.5,
                       label=f"{depth} hidden layer{'s' if depth > 1 else ''}")
        plotstyle.log_count_axis(ax)
        ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel(objective)
    ax.legend()
    plotstyle.save(fig, path, f"{settings['name']}: {settings['method']} search, {len(results)} trainings")


def main(argv):
    args = parse_args(argv)
    search = load_json(args.search, "search file")
    if not isinstance(search, dict) or "space" not in search or "base" not in search:
        fail(f"{args.search}: needs at least \"base\" and \"space\"")
    settings = {**DEFAULTS, **{k: v for k, v in search.items() if k in DEFAULTS}}
    settings.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    settings["name"] = search.get("name", os.path.splitext(os.path.basename(args.search))[0])

    base = search["base"]
    if isinstance(base, str):
        base = load_json(resolve(base, args.search))
    try:
        base = normalize_config(base)
    except ConfigError as e:
        fail(f"base config: {e}")
    try:
        hpo.check_space(search["space"])
    except hpo.SpaceError as e:
        fail(f"{args.search}: {e}")

    data = load_dataset(args.data, settings["train_samples"], settings["val_samples"], settings["seed"])
    out_dir = args.out or os.path.join("results", f"hpo_{settings['name']}")
    os.makedirs(out_dir, exist_ok=True)
    fit_kwargs = {"epochs": settings["epochs"]} if settings["epochs"] else {}
    objective = settings["objective"]
    print(f"{settings['name']}: {settings['method']} search on {len(data['X_train'])} training /"
          f" {len(data['X_val'])} validation samples, {args.workers} workers, objective {objective}",
          file=sys.stderr)

    done = [0]

    def evaluate(configs):
        total = len(configs)
        start = done[0]

        def report(result):
            done[0] += 1
            if result.get("error"):
                status = f"error: {result['error']}"
            else:
                status = (f"val_loss {result['val_loss']:.4f}  val_acc {100 * result['val_acc']:.2f}%"
                          f"  f1 {result['val_macro_f1']:.4f}  {result['epochs']} ep")
            print(f"  [{done[0] - start:>3}/{total}] {status}  ({brief(result['config'])})"
                  f"  {result['time']:.1f}s", file=sys.stderr)

        return train_many(configs, data["X_train"], data["Y_train"], data["X_val"], data["Y_val"],
                          workers=args.workers, fit_kwargs=fit_kwargs, on_result=report)

    rng = np.random.default_rng(settings["seed"])
    t0 = time.perf_counter()
    try:
        if settings["method"] == "grid":
            results = hpo.grid_search(search["space"], base, evaluate)
        elif settings["method"] == "random":
            results = hpo.random_search(search["space"], base, evaluate, settings["trials"], rng)
        else:
            def on_generation(g, ranked):
                top = ranked[0]
                print(f"generation {g + 1}/{settings['generations']}: best {objective}"
                      f" {top.get(objective, float('nan')):.4f} ({short(top['genes'])})", file=sys.stderr)
            results = hpo.genetic_search(search["space"], base, evaluate, rng,
                                         population=settings["population"], generations=settings["generations"],
                                         elite=settings["elite"], mutation_rate=settings["mutation_rate"],
                                         tournament=settings["tournament"], objective=objective,
                                         on_generation=on_generation)
    except (hpo.SpaceError, ConfigError) as e:
        fail(str(e))
    elapsed = time.perf_counter() - t0

    rows = [row(i, r, objective) for i, r in enumerate(results)]
    write_csv(os.path.join(out_dir, "trials.csv"), rows)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "elapsed_s": elapsed, "results": results}, f, indent=1)
    winner = hpo.best(results, objective)
    if not math.isfinite(hpo.score(winner, objective)):
        fail("every training failed, see results.json")
    best_config = dict(winner["config"])
    best_config["name"] = f"{settings['name']}_best"
    best_config["description"] = (f"best of {len(results)} trainings ({settings['method']} search,"
                                  f" {objective} {winner[objective]:.4f})")
    with open(os.path.join(out_dir, "best.conf"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
        f.write("\n")

    ranked = sorted(rows, key=lambda r: r["score"])[:10]
    lines = [f"# {settings['name']} — {settings['method']} search", "",
             f"{len(results)} trainings in {elapsed / 60:.1f} min, {args.workers} workers,"
             f" {len(data['X_train'])} training / {len(data['X_val'])} validation samples,"
             f" objective `{objective}`.", "",
             "| rank | layers | genes | val_loss | val_acc | macro F1 | acc gap | epochs | params | time |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        genes = ", ".join(f"{k[5:]}={v:.3g}" if isinstance(v, float) else f"{k[5:]}={v}"
                          for k, v in r.items() if k.startswith("gene:"))
        lines.append(f"| {i} | {r['layers']} | {genes} | {r['val_loss']} | {r['val_acc']} | {r['val_macro_f1']}"
                     f" | {r.get('acc_gap', '')} | {r['epochs']} | {r['n_params']} | {r['time_s']}s |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if args.plot:
        plot(results, settings, os.path.join(out_dir, "hpo.png"))

    print(f"best: {objective} {winner[objective]:.4f} with {short(winner['genes'])}"
          f" ({describe_layers(winner['config'])})", file=sys.stderr)
    print(f"{len(results)} trainings in {elapsed:.0f}s, written to {out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
