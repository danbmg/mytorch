"""Hyperparameter optimisation: grid search, random search and a genetic algorithm.

A search space maps each hyperparameter ("gene") to the values it may take:

    {"optimizer.lr": {"log_uniform": [1e-4, 1e-2]},
     "training.batch_size": {"choice": [128, 256, 512]},
     "hidden.layers": {"choice": [1, 2, 3]}}

Keys are dotted paths in the network config (see set_path), or one of the
architecture genes, which rebuild the hidden layers (the output layer is kept
from the base config):

    hidden.layers      number of hidden layers
    hidden.size        neurons of the first hidden layer
    hidden.ratio       size of a hidden layer / size of the previous one (1 = constant, 0.5 = funnel)
    hidden.activation  activation of every hidden layer
    hidden.dropout     dropout of every hidden layer

Value kinds: {"choice": [...]}, {"uniform": [lo, hi]}, {"log_uniform": [lo, hi]},
{"int_uniform": [lo, hi]} (bounds included), {"value": v} (fixed).

The searches do not train anything themselves: they call `evaluate(configs)`,
which must return one result dict per config (see parallel.train_many), so
that every generation / batch of trials is trained in parallel.
"""

import copy
import itertools
import json
import math

import numpy as np

from .config import normalize_config

KINDS = ("choice", "uniform", "log_uniform", "int_uniform", "value")
HIDDEN_GENES = ("layers", "size", "ratio", "activation", "dropout")
OBJECTIVES = {"val_loss": 1, "val_acc": -1, "val_macro_f1": -1}  # 1: minimise, -1: maximise


class SpaceError(ValueError):
    pass


# ------------------------------------------------------------------ config edit
def set_path(config, path, value):
    """Set config[a][b]... = value for path "a.b...". In lists, "*" means every
    item and "hidden" every item but the last: "layers.hidden.dropout"."""
    keys = path.split(".")
    targets = [config]
    for key in keys[:-1]:
        following = []
        for target in targets:
            if isinstance(target, list):
                following.extend(_list_items(target, key, path))
            else:
                if not isinstance(target.get(key), (dict, list)):
                    target[key] = {}
                following.append(target[key])
        targets = following
    last = keys[-1]
    for target in targets:
        if isinstance(target, list):
            for i in _list_indices(target, last, path):
                target[i] = copy.deepcopy(value)
        else:
            target[last] = copy.deepcopy(value)


def _list_indices(items, key, path):
    if key == "*":
        return range(len(items))
    if key == "hidden":
        return range(len(items) - 1)
    try:
        return [int(key)]
    except ValueError:
        raise SpaceError(f"{path}: '{key}' is not a list index (use a number, * or hidden)") from None


def _list_items(items, key, path):
    return [items[i] for i in _list_indices(items, key, path)]


def apply_genes(base, genes):
    """Network config = base config with the genes applied."""
    config = copy.deepcopy(base)
    hidden = {key[len("hidden."):]: value for key, value in genes.items() if key.startswith("hidden.")}
    if hidden:
        template = config["layers"][:-1]
        first = template[0] if template else {}
        size = hidden.get("size", first.get("size", 128))
        ratio = hidden.get("ratio", 1.0)
        layers = []
        for _ in range(int(hidden.get("layers", len(template)))):
            layer = {"size": max(int(round(size)), 1),
                     "activation": hidden.get("activation", first.get("activation", "relu")),
                     "dropout": hidden.get("dropout", first.get("dropout", 0.0))}
            if "init" in first and "activation" not in hidden:
                layer["init"] = first["init"]
            layers.append(layer)
            size *= ratio
        config["layers"] = layers + [config["layers"][-1]]
    for key, value in genes.items():
        if key in ("optimizer.name", "lr_schedule.name"):
            # another optimizer / schedule: the settings of the old one do not apply to it
            section = config.get(key.split(".")[0])
            section = section if isinstance(section, dict) else {"name": section}
            if section.get("name") != value:
                kept = {"lr": section["lr"]} if "lr" in section else {}
                config[key.split(".")[0]] = {"name": value, **kept}
        elif not key.startswith("hidden."):
            set_path(config, key, value)
    return config


# ------------------------------------------------------------------ space
def check_space(space):
    if not isinstance(space, dict) or not space:
        raise SpaceError("the search space must be a non-empty object")
    for key, spec in space.items():
        if key.startswith("hidden.") and key[len("hidden."):] not in HIDDEN_GENES:
            raise SpaceError(f"{key}: unknown architecture gene (choices: hidden.{', hidden.'.join(HIDDEN_GENES)})")
        if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in KINDS:
            raise SpaceError(f"{key}: must be one of " + ", ".join('{"%s": ...}' % k for k in KINDS))
        kind, values = next(iter(spec.items()))
        if kind == "choice" and (not isinstance(values, list) or not values):
            raise SpaceError(f"{key}: choice needs a non-empty list")
        if kind in ("uniform", "log_uniform", "int_uniform"):
            if not isinstance(values, list) or len(values) != 2 or values[0] > values[1]:
                raise SpaceError(f"{key}: {kind} needs [low, high] with low <= high")
            if kind == "log_uniform" and values[0] <= 0:
                raise SpaceError(f"{key}: log_uniform bounds must be > 0")


def sample(spec, rng):
    kind, values = next(iter(spec.items()))
    if kind == "choice":
        return copy.deepcopy(values[int(rng.integers(len(values)))])
    if kind == "uniform":
        return float(rng.uniform(values[0], values[1]))
    if kind == "log_uniform":
        return float(math.exp(rng.uniform(math.log(values[0]), math.log(values[1]))))
    if kind == "int_uniform":
        return int(rng.integers(values[0], values[1] + 1))
    return copy.deepcopy(values)


def random_genes(space, rng):
    return {key: sample(spec, rng) for key, spec in space.items()}


def grid(space):
    """Every combination of the space; only choice and value genes are allowed."""
    axes = []
    for key, spec in space.items():
        kind, values = next(iter(spec.items()))
        if kind == "value":
            values = [values]
        elif kind != "choice":
            raise SpaceError(f"{key}: grid search only takes choice or value genes")
        axes.append([(key, v) for v in values])
    return [dict(combo) for combo in itertools.product(*axes)]


def mutate(genes, space, rng, rate):
    """Change each gene with probability `rate`: small move for numbers, new value for choices."""
    child = dict(genes)
    for key, spec in space.items():
        if rng.random() >= rate:
            continue
        kind, values = next(iter(spec.items()))
        if kind == "choice" and len(values) > 1:
            others = [v for v in values if v != genes[key]] or values
            child[key] = copy.deepcopy(others[int(rng.integers(len(others)))])
        elif kind == "uniform":
            span = values[1] - values[0]
            child[key] = float(np.clip(genes[key] + rng.normal(0, 0.15 * span), *values))
        elif kind == "log_uniform":
            child[key] = float(np.clip(genes[key] * math.exp(rng.normal(0, 0.5)), *values))
        elif kind == "int_uniform":
            step = int(rng.choice([-1, 1]) * max(1, round(0.15 * (values[1] - values[0]))))
            child[key] = int(np.clip(genes[key] + step, *values))
    return child


def crossover(a, b, rng):
    """Uniform crossover: each gene comes from one parent or the other, 50/50."""
    return {key: copy.deepcopy(a[key] if rng.random() < 0.5 else b[key]) for key in a}


# ------------------------------------------------------------------ scoring
def score(result, objective="val_loss"):
    """Lower is better; failed trainings get +inf."""
    sign = OBJECTIVES[objective]
    value = result.get(objective)
    if result.get("error") or value is None or not math.isfinite(value):
        return math.inf
    return sign * value


def best(results, objective="val_loss"):
    return min(results, key=lambda r: score(r, objective))


def _key(genes):
    return json.dumps(genes, sort_keys=True)


def _evaluate(base, genes_list, evaluate, extra):
    configs = []
    for genes in genes_list:
        config = apply_genes(base, genes)
        normalize_config(config)  # invalid combination: fail now, with the faulty key
        configs.append(config)
    results = evaluate(configs)
    for genes, result in zip(genes_list, results):
        result["genes"] = genes
        result.update(extra)
    return results


# ------------------------------------------------------------------ searches
def grid_search(space, base, evaluate):
    check_space(space)
    return _evaluate(base, grid(space), evaluate, {"method": "grid"})


def random_search(space, base, evaluate, n_trials, rng):
    check_space(space)
    genes_list, seen = [], set()
    for _ in range(n_trials * 20):
        genes = random_genes(space, rng)
        if _key(genes) not in seen:
            seen.add(_key(genes))
            genes_list.append(genes)
        if len(genes_list) == n_trials:
            break
    return _evaluate(base, genes_list, evaluate, {"method": "random"})


def genetic_search(space, base, evaluate, rng, population=10, generations=5, elite=2,
                   mutation_rate=0.25, tournament=3, objective="val_loss", on_generation=None):
    """Genetic algorithm.

    Each generation: the `elite` best individuals are kept as they are; the rest
    of the population are children of two parents picked by tournament (the best
    of `tournament` random individuals), mixed by crossover then mutated.
    Individuals already evaluated are not trained again.
    """
    check_space(space)
    if not 0 <= elite < population:
        raise SpaceError("elite must be in [0, population)")
    pop = [random_genes(space, rng) for _ in range(population)]
    cache, results = {}, []
    for generation in range(generations):
        todo, keys = [], set()
        for genes in pop:
            if _key(genes) not in cache and _key(genes) not in keys:
                keys.add(_key(genes))
                todo.append(genes)
        for result in _evaluate(base, todo, evaluate, {"method": "genetic", "generation": generation}):
            cache[_key(result["genes"])] = result
            results.append(result)
        ranked = sorted(pop, key=lambda genes: score(cache[_key(genes)], objective))
        if on_generation:
            on_generation(generation, [cache[_key(genes)] for genes in ranked])
        if generation == generations - 1:
            break
        following = [dict(genes) for genes in ranked[:elite]]
        while len(following) < population:
            parents = []
            for _ in range(2):
                picked = [pop[int(i)] for i in rng.choice(len(pop), size=min(tournament, len(pop)), replace=False)]
                parents.append(min(picked, key=lambda genes: score(cache[_key(genes)], objective)))
            following.append(mutate(crossover(parents[0], parents[1], rng), space, rng, mutation_rate))
        pop = following
    return results
