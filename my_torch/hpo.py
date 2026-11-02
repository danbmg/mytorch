"""Hyperparameter search: grid, random search and genetic algorithm.

A search space maps each hyperparameter ("gene") to its possible values:
    {"optimizer.lr": {"log_uniform": [1e-4, 1e-2]}, "hidden.layers": {"choice": [1, 2, 3]}}
Value kinds: {"choice": [...]}, {"uniform": [lo, hi]}, {"log_uniform": [lo, hi]}, {"value": v}.

Keys are dotted paths in the config (see set_path), or architecture genes that
rebuild the hidden layers (the output layer is kept from the base config):
hidden.layers, hidden.size (first layer), hidden.ratio (next layer / previous),
hidden.activation, hidden.dropout.

The searches train nothing themselves: they call evaluate(configs), which returns
one result per config (parallel.train_many), so each batch of trials is trained
in parallel.
"""

import copy
import itertools
import json
import math

import numpy as np

from .config import normalize_config

KINDS = ("choice", "uniform", "log_uniform", "value")
HIDDEN_GENES = ("layers", "size", "ratio", "activation", "dropout")
OBJECTIVES = {"val_loss": 1, "val_acc": -1, "val_macro_f1": -1}  # 1: minimise, -1: maximise


class SpaceError(ValueError):
    pass


def _indices(items, key, path):
    if key == "*":
        return range(len(items))
    if key == "hidden":
        return range(len(items) - 1)
    if key.isdigit():
        return [int(key)]
    raise SpaceError(f"{path}: '{key}' is not a list index (use a number, * or hidden)")


def set_path(config, path, value):
    """config[a][b]... = value for path "a.b...". In a list, "*" is every item,
    "hidden" every item but the last, a number one item: "layers.hidden.dropout"."""
    *keys, last = path.split(".")
    targets = [config]
    for key in keys:
        following = []
        for target in targets:
            if isinstance(target, list):
                following += [target[i] for i in _indices(target, key, path)]
            else:
                if not isinstance(target.get(key), (dict, list)):
                    target[key] = {}
                following.append(target[key])
        targets = following
    for target in targets:
        for i in (_indices(target, last, path) if isinstance(target, list) else [last]):
            target[i] = copy.deepcopy(value)


def apply_genes(base, genes):
    """Network config = base config with the genes applied."""
    config = copy.deepcopy(base)
    hidden = {k[len("hidden."):]: v for k, v in genes.items() if k.startswith("hidden.")}
    if hidden:
        first = config["layers"][0] if len(config["layers"]) > 1 else {}
        size = hidden.get("size", first.get("size", 128))
        layers = []
        for _ in range(int(hidden.get("layers", len(config["layers"]) - 1))):
            layers.append({"size": max(int(round(size)), 1),
                           "activation": hidden.get("activation", first.get("activation", "relu")),
                           "dropout": hidden.get("dropout", first.get("dropout", 0.0))})
            size *= hidden.get("ratio", 1.0)
        config["layers"] = layers + [config["layers"][-1]]
    for key, value in genes.items():
        section = key.split(".")[0]
        if key in ("optimizer.name", "lr_schedule.name") and config.get(section, {}).get("name") != value:
            # another optimizer / schedule: the settings of the old one do not apply, the lr does
            config[section] = {"name": value, **({"lr": config[section]["lr"]} if "lr" in config.get(section, {}) else {})}
        elif not key.startswith("hidden."):
            set_path(config, key, value)
    return config


def check_space(space):
    if not isinstance(space, dict) or not space:
        raise SpaceError("the search space must be a non-empty object")
    for key, spec in space.items():
        if key.startswith("hidden.") and key[len("hidden."):] not in HIDDEN_GENES:
            raise SpaceError(f"{key}: unknown architecture gene (choices: hidden.{', hidden.'.join(HIDDEN_GENES)})")
        if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in KINDS:
            raise SpaceError(f"{key}: must be one of " + ", ".join('{"%s": ...}' % k for k in KINDS))
        kind, values = next(iter(spec.items()))
        if kind == "choice" and not (isinstance(values, list) and values):
            raise SpaceError(f"{key}: choice needs a non-empty list")
        if kind in ("uniform", "log_uniform"):
            ok = isinstance(values, list) and len(values) == 2 and values[0] <= values[1]
            if not ok or (kind == "log_uniform" and values[0] <= 0):
                raise SpaceError(f"{key}: {kind} needs [low, high] with low <= high (low > 0 for log_uniform)")


def sample(spec, rng):
    kind, values = next(iter(spec.items()))
    if kind == "choice":
        return copy.deepcopy(values[int(rng.integers(len(values)))])
    if kind == "uniform":
        return float(rng.uniform(values[0], values[1]))
    if kind == "log_uniform":
        return float(math.exp(rng.uniform(math.log(values[0]), math.log(values[1]))))
    return copy.deepcopy(values)


def grid(space):
    """Every combination of the space (choice and value genes only)."""
    axes = []
    for key, spec in space.items():
        kind, values = next(iter(spec.items()))
        if kind not in ("choice", "value"):
            raise SpaceError(f"{key}: grid search only takes choice or value genes")
        axes.append([(key, v) for v in (values if kind == "choice" else [values])])
    return [dict(combo) for combo in itertools.product(*axes)]


def mutate(genes, space, rng, rate):
    """Each gene changes with probability `rate`: another value for a choice, a small move for a number."""
    child = dict(genes)
    for key, spec in space.items():
        if rng.random() >= rate:
            continue
        kind, values = next(iter(spec.items()))
        if kind == "choice" and len(values) > 1:
            others = [v for v in values if v != genes[key]]
            child[key] = copy.deepcopy(others[int(rng.integers(len(others)))])
        elif kind == "uniform":
            child[key] = float(np.clip(genes[key] + rng.normal(0, 0.15 * (values[1] - values[0])), *values))
        elif kind == "log_uniform":
            child[key] = float(np.clip(genes[key] * math.exp(rng.normal(0, 0.5)), *values))
    return child


def crossover(a, b, rng):
    """Uniform crossover: each gene comes from one parent or the other, 50/50."""
    return {key: copy.deepcopy(a[key] if rng.random() < 0.5 else b[key]) for key in a}


def score(result, objective="val_loss"):
    """Lower is better; a failed training scores +inf."""
    value = result.get(objective)
    if result.get("error") or value is None or not math.isfinite(value):
        return math.inf
    return OBJECTIVES[objective] * value


def _key(genes):
    return json.dumps(genes, sort_keys=True)


def _evaluate(base, genes_list, evaluate, extra):
    configs = [apply_genes(base, genes) for genes in genes_list]
    for config in configs:
        normalize_config(config)  # an invalid combination fails now, naming the faulty key
    results = evaluate(configs)
    for genes, result in zip(genes_list, results):
        result.update(genes=genes, **extra)
    return results


def grid_search(space, base, evaluate):
    check_space(space)
    return _evaluate(base, grid(space), evaluate, {"method": "grid"})


def random_search(space, base, evaluate, n_trials, rng):
    check_space(space)
    unique = {}
    for _ in range(n_trials * 20):
        genes = {key: sample(spec, rng) for key, spec in space.items()}
        unique.setdefault(_key(genes), genes)
        if len(unique) == n_trials:
            break
    return _evaluate(base, list(unique.values()), evaluate, {"method": "random"})


def genetic_search(space, base, evaluate, rng, population=10, generations=5, elite=2,
                   mutation_rate=0.25, tournament=3, objective="val_loss", on_generation=None):
    """Each generation keeps its `elite` best individuals; the others are replaced by
    children of two parents picked by tournament (best of `tournament` random
    individuals), crossed then mutated. Known individuals are not trained again."""
    check_space(space)
    if not 0 <= elite < population:
        raise SpaceError("elite must be in [0, population)")
    pop = [{key: sample(spec, rng) for key, spec in space.items()} for _ in range(population)]
    cache, results = {}, []
    for generation in range(generations):
        todo = list({_key(g): g for g in pop if _key(g) not in cache}.values())
        for result in _evaluate(base, todo, evaluate, {"method": "genetic", "generation": generation}):
            cache[_key(result["genes"])] = result
            results.append(result)
        fitness = lambda genes: score(cache[_key(genes)], objective)  # noqa: E731
        ranked = sorted(pop, key=fitness)
        if on_generation:
            on_generation(generation, [cache[_key(g)] for g in ranked])
        if generation == generations - 1:
            break
        following = [dict(g) for g in ranked[:elite]]
        while len(following) < population:
            parents = [min((pop[int(i)] for i in rng.choice(population, size=min(tournament, population), replace=False)),
                           key=fitness) for _ in range(2)]
            following.append(mutate(crossover(*parents, rng), space, rng, mutation_rate))
        pop = following
    return results
