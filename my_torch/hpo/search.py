"""Grid and random searches, and what every search shares: the score of a
result (lower is better, whatever the objective) and the evaluation of a batch
of gene sets, which validates every config before anything is trained."""

import json
import math

from ..config import normalize_config
from .genes import apply_genes
from .space import check_space, grid, sample

OBJECTIVES = {"val_loss": 1, "val_acc": -1, "val_macro_f1": -1}  # 1: minimize, -1: maximize


def score(result, objective="val_loss"):
    """Lower is better; a failed training scores +inf."""
    value = result.get(objective)
    if result.get("error") or value is None or not math.isfinite(value):
        return math.inf
    return OBJECTIVES[objective] * value


def key(genes):
    """A gene set as a hashable string: the same genes are never trained twice."""
    return json.dumps(genes, sort_keys=True)


def evaluate_genes(base, genes_list, evaluate, extra):
    configs = [apply_genes(base, genes) for genes in genes_list]
    for config in configs:
        normalize_config(config)  # an invalid combination fails now, naming the faulty key
    results = evaluate(configs)
    for genes, result in zip(genes_list, results):
        result.update(genes=genes, **extra)
    return results


def grid_search(space, base, evaluate):
    check_space(space)
    return evaluate_genes(base, grid(space), evaluate, {"method": "grid"})


def random_search(space, base, evaluate, n_trials, rng):
    check_space(space)
    unique = {}
    for _ in range(n_trials * 20):
        genes = {k: sample(spec, rng) for k, spec in space.items()}
        unique.setdefault(key(genes), genes)
        if len(unique) == n_trials:
            break
    return evaluate_genes(base, list(unique.values()), evaluate, {"method": "random"})
