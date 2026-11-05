"""Genetic algorithm: each generation keeps its `elite` best individuals; the
others are replaced by children of two parents picked by tournament (best of
`tournament` random individuals), crossed then mutated. Known individuals are
not trained again (a cache keyed by their genes)."""

import copy
import math

import numpy as np

from .search import evaluate_genes, key, score
from .space import SpaceError, check_space, sample


def mutate(genes, space, rng, rate):
    """Each gene changes with probability `rate`: another value for a choice, a small move for a number."""
    child = dict(genes)
    for k, spec in space.items():
        if rng.random() >= rate:
            continue
        kind, values = next(iter(spec.items()))
        if kind == "choice" and len(values) > 1:
            others = [v for v in values if v != genes[k]]
            child[k] = copy.deepcopy(others[int(rng.integers(len(others)))])
        elif kind == "uniform":
            child[k] = float(np.clip(genes[k] + rng.normal(0, 0.15 * (values[1] - values[0])), *values))
        elif kind == "log_uniform":
            child[k] = float(np.clip(genes[k] * math.exp(rng.normal(0, 0.5)), *values))
    return child


def crossover(a, b, rng):
    """Uniform crossover: each gene comes from one parent or the other, 50/50."""
    return {k: copy.deepcopy(a[k] if rng.random() < 0.5 else b[k]) for k in a}


def _tournament(pop, fitness, rng, size):
    """The best of `size` individuals drawn at random."""
    return min((pop[int(i)] for i in rng.choice(len(pop), size=min(size, len(pop)), replace=False)), key=fitness)


def _next_generation(pop, ranked, fitness, space, rng, elite, mutation_rate, tournament):
    following = [dict(g) for g in ranked[:elite]]
    while len(following) < len(pop):
        parents = [_tournament(pop, fitness, rng, tournament) for _ in range(2)]
        following.append(mutate(crossover(*parents, rng), space, rng, mutation_rate))
    return following


def genetic_search(space, base, evaluate, rng, population=10, generations=5, elite=2,
                   mutation_rate=0.25, tournament=3, objective="val_loss", on_generation=None):
    """Every result of every generation; on_generation(g, ranked results) after each one."""
    check_space(space)
    if not 0 <= elite < population:
        raise SpaceError("elite must be in [0, population)")
    pop = [{k: sample(spec, rng) for k, spec in space.items()} for _ in range(population)]
    cache = {}  # genes key -> result: an individual is never trained twice
    results = []
    for generation in range(generations):
        todo = list({key(g): g for g in pop if key(g) not in cache}.values())
        for result in evaluate_genes(base, todo, evaluate, {"method": "genetic", "generation": generation}):
            cache[key(result["genes"])] = result
            results.append(result)

        def fitness(genes):
            return score(cache[key(genes)], objective)

        ranked = sorted(pop, key=fitness)
        if on_generation:
            on_generation(generation, [cache[key(g)] for g in ranked])
        if generation == generations - 1:
            break
        pop = _next_generation(pop, ranked, fitness, space, rng, elite, mutation_rate, tournament)
    return results
