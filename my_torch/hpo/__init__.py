"""Hyperparameter search: grid, random search and genetic algorithm.

A search space maps each hyperparameter ("gene") to its possible values:
    {"optimizer.lr": {"log_uniform": [1e-4, 1e-2]}, "hidden.layers": {"choice": [1, 2, 3]}}
Value kinds: {"choice": [...]}, {"uniform": [lo, hi]}, {"log_uniform": [lo, hi]}, {"value": v}.

Keys are dotted paths in the config (genes.py), or architecture genes that
rebuild the hidden layers (the output layer is kept from the base config):
hidden.layers, hidden.size (first layer), hidden.ratio (next layer / previous),
hidden.activation, hidden.dropout.

The searches train nothing themselves: they call evaluate(configs), which returns
one result per config (parallel.train_many), so each batch of trials is trained
in parallel. space.py: the space and its sampling; search.py: grid and random;
genetic.py: the genetic algorithm.
"""

from .genes import apply_genes, set_path
from .genetic import crossover, genetic_search, mutate
from .search import OBJECTIVES, grid_search, random_search, score
from .space import HIDDEN_GENES, KINDS, SpaceError, check_space, grid, sample

__all__ = ["HIDDEN_GENES", "KINDS", "OBJECTIVES", "SpaceError", "apply_genes", "check_space", "crossover",
           "genetic_search", "grid", "grid_search", "mutate", "random_search", "sample", "score", "set_path"]
