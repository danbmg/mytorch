"""The search space: what each gene may be, how one value is drawn, every
combination for a grid. One sampler per kind of value in a registry, so that
check_space() and sample() agree on the kinds."""

import copy
import itertools
import math

HIDDEN_GENES = ("layers", "size", "ratio", "activation", "dropout")


class SpaceError(ValueError):
    pass


def _choice(values, rng):
    return copy.deepcopy(values[int(rng.integers(len(values)))])


def _uniform(values, rng):
    return float(rng.uniform(values[0], values[1]))


def _log_uniform(values, rng):
    return float(math.exp(rng.uniform(math.log(values[0]), math.log(values[1]))))


def _value(values, rng):
    return copy.deepcopy(values)


SAMPLERS = {"choice": _choice, "uniform": _uniform, "log_uniform": _log_uniform, "value": _value}
KINDS = tuple(SAMPLERS)


def _check_gene(key, spec):
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


def check_space(space):
    if not isinstance(space, dict) or not space:
        raise SpaceError("the search space must be a non-empty object")
    for key, spec in space.items():
        _check_gene(key, spec)


def sample(spec, rng):
    """One value drawn from a gene spec: {"choice": [...]}, {"uniform": [lo, hi]}..."""
    kind, values = next(iter(spec.items()))
    return SAMPLERS[kind](values, rng)


def grid(space):
    """Every combination of the space (choice and value genes only)."""
    axes = []
    for key, spec in space.items():
        kind, values = next(iter(spec.items()))
        if kind not in ("choice", "value"):
            raise SpaceError(f"{key}: grid search only takes choice or value genes")
        axes.append([(key, v) for v in (values if kind == "choice" else [values])])
    return [dict(combo) for combo in itertools.product(*axes)]
