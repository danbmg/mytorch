"""Deduplication and stratified split of a dataset.

Duplicates are removed before the split, else the same position can be in the
training and the test parts (7 460 duplicates in the subject's dataset). The
split is stratified: every group (class, or class x source file) is cut in the
same proportions, and the same seed gives the same parts.
"""

import numpy as np


def deduplicate(keys, labels):
    """Indices of the first occurrence of each position. A position seen with two different
    labels is ambiguous: all its lines are dropped. -> (kept, n_duplicates, n_conflicts)."""
    first, conflicts = {}, set()
    for i, (key, label) in enumerate(zip(keys, labels)):
        if first.setdefault(key, i) != i and labels[first[key]] != label:
            conflicts.add(key)
    kept = sorted(i for key, i in first.items() if key not in conflicts)
    return np.array(kept, dtype=np.int64), len(keys) - len(first), len(conflicts)


def stratified_split(groups, fractions, seed):
    """Sample indices split into len(fractions) shuffled parts, each group (class, or class x
    source file) cut in the same proportions. Same seed, same parts."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    bounds = np.cumsum(fractions) / sum(fractions)
    parts = [[] for _ in fractions]
    for group in np.unique(groups):
        idx = rng.permutation(np.flatnonzero(groups == group))
        for part, chunk in zip(parts, np.split(idx, np.round(bounds[:-1] * len(idx)).astype(int))):
            part.append(chunk)
    return [rng.permutation(np.concatenate(part)) for part in parts]
