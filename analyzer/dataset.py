"""Chess files ('<FEN> [label]' per line) -> encoded inputs and class indices;
deduplication and stratified split of the dataset."""

import numpy as np

from .fen import active_inputs, encode_many, parse_fen, split_line
from .labels import label_index

ENCODE_CHUNK = 50000  # lines encoded at once: bounds the memory of the Python lists


class ChessFileError(ValueError):
    pass


def read_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()
    except UnicodeDecodeError:
        raise ChessFileError(f"{path}: not a text file") from None
    except OSError as e:
        raise ChessFileError(f"{path}: {e.strerror}") from None


def parse_boards(lines, path="<input>", labels=None, require_labels=False):
    """Encode every non-blank line -> (X (N, 773) uint8, y (N,) index in `labels`, -1 if none).

    require_labels (training): every line needs one of `labels`. Errors name the file
    and the line: "train.txt:12: rank 3 ('ppp') has 3 squares instead of 8"."""
    parts, active, y = [], [], []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            fen, label = split_line(line)
            position = parse_fen(fen)
            if label is None and require_labels:
                raise ValueError("no expected output after the FEN (for example 'Check White')")
            index = -1
            if label is not None and labels is not None:
                try:
                    index = label_index(label, labels)
                except ValueError:
                    if require_labels:
                        raise
        except ValueError as e:
            raise ChessFileError(f"{path}:{lineno}: {e}") from None
        active.append(active_inputs(position))
        y.append(index)
        if len(active) == ENCODE_CHUNK:
            parts.append(encode_many(active))
            active = []
    if not y:
        raise ChessFileError(f"{path}: no chessboard in this file")
    return np.concatenate(parts + [encode_many(active)]), np.array(y, dtype=np.int64)


def read_boards(path, labels=None, require_labels=False):
    return parse_boards(read_lines(path), path, labels, require_labels)


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
