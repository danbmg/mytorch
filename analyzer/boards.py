"""Chess files ('<FEN> [label]' per line) -> encoded inputs and class indices.

Lines are parsed one by one (a bad one is reported with its number) but encoded
by chunks of 50 000 in one numpy call each: the Python lists stay small and the
723 019 lines of the dataset are encoded in a few seconds.
"""

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


def _label(label, labels, require_labels):
    """Index of the label in `labels` (-1 if none or unknown); training requires a known one."""
    if label is None and require_labels:
        raise ValueError("no expected output after the FEN (for example 'Check White')")
    if label is None or labels is None:
        return -1
    try:
        return label_index(label, labels)
    except ValueError:
        if require_labels:
            raise
        return -1


def _board(line, labels, require_labels):
    """One line -> (active inputs, class index); ValueError names what is wrong."""
    fen, label = split_line(line)
    position = parse_fen(fen)
    index = _label(label, labels, require_labels)
    return active_inputs(position), index


def parse_boards(lines, path="<input>", labels=None, require_labels=False):
    """Encode every non-blank line -> (X (N, 773) uint8, y (N,) index in `labels`, -1 if none).

    require_labels (training): every line needs one of `labels`. Errors name the file
    and the line: "train.txt:12: rank 3 ('ppp') has 3 squares instead of 8"."""
    parts, active, y = [], [], []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            inputs, index = _board(line, labels, require_labels)
        except ValueError as e:
            raise ChessFileError(f"{path}:{lineno}: {e}") from None
        active.append(inputs)
        y.append(index)
        if len(active) == ENCODE_CHUNK:
            parts.append(encode_many(active))
            active = []
    if not y:
        raise ChessFileError(f"{path}: no chessboard in this file")
    return np.concatenate(parts + [encode_many(active)]), np.array(y, dtype=np.int64)


def read_boards(path, labels=None, require_labels=False):
    return parse_boards(read_lines(path), path, labels, require_labels)
