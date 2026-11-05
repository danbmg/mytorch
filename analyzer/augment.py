"""Symmetries of chess: the versions of a position whose label is known.

- colors: the board turned upside down (rank 1 <-> rank 8) and White and Black
  swapped (pieces, side to move, castling rights). The label swaps its color:
  "Check White" <-> "Check Black".
- files: the board mirrored left to right (file a <-> file h), castling rights
  king side <-> queen side. The label does not change: every piece moves the
  same way mirrored, and castling, the only rule that is not symmetric, never
  changes a check (a king in check cannot castle).

With both, every position has 4 versions. They serve twice: the training set
is 4x larger at no labelling cost (training part only, never validation or
test), and --predict averages the network's answers on the 4 versions of each
board (analyzer/modes.py). The move generator of the bonus confirms the label
of the 4 versions of the 723 019 dataset positions:
python3 bonus/validate_movegen.py --symmetries
"""

import numpy as np

from .fen import INPUT_SIZE

SIDE = 768  # index of the side-to-move input
SYMMETRIES = ("identity", "colors", "files", "colors+files")


def _board_permutation(flip_colors, flip_files):
    """perm such that X[:, perm] is the encoding of the transformed board (the side bit apart).

    perm[i] is the input that input i becomes. X[:, perm] needs the opposite (where each
    input comes from), which is the same thing here: every symmetry is its own inverse."""
    perm = np.arange(INPUT_SIZE)
    for plane in range(12):
        for square in range(64):
            row, col = divmod(square, 8)
            new_plane = plane
            if flip_colors:
                new_plane = (plane + 6) % 12  # P..K <-> p..k
                row = 7 - row
            if flip_files:
                col = 7 - col
            perm[plane * 64 + square] = new_plane * 64 + 8 * row + col
    castling = {"K": 769, "Q": 770, "k": 771, "q": 772}
    for right, index in castling.items():
        if flip_colors:
            right = right.swapcase()
        if flip_files:
            right = {"K": "Q", "Q": "K", "k": "q", "q": "k"}[right]
        perm[index] = castling[right]
    return perm


PERMUTATIONS = {
    "identity": _board_permutation(False, False),
    "colors": _board_permutation(True, False),
    "files": _board_permutation(False, True),
    "colors+files": _board_permutation(True, True),
}


def flip_label(label):
    """'Check White' -> 'Check Black'; a label without a color is unchanged."""
    return label.replace("White", "\0").replace("Black", "White").replace("\0", "Black")


def label_mapping(symmetry, labels):
    """mapping[i] = index of the label of class i once the board is transformed."""
    if "colors" not in symmetry:
        return np.arange(len(labels))
    return np.array([labels.index(flip_label(label)) for label in labels])


def transform(X, symmetry):
    """Encoded boards X (N, 773) -> the same boards transformed by `symmetry`."""
    X = np.asarray(X)[:, PERMUTATIONS[symmetry]]
    if "colors" in symmetry:
        X[:, SIDE] = 1 - X[:, SIDE]
    return X


def augment(X, y, labels, symmetries):
    """X and y followed by their transformed copies, one copy per symmetry -> (X', y').

    The copies are written straight into the final array: 4x the training set
    (1.8 GB) without a temporary copy of it."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    X_out = np.empty((n * (len(symmetries) + 1), X.shape[1]), dtype=X.dtype)
    y_out = np.empty(n * (len(symmetries) + 1), dtype=y.dtype)
    X_out[:n] = X
    y_out[:n] = y
    for k, symmetry in enumerate(symmetries, 1):
        part = slice(k * n, (k + 1) * n)
        np.take(X, PERMUTATIONS[symmetry], axis=1, out=X_out[part], mode="clip")  # "clip": no hidden buffer
        if "colors" in symmetry:
            X_out[part, SIDE] = 1 - X_out[part, SIDE]
        mapping = label_mapping(symmetry, labels)
        y_out[part] = np.where(y >= 0, mapping[np.maximum(y, 0)], y)  # -1 (no label) stays -1
    return X_out, y_out


def usable_symmetries(labels):
    """All 4 symmetries when every label has its color-swapped counterpart among the
    labels (the chess labels do), else only those that keep the colors."""
    if all(flip_label(label) in labels for label in labels):
        return SYMMETRIES
    return ("identity", "files")


def symmetric_proba(net, X, labels):
    """The network's probabilities averaged over the symmetric versions of each board.

    The answer on a transformed board is about the transformed label: its columns
    are put back in the order of the original labels before averaging."""
    symmetries = usable_symmetries(labels)
    total = np.zeros((len(X), len(labels)))
    for symmetry in symmetries:
        proba = net.predict_proba(transform(X, symmetry))
        total += proba[:, label_mapping(symmetry, labels)]
    return total / len(symmetries)


def transform_fen(fen, symmetry):
    """The FEN of the transformed position (to check the encoded version and the labels)."""
    board, side, castling, ep, *counters = fen.split()
    rows = board.split("/")
    if "colors" in symmetry:
        rows = [row.swapcase() for row in reversed(rows)]
        side = "b" if side == "w" else "w"
        castling = castling.swapcase()
        ep = ep if ep == "-" else ep[0] + {"3": "6", "6": "3"}[ep[1]]
    if "files" in symmetry:
        rows = [row[::-1] for row in rows]
        castling = castling.translate(str.maketrans("KQkq", "QKqk"))
        ep = ep if ep == "-" else "abcdefgh"[7 - "abcdefgh".index(ep[0])] + ep[1]
    castling = "".join(c for c in "KQkq" if c in castling) or "-"
    return " ".join(["/".join(rows), side, castling, ep, *counters])
