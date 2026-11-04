"""Data augmentation by colour inversion: the mirror position where White and
Black swap places.

The board is flipped vertically (rank 1 becomes rank 8), the pieces change
colour, the side to move and the castling rights are swapped, and so is the
colour of the label: "Check White" becomes "Check Black". The rules of chess
are symmetric, so the mirrored position has exactly the mirrored label: this
doubles the training data at no labelling cost. Applied to the training part
only, never to validation or test.
"""

import numpy as np

from .fen import INPUT_SIZE

_PLANES, _SQUARES = 12, 64


def _permutation():
    """perm such that X[:, perm] is the mirrored encoding (the side bit is inverted apart)."""
    perm = np.arange(INPUT_SIZE)
    for plane in range(_PLANES):
        for square in range(_SQUARES):
            row, col = divmod(square, 8)
            perm[plane * _SQUARES + square] = (plane + 6) % _PLANES * _SQUARES + 8 * (7 - row) + col
    perm[769], perm[770], perm[771], perm[772] = 771, 772, 769, 770  # K Q k q -> k q K Q
    return perm


PERMUTATION = _permutation()


def flip_label(label):
    """'Check White' -> 'Check Black'; a label without a colour is unchanged."""
    return label.replace("White", "\0").replace("Black", "White").replace("\0", "Black")


def flip_colors(X, y, labels):
    """Mirrored copies of the encoded boards X (N, 773) and of their class indices y."""
    X = np.asarray(X)[:, PERMUTATION]
    X[:, 768] = 1 - X[:, 768]
    mapping = np.array([labels.index(flip_label(label)) for label in labels])
    y = np.asarray(y)
    return X, np.where(y >= 0, mapping[np.maximum(y, 0)], y)


def flip_fen(fen):
    """The FEN of the mirrored position (for checks against the encoded version)."""
    fields = fen.split()
    board = "/".join(row.swapcase() for row in reversed(fields[0].split("/")))
    side = "b" if fields[1] == "w" else "w"
    castling = "".join(c for c in "KQkq" if c.swapcase() in fields[2]) or "-"
    ep = fields[3] if fields[3] == "-" else fields[3][0] + {"3": "6", "6": "3"}[fields[3][1]]
    return " ".join([board, side, castling, ep] + fields[4:])
