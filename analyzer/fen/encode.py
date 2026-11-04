"""Encoding of a position as network inputs: 773 values 0/1 (uint8)

  0..767    12 planes of 64 squares, one per piece and colour (P N B R Q K p n b r q k);
            square = 8 * row + column, row 0 = 8th rank (FEN order), column 0 = file a
  768       side to move (1 = white)
  769..772  castling rights K Q k q

Not encoded: the move counters (they never change a check) and the en passant
square (209 of the 723 019 dataset lines have one). One-hot rather than a
piece code per square: every piece on every square gets its own weight, no
artificial order between the pieces. Whole files are encoded in one numpy
call from the lists of active inputs (encode_many).
"""

import numpy as np

from .parse import PIECES, parse_fen

INPUT_SIZE = 773
CASTLING = {"K": 769, "Q": 770, "k": 771, "q": 772}


def active_inputs(position):
    """Indices of the inputs equal to 1 for this position."""
    active = [PIECES.index(piece) * 64 + square for square, piece in position.pieces]
    return active + ([768] if position.side == "w" else []) + [CASTLING[c] for c in position.castling if c != "-"]


def encode_many(active_lists):
    """List of active_inputs() results -> (N, 773) uint8 matrix, filled in one numpy call."""
    X = np.zeros((len(active_lists), INPUT_SIZE), dtype=np.uint8)
    rows = [i for i, active in enumerate(active_lists) for _ in active]
    X[rows, [j for active in active_lists for j in active]] = 1
    return X


def fen_to_vector(fen):
    """FEN -> (773,) uint8 vector. Raises FenError (a ValueError) if the FEN is invalid."""
    return encode_many([active_inputs(parse_fen(fen))])[0]
