"""Chess positions in FEN notation: parsing, validation, encoding as network inputs.

Encoding: 773 values 0/1 (uint8)
  0..767    12 planes of 64 squares, one per piece and colour (P N B R Q K p n b r q k);
            square = 8 * row + column, row 0 = 8th rank (FEN order), column 0 = file a
  768       side to move (1 = white)
  769..772  castling rights K Q k q
Not encoded: the move counters (they never change a check) and the en passant
square (209 of the 723 019 dataset lines have one).
"""

import re
from collections import namedtuple

import numpy as np

PIECES = "PNBRQKpnbrqk"
CASTLING = {"K": 769, "Q": 770, "k": 771, "q": 772}
INPUT_SIZE = 773

Position = namedtuple("Position", "pieces side castling en_passant")  # pieces: [(square, letter)]


class FenError(ValueError):
    pass


def parse_fen(fen):
    """FEN (4 to 6 fields: the move counters are optional) -> Position. FenError names the faulty field."""
    fields = fen.split()
    if not 4 <= len(fields) <= 6:
        raise FenError(f"a FEN has 6 fields (at least 4), got {len(fields)}")
    board, side, castling, ep = fields[:4]
    rows = board.split("/")
    if len(rows) != 8:
        raise FenError(f"the board needs 8 ranks separated by '/', got {len(rows)}")
    pieces = []
    for row, text in enumerate(rows):
        col = 0
        for ch in text:
            if "1" <= ch <= "8":
                col += int(ch)
            elif ch in PIECES:
                if col < 8:
                    pieces.append((8 * row + col, ch))
                col += 1
            else:
                raise FenError(f"invalid character '{ch}' in rank {8 - row} ('{text}')")
        if col != 8:
            raise FenError(f"rank {8 - row} ('{text}') has {col} squares instead of 8")
    for king, colour in (("K", "white"), ("k", "black")):
        count = sum(piece == king for _, piece in pieces)
        if count != 1:
            raise FenError(f"the board needs exactly one {colour} king, got {count}")
    if side not in ("w", "b"):
        raise FenError(f"side to move must be 'w' or 'b', got '{side}'")
    if castling != "-" and (len(set(castling)) != len(castling) or not set(castling) <= set(CASTLING)):
        raise FenError(f"castling rights must be '-' or letters among KQkq, got '{castling}'")
    if ep != "-" and not (len(ep) == 2 and ep[0] in "abcdefgh" and ep[1] in "36"):
        raise FenError(f"en passant square must be '-' or a square of rank 3 or 6, got '{ep}'")
    for name, value in zip(("halfmove clock", "fullmove number"), fields[4:]):
        if not value.isdigit():
            raise FenError(f"{name} must be a non-negative integer, got '{value}'")
    return Position(pieces, side, castling, ep)


def position_key(fen):
    """Board, side, castling and en passant in canonical form: two lines with the same key
    are the same position whatever their move counters (used to remove duplicates)."""
    board, side, castling, ep = fen.split()[:4]
    board = re.sub(r"\d{2,}", lambda m: str(sum(int(d) for d in m.group())), board)
    return f"{board} {side} {''.join(c for c in 'KQkq' if c in castling) or '-'} {ep}"


def split_line(line):
    """'<FEN> [label]' -> (fen, label or None). The FEN is the first 4 fields, plus the
    move counters when present (the only numeric fields); the label is the rest."""
    tokens = line.split()
    if len(tokens) < 4:
        raise FenError(f"expected a FEN position (at least 4 fields), got {len(tokens)} field(s)")
    n = 4
    while n < min(6, len(tokens)) and tokens[n].isdigit():
        n += 1
    return " ".join(tokens[:n]), " ".join(tokens[n:]) or None


def parse_line(line):
    """Same as split_line, and the FEN is validated."""
    fen, label = split_line(line)
    parse_fen(fen)
    return fen, label


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
