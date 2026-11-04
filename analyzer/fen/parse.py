"""Parsing and validation of a FEN: board, side to move, castling rights, en
passant square, move counters (optional).

Every check names the faulty field and its value, so that an invalid line in a
chess file is reported as "train.txt:12: rank 3 ('ppp') has 3 squares instead
of 8" and nothing is silently predicted on a broken position.
"""

from collections import namedtuple

PIECES = "PNBRQKpnbrqk"
CASTLING_LETTERS = "KQkq"

Position = namedtuple("Position", "pieces side castling en_passant")  # pieces: [(square, letter)]


class FenError(ValueError):
    pass


def _rank(row, text):
    """The pieces of one rank -> [(square, letter)]; square = 8 * row + column."""
    col, pieces = 0, []
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
    return pieces


def _board(board):
    rows = board.split("/")
    if len(rows) != 8:
        raise FenError(f"the board needs 8 ranks separated by '/', got {len(rows)}")
    pieces = [piece for row, text in enumerate(rows) for piece in _rank(row, text)]
    for king, colour in (("K", "white"), ("k", "black")):
        count = sum(piece == king for _, piece in pieces)
        if count != 1:
            raise FenError(f"the board needs exactly one {colour} king, got {count}")
    return pieces


def _fields(side, castling, ep, counters):
    if side not in ("w", "b"):
        raise FenError(f"side to move must be 'w' or 'b', got '{side}'")
    if castling != "-" and (len(set(castling)) != len(castling) or not set(castling) <= set(CASTLING_LETTERS)):
        raise FenError(f"castling rights must be '-' or letters among KQkq, got '{castling}'")
    if ep != "-" and not (len(ep) == 2 and ep[0] in "abcdefgh" and ep[1] in "36"):
        raise FenError(f"en passant square must be '-' or a square of rank 3 or 6, got '{ep}'")
    for name, value in zip(("halfmove clock", "fullmove number"), counters):
        if not value.isdigit():
            raise FenError(f"{name} must be a non-negative integer, got '{value}'")


def parse_fen(fen):
    """FEN (4 to 6 fields: the move counters are optional) -> Position. FenError names the faulty field."""
    fields = fen.split()
    if not 4 <= len(fields) <= 6:
        raise FenError(f"a FEN has 6 fields (at least 4), got {len(fields)}")
    board, side, castling, ep = fields[:4]
    pieces = _board(board)
    _fields(side, castling, ep, fields[4:])
    return Position(pieces, side, castling, ep)
