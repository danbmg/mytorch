"""Lines of a chess file: '<FEN> [expected output]'.

The FEN is the first 4 fields plus the move counters when present (the only
numeric fields), the label is the rest: 'Check White' with its space needs no
quoting. position_key() ignores the counters and the order of the castling
letters, so that two lines of the same position are seen as duplicates.
"""

import re

from .parse import parse_fen


def split_line(line):
    """'<FEN> [label]' -> (fen, label or None)."""
    tokens = line.split()
    if len(tokens) < 4:
        raise ValueError(f"expected a FEN position (at least 4 fields), got {len(tokens)} field(s)")
    n = 4
    while n < min(6, len(tokens)) and tokens[n].isdigit():
        n += 1
    return " ".join(tokens[:n]), " ".join(tokens[n:]) or None


def parse_line(line):
    """Same as split_line, and the FEN is validated."""
    fen, label = split_line(line)
    parse_fen(fen)
    return fen, label


def position_key(fen):
    """Board, side, castling and en passant in canonical form: two lines with the same key
    are the same position whatever their move counters (used to remove duplicates)."""
    board, side, castling, ep = fen.split()[:4]
    board = re.sub(r"\d{2,}", lambda m: str(sum(int(d) for d in m.group())), board)
    return f"{board} {side} {''.join(c for c in 'KQkq' if c in castling) or '-'} {ep}"
