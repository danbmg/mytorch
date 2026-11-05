"""Legal move generator for the stalemate bonus (no chess library).

It labels a position exactly (Nothing, Check <Color>, Checkmate <Color>, Stalemate)
and finds the 723 019 labels of the dataset (bonus/validate_movegen.py).

Board: list of 64 characters ('.' empty, white upper case), square = 8 * row + column,
row 0 = 8th rank (as in analyzer/fen.py). Legal moves = pseudo-legal moves (pieces,
castling, en passant, promotions) that do not leave their own king attacked: that
single test covers pins, discovered checks and an en passant capture that uncovers a rank.
"""

from analyzer.fen import parse_fen

ROOK_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BISHOP_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
KNIGHT_JUMPS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))


def _on_board(sq, dr, dc, steps):
    """Squares reached from sq by moving (dr, dc) up to `steps` times, nearest first."""
    r, c, out = sq // 8, sq % 8, []
    for k in range(1, steps + 1):
        if not (0 <= r + k * dr < 8 and 0 <= c + k * dc < 8):
            break
        out.append(8 * (r + k * dr) + c + k * dc)
    return tuple(out)


KNIGHT = [tuple(t for d in KNIGHT_JUMPS for t in _on_board(sq, *d, 1)) for sq in range(64)]
KING = [tuple(t for d in ROOK_DIRS + BISHOP_DIRS for t in _on_board(sq, *d, 1)) for sq in range(64)]
ROOK_RAYS = [tuple(filter(None, (_on_board(sq, *d, 7) for d in ROOK_DIRS))) for sq in range(64)]
BISHOP_RAYS = [tuple(filter(None, (_on_board(sq, *d, 7) for d in BISHOP_DIRS))) for sq in range(64)]
# (right, king from, king to, rook from, rook to, squares that must be empty, squares that must not be attacked)
CASTLES = (("K", 60, 62, 63, 61, (61, 62), (60, 61, 62)), ("Q", 60, 58, 56, 59, (57, 58, 59), (60, 59, 58)),
           ("k", 4, 6, 7, 5, (5, 6), (4, 5, 6)), ("q", 4, 2, 0, 3, (1, 2, 3), (4, 3, 2)))


class Board:
    def __init__(self, squares, white, castling="-", ep=None):
        self.squares = squares  # 64 characters, '.' = empty
        self.white = white  # True: White to move
        self.castling = castling
        self.ep = ep  # en passant target square, or None

    @classmethod
    def from_fen(cls, fen):
        pos = parse_fen(fen)
        squares = ["."] * 64
        for sq, piece in pos.pieces:
            squares[sq] = piece
        ep = None
        if pos.en_passant != "-":
            ep = 8 * (8 - int(pos.en_passant[1])) + "abcdefgh".index(pos.en_passant[0])
        return cls(squares, pos.side == "w", pos.castling, ep)

    def to_fen(self):
        rows = []
        for r in range(8):
            row = "".join(self.squares[8 * r:8 * r + 8])
            for n in range(8, 0, -1):
                row = row.replace("." * n, str(n))
            rows.append(row)
        ep = "-" if self.ep is None else "abcdefgh"[self.ep % 8] + str(8 - self.ep // 8)
        return f"{'/'.join(rows)} {'w' if self.white else 'b'} {self.castling} {ep} 0 1"

    def king(self, white):
        return self.squares.index("K" if white else "k")


def attacked(squares, sq, by_white):
    """Is square `sq` attacked by the pieces of one side?"""
    pawn, knight, bishop, rook, queen, king = "PNBRQK" if by_white else "pnbrqk"
    r, c = divmod(sq, 8)
    pr = r + 1 if by_white else r - 1  # a white pawn attacks towards row 0, so from the row below
    if 0 <= pr < 8 and any(0 <= pc < 8 and squares[8 * pr + pc] == pawn for pc in (c - 1, c + 1)):
        return True
    if any(squares[t] == knight for t in KNIGHT[sq]) or any(squares[t] == king for t in KING[sq]):
        return True
    for rays, slider in ((ROOK_RAYS, rook), (BISHOP_RAYS, bishop)):
        for ray in rays[sq]:
            for t in ray:
                if squares[t] != ".":
                    if squares[t] in (slider, queen):
                        return True
                    break
    return False


def in_check(board):
    return attacked(board.squares, board.king(board.white), not board.white)


def pseudo_moves(board):
    """(from, to, promotion piece or None, 'ep' / 'castle' / None), ignoring the own king's safety."""
    squares, white = board.squares, board.white
    own = str.isupper if white else str.islower
    for sq, p in enumerate(squares):
        if p == "." or not own(p):
            continue
        kind = p.upper()
        if kind == "P":
            yield from _pawn_moves(board, sq, own)
            continue
        if kind in "NK":
            rays = [(t,) for t in (KNIGHT if kind == "N" else KING)[sq]]
        else:
            rays = (ROOK_RAYS[sq] if kind != "B" else ()) + (BISHOP_RAYS[sq] if kind != "R" else ())
        for ray in rays:
            for t in ray:
                if squares[t] == "." or not own(squares[t]):
                    yield (sq, t, None, None)
                if squares[t] != ".":
                    break
    king, rook = ("K", "R") if white else ("k", "r")
    for right, k_from, k_to, r_from, _, empty, safe in CASTLES:
        if (right in board.castling and right.isupper() == white and squares[k_from] == king
                and squares[r_from] == rook and all(squares[s] == "." for s in empty)
                and not any(attacked(squares, s, not white) for s in safe)):
            yield (k_from, k_to, None, "castle")


def _pawn_moves(board, sq, own):
    squares, white = board.squares, board.white
    r, c = divmod(sq, 8)
    r1 = r + (-1 if white else 1)
    if not 0 <= r1 < 8:
        return
    targets = []
    if squares[8 * r1 + c] == ".":
        targets.append(8 * r1 + c)
        r2 = r + (-2 if white else 2)
        if r == (6 if white else 1) and squares[8 * r2 + c] == ".":
            yield (sq, 8 * r2 + c, None, None)
    for t in (8 * r1 + cc for cc in (c - 1, c + 1) if 0 <= cc < 8):
        if squares[t] != "." and not own(squares[t]):
            targets.append(t)
        elif t == board.ep:
            yield (sq, t, None, "ep")
    for t in targets:
        if r1 in (0, 7):
            yield from ((sq, t, piece if white else piece.lower(), None) for piece in "QRBN")
        else:
            yield (sq, t, None, None)


def make_move(squares, move):
    """Squares after the move (a new list; castling rights and en passant are not updated)."""
    frm, to, promotion, special = move
    after = list(squares)
    after[to], after[frm] = promotion or after[frm], "."
    if special == "ep":
        after[8 * (frm // 8) + to % 8] = "."
    elif special == "castle":
        _, _, _, r_from, r_to, _, _ = next(c for c in CASTLES if (c[1], c[2]) == (frm, to))
        after[r_to], after[r_from] = after[r_from], "."
    return after


def legal_moves(board):
    """Legal moves of the side to move (a generator: stop at the first one to know if there is any)."""
    king = board.king(board.white)
    for move in pseudo_moves(board):
        if not attacked(make_move(board.squares, move), move[1] if move[0] == king else king, not board.white):
            yield move


def status(board):
    """'Nothing', 'Check White', 'Checkmate Black', 'Stalemate'... (color = the side giving check)."""
    board = Board.from_fen(board) if isinstance(board, str) else board
    attacker = "Black" if board.white else "White"
    moves = next(legal_moves(board), None) is not None
    if in_check(board):
        return f"Check {attacker}" if moves else f"Checkmate {attacker}"
    return "Nothing" if moves else "Stalemate"
