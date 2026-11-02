#!/usr/bin/env python3
"""Generate labelled stalemate positions and build the 6-class dataset of the bonus.

usage: python3 bonus/gen_stalemate.py [--count 30000] [--data data/chess.npz] [--out bonus/data] [--seed 0]

The subject's dataset has no stalemate. Random positions with little material
(side to move: a king, sometimes pawns or a piece; other side: a king and 1-4
pieces) are labelled exactly by bonus/movegen.py and kept to fill these quotas:
Stalemate N; Nothing with <= 2 legal moves N ("almost stalemate", so the network
does not learn "few pieces = stalemate"); any Nothing N/2; Check N/2; Checkmate N/2.
Deduplicated, split 80/10/10 (stratified, seeded), added to the parts of --data:
OUT/generated.txt and OUT/chess6.npz (labels: the 5 of the subject + Stalemate).
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analyzer.dataset import parse_boards, stratified_split  # noqa: E402
from analyzer.fen import position_key  # noqa: E402
from analyzer.labels import LABELS  # noqa: E402
from bonus.movegen import Board, attacked, in_check, legal_moves  # noqa: E402

import numpy as np  # noqa: E402

LABELS6 = LABELS + ["Stalemate"]
BATCH = 20000  # positions drawn per task
EDGE = [sq for sq in range(64) if sq // 8 in (0, 7) or sq % 8 in (0, 7)]
ATTACKERS = ["Q", "R", "QR", "RR", "QB", "QN", "RB", "RN", "BBN", "QP", "RP", "QPP", "RBN", "QRP", "BNP", "QQ"]
DEFENDERS = ["", "", "", "P", "P", "PP", "B", "N", "R", "PPP"]


def place(rng):
    """A random legal position with the material above, or None."""
    white = rng.random() < 0.5  # side to move = the side that may be stalemated
    squares, free = ["."] * 64, set(range(64))

    def put(piece, choices):
        choices = [s for s in choices if s in free and not (piece in "Pp" and s // 8 in (0, 7))]
        if choices:
            sq = rng.choice(choices)
            squares[sq] = piece
            free.discard(sq)

    own, other = (str.upper, str.lower) if white else (str.lower, str.upper)
    put(own("K"), EDGE if rng.random() < 0.5 else range(64))  # half the time on an edge: where stalemates happen
    put(other("K"), range(64))
    for piece in rng.choice(DEFENDERS):
        put(own(piece), range(64))
    for piece in rng.choice(ATTACKERS):
        put(other(piece), range(64))
    board = Board(squares, white)
    (wr, wc), (br, bc) = divmod(board.king(True), 8), divmod(board.king(False), 8)
    # adjacent kings, or the side that just moved left in check: illegal
    if (abs(wr - br) <= 1 and abs(wc - bc) <= 1) or attacked(squares, board.king(not white), white):
        return None
    return board


def classify(board):
    """Quota of a position: Stalemate, Check, Checkmate, 'Nothing (near)' (<= 2 legal moves) or 'Nothing (any)'."""
    n = 0
    for _ in legal_moves(board):
        n += 1
        if n > 2:
            break
    if in_check(board):
        return ("Check" if n else "Checkmate"), f"{'Check' if n else 'Checkmate'} {'Black' if board.white else 'White'}"
    return ("Stalemate", "Stalemate") if n == 0 else ("Nothing (near)" if n <= 2 else "Nothing (any)", "Nothing")


def work(seed):
    rng = random.Random(seed)
    return [(board.to_fen(), *classify(board)) for board in (place(rng) for _ in range(BATCH)) if board is not None]


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--count", type=int, default=30000)
    p.add_argument("--data", default=os.path.join(ROOT, "data", "chess.npz"))
    p.add_argument("--out", default=os.path.join(ROOT, "bonus", "data"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    if not os.path.exists(args.data):
        print(f"{args.data} missing: run python3 scripts/dedup_split.py first", file=sys.stderr)
        return 84
    quotas = {"Stalemate": args.count, "Nothing (near)": args.count, "Nothing (any)": args.count // 2,
              "Check": args.count // 2, "Checkmate": args.count // 2}
    filled, seen, lines, drawn, t0 = dict.fromkeys(quotas, 0), set(), [], 0, time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        # task i has its own seed: the result only depends on --seed
        for batch in pool.imap(work, (args.seed * 1000003 + i for i in range(10**9))):
            drawn += BATCH
            for fen, quota, label in batch:
                if filled[quota] < quotas[quota] and position_key(fen) not in seen:
                    seen.add(position_key(fen))
                    lines.append(f"{fen} {label}")
                    filled[quota] += 1
            if filled == quotas:
                pool.terminate()
                break
    print(f"{len(lines)} positions kept out of {drawn} drawn in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "generated.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    X_gen, Y_gen = parse_boards(lines, "generated.txt", LABELS6, require_labels=True)
    arrays = {"labels": np.array(LABELS6)}
    with np.load(args.data, allow_pickle=False) as base:
        arrays["subsets"] = np.array([str(s) for s in base["subsets"]] + ["generated"])
        for name, idx in zip(("train", "val", "test"), stratified_split(Y_gen, (0.8, 0.1, 0.1), args.seed)):
            arrays[f"X_{name}"] = np.concatenate([base[f"X_{name}"], X_gen[idx]])
            arrays[f"Y_{name}"] = np.concatenate([base[f"Y_{name}"], Y_gen[idx]])
            arrays[f"S_{name}"] = np.concatenate([base[f"S_{name}"], np.full(len(idx), len(base["subsets"]))])
    with open(os.path.join(args.out, "chess6.npz"), "wb") as f:
        np.savez(f, **arrays)
    print(f"written {args.out}/generated.txt and chess6.npz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
