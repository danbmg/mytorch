#!/usr/bin/env python3
"""Deduplicate the chess dataset, split it 80/10/10 and encode it (T0.3).

usage: python3 scripts/dedup_split.py [DATASET_DIR] [OUT_DIR] [--seed 42]   (default: dataset/ data/)

- duplicates = same board, side, castling and en passant (the move counters do not
  matter), removed BEFORE the split, else a position can be in train AND test;
- split stratified by source file: every class and subset (10, 20, many pieces)
  keeps its proportion in each part;
- writes OUT/{train,val,test}.txt (usable by my_torch_analyzer) and OUT/chess.npz,
  the encoded dataset of the scripts: X_* (N, 773) uint8, Y_* class, S_* subset.
"""

import argparse
import glob
import os
import sys

from common import ROOT, fail

from analyzer.boards import ChessFileError, parse_boards, read_lines
from analyzer.split import deduplicate, stratified_split
from analyzer.fen import parse_fen, position_key, split_line
from analyzer.labels import LABELS, label_index

import numpy as np


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("dataset", nargs="?", default=os.path.join(ROOT, "dataset"))
    p.add_argument("out", nargs="?", default=os.path.join(ROOT, "data"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.dataset, "*", "*.txt")))
    if not paths:
        fail(f"{args.dataset}: no <class>/<subset>.txt file")
    subsets = sorted({os.path.splitext(os.path.basename(path))[0] for path in paths})
    lines, keys, y, source = [], [], [], []
    for file_index, path in enumerate(paths):
        try:
            file_lines = read_lines(path)
        except ChessFileError as e:
            fail(str(e))
        for lineno, line in enumerate(file_lines, 1):
            if not line.strip():
                continue
            try:
                fen, label = split_line(line)
                parse_fen(fen)
                y.append(label_index(label or "", LABELS))
            except ValueError as e:
                fail(f"{path}:{lineno}: {e}")
            lines.append(line.strip())
            keys.append(position_key(fen))
            source.append(file_index)
    y, source = np.array(y), np.array(source)
    subset_of = np.array([subsets.index(os.path.splitext(os.path.basename(path))[0]) for path in paths])
    kept, duplicates, conflicts = deduplicate(keys, y)
    print(f"{len(lines)} lines, {duplicates} duplicates, {conflicts} conflicting positions -> {len(kept)} kept",
          file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    arrays = {"labels": np.array(LABELS), "subsets": np.array(subsets)}
    for name, idx in zip(("train", "val", "test"), stratified_split(source[kept], (0.8, 0.1, 0.1), args.seed)):
        rows = kept[idx]
        part = [lines[i] for i in rows]
        with open(os.path.join(args.out, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(part) + "\n")
        arrays[f"X_{name}"], arrays[f"Y_{name}"] = parse_boards(part, name, LABELS, require_labels=True)
        arrays[f"S_{name}"] = subset_of[source[rows]]
        print(f"  {name:<5} {len(rows):>7} boards: " + ", ".join(
            f"{label} {c}" for label, c in zip(LABELS, np.bincount(arrays[f'Y_{name}'], minlength=len(LABELS)))),
            file=sys.stderr)
    with open(os.path.join(args.out, "chess.npz"), "wb") as f:
        np.savez(f, **arrays)  # uncompressed: loads in about a second (compressed: 10x smaller, 10x slower)
    print(f"written {args.out}/{{train,val,test}}.txt and {args.out}/chess.npz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
