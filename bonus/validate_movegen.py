#!/usr/bin/env python3
"""Check bonus/movegen.py against the dataset: it must find every label.

usage: python3 bonus/validate_movegen.py [DATASET_DIR] [--workers N] [--symmetries]

--symmetries also checks the 3 transformed versions of every position
(analyzer/augment.py): colors swapped, files mirrored, both. Each one must get
the label the augmentation gives it, which proves that the 4x larger training
set and the averaged predictions only use correct labels.
"""

import argparse
import functools
import glob
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analyzer.augment import SYMMETRIES, flip_label, transform_fen  # noqa: E402
from analyzer.fen import split_line  # noqa: E402
from bonus.movegen import status  # noqa: E402


def check(line, symmetries):
    """None if the move generator finds the label of every version of the position, else the first error."""
    fen, label = split_line(line)
    for symmetry in symmetries:
        version = transform_fen(fen, symmetry)
        expected = flip_label(label) if "colors" in symmetry else label
        found = status(version)
        if found != expected:
            return f"{line}  -> {symmetry}: {version} is {found}, expected {expected}"
    return None


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("dataset", nargs="?", default=os.path.join(ROOT, "dataset"))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--symmetries", action="store_true", help="also check the 3 transformed versions")
    args = p.parse_args(argv)
    symmetries = SYMMETRIES if args.symmetries else ("identity",)
    work = functools.partial(check, symmetries=symmetries)
    start = time.perf_counter()
    total = 0
    wrong = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for path in sorted(glob.glob(os.path.join(args.dataset, "*", "*.txt"))):
            with open(path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            errors = [error for error in pool.imap(work, lines, chunksize=2000) if error]
            total += len(lines)
            wrong += errors
            print(f"  {os.path.relpath(path, args.dataset):<28} {len(lines):>7} positions, {len(errors)} wrong",
                  file=sys.stderr)
    if total == 0:
        print(f"validate_movegen.py: no */*.txt file in {args.dataset}", file=sys.stderr)
        return 84
    print("\n".join(wrong[:20]))
    versions = f" x {len(symmetries)} versions" if len(symmetries) > 1 else ""
    print(f"{total - len(wrong)} / {total} positions{versions} with every label found"
          f" ({time.perf_counter() - start:.0f}s)")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
