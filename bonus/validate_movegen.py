#!/usr/bin/env python3
"""Check bonus/movegen.py against the dataset: it must find every label.

usage: python3 bonus/validate_movegen.py [DATASET_DIR] [--workers N]
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analyzer.fen import split_line  # noqa: E402
from bonus.movegen import status  # noqa: E402


def check(line):
    fen, label = split_line(line)
    return None if status(fen) == label else f"{line}  -> movegen says {status(fen)}"


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("dataset", nargs="?", default=os.path.join(ROOT, "dataset"))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    t0, total, wrong = time.perf_counter(), 0, []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for path in sorted(glob.glob(os.path.join(args.dataset, "*", "*.txt"))):
            with open(path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            errors = [r for r in pool.imap(check, lines, chunksize=2000) if r]
            total, wrong = total + len(lines), wrong + errors
            print(f"  {os.path.relpath(path, args.dataset):<28} {len(lines):>7} positions, {len(errors)} wrong",
                  file=sys.stderr)
    print("\n".join(wrong[:20]))
    print(f"{total - len(wrong)} / {total} labels found in {time.perf_counter() - t0:.0f}s")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
