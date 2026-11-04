"""Chess files: every line encoded, labels resolved, errors naming the file and the line (T1.8, T1.10)."""

import unittest
from unittest import mock

import numpy as np

from analyzer import boards
from analyzer.boards import ChessFileError, parse_boards, read_boards
from analyzer.labels import LABELS
from helpers import FIXTURE

KINGS = "8/8/8/8/8/8/k7/K7 b - - 0 1"


class TestBoards(unittest.TestCase):
    def test_fixture_chunks_and_labels(self):
        X, y = read_boards(FIXTURE, LABELS, require_labels=True)
        with mock.patch.object(boards, "ENCODE_CHUNK", 5):
            X5, _ = read_boards(FIXTURE, LABELS)
        np.testing.assert_array_equal(X, X5)
        self.assertEqual((X.shape, y[0]), ((24, 773), 1))
        lines = [KINGS + " check  white", "  ", KINGS, KINGS + " Stalemate"]
        self.assertEqual(parse_boards(lines, labels=LABELS)[1].tolist(), [1, -1, -1])
        for bad, message in ((lines[:3], "f.txt:3: no expected output"), (lines[3:], "f.txt:1: unknown label 'Stalemate'")):
            with self.assertRaisesRegex(ChessFileError, message):
                parse_boards(bad, "f.txt", LABELS, require_labels=True)
        with self.assertRaisesRegex(ChessFileError, "no chessboard"):
            parse_boards(["", " "])


if __name__ == "__main__":
    unittest.main()
