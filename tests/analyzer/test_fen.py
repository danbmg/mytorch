"""FEN parsing, validation and encoding into the 773 inputs (T1.8)."""

import unittest

import numpy as np

from analyzer.fen import FenError, fen_to_vector, parse_fen, position_key, split_line

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TestFen(unittest.TestCase):
    def test_encoding(self):
        x = fen_to_vector(START)
        self.assertEqual((x.shape, x.dtype, int(x.sum())), ((773,), np.uint8, 32 + 1 + 4))
        self.assertEqual((x[5 * 64 + 60], x[11 * 64 + 4]), (1, 1))  # white king e1 (row 7, col 4), black king e8
        self.assertEqual(fen_to_vector("r3k3/8/8/8/8/8/8/4K2R b Kq - 0 1")[768:].tolist(), [0, 1, 0, 0, 1])

    def test_invalid_fens_name_the_problem(self):
        for fen, message in (("8/8/8/8/8/8/K6k w - - 0 1", "8 ranks"), ("9/8/8/8/8/8/8/K6k w - - 0 1", "invalid character '9'"),
                             ("8/8/8/8/8/8/8/K6kp w - - 0 1", "9 squares"), ("8/8/8/8/8/8/8/K7 w - - 0 1", "one black king"),
                             ("8/8/8/8/8/8/8/K6k x - - 0 1", "side to move"), ("8/8/8/8/8/8/8/K6k w KK - 0 1", "castling"),
                             ("8/8/8/8/8/8/8/K6k w - e4 0 1", "en passant"), ("8/8/8/8/8/8/8/K6k w - - -1 1", "halfmove"),
                             ("8/8/8/8/8/8/8/K6k w -", "6 fields")):
            with self.subTest(fen=fen), self.assertRaisesRegex(FenError, message):
                parse_fen(fen)

    def test_lines_and_keys(self):
        self.assertEqual(split_line(START + " Check White\r"), (START, "Check White"))
        self.assertEqual(split_line("8/8/8/8/8/8/k7/K7 b - - Nothing"), ("8/8/8/8/8/8/k7/K7 b - -", "Nothing"))
        self.assertEqual(split_line(START), (START, None))
        self.assertEqual(position_key("8/8/8/8/8/8/k7/K7 b Kq - 3 40"), position_key("44/8/8/8/8/8/k7/K7 b qK - 0 1"))


if __name__ == "__main__":
    unittest.main()
