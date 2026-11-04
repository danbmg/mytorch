"""The legal move generator: perft reference counts, status of known positions, agreement with the dataset (T2.9)."""

import unittest

from analyzer.fen import split_line
from bonus.movegen import Board, legal_moves, status
from helpers import FIXTURE


class TestMovegen(unittest.TestCase):
    def test_number_of_legal_moves_of_reference_positions(self):
        # the usual "perft" test positions, depth 1 (chessprogramming.org/Perft_Results)
        cases = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 20),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 48),  # castling, pins
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 14),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6),  # in check, promotions
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 44),
            ("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1", 6),  # en passant would expose the king: illegal
        ]
        for fen, expected in cases:
            with self.subTest(fen=fen):
                self.assertEqual(len(list(legal_moves(Board.from_fen(fen)))), expected)

    def test_status(self):
        cases = [
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "Stalemate"),
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "Checkmate White"),
            ("7k/6Q1/8/8/8/8/8/6K1 b - - 0 1", "Check White"),
            ("6k1/8/8/8/8/8/5PPP/r5K1 w - - 0 1", "Checkmate Black"),
            ("8/8/8/8/8/8/8/K1k5 w - - 0 1", "Nothing"),
        ]
        for fen, expected in cases:
            with self.subTest(fen=fen):
                self.assertEqual(status(fen), expected)

    def test_agrees_with_the_dataset(self):
        with open(FIXTURE) as f:
            for line in f.read().splitlines():
                fen, label = split_line(line)
                self.assertEqual(status(fen), label, fen)

    def test_fen_round_trip(self):
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b Kq e3 0 1"
        self.assertEqual(Board.from_fen(fen).to_fen(), fen)


if __name__ == "__main__":
    unittest.main()
