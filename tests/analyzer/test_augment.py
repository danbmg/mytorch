"""Colour inversion: the encoded mirror equals the encoding of the mirrored FEN, the labels
swap colours, and mirroring twice gives the original back."""

import unittest

import numpy as np

from analyzer.augment import flip_colors, flip_fen, flip_label
from analyzer.fen import fen_to_vector, parse_line
from analyzer.labels import LABELS
from helpers import FIXTURE


class TestAugment(unittest.TestCase):
    def test_mirror_matches_the_mirrored_fen(self):
        with open(FIXTURE) as f:
            lines = f.read().splitlines()
        fens = ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b Kq e3 0 1"] + [parse_line(l)[0] for l in lines]
        X = np.stack([fen_to_vector(fen) for fen in fens])
        flipped, _ = flip_colors(X, np.zeros(len(fens), dtype=int), LABELS)
        np.testing.assert_array_equal(flipped, np.stack([fen_to_vector(flip_fen(fen)) for fen in fens]))
        np.testing.assert_array_equal(flip_colors(flipped, np.zeros(len(fens), dtype=int), LABELS)[0], X)
        self.assertEqual(flip_fen(fens[0]), "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R w Qk e6 0 1")

    def test_labels_swap_colours(self):
        self.assertEqual([flip_label(label) for label in LABELS],
                         ["Nothing", "Check Black", "Check White", "Checkmate Black", "Checkmate White"])
        _, y = flip_colors(np.zeros((6, 773), dtype=np.uint8), np.array([0, 1, 2, 3, 4, -1]), LABELS)
        self.assertEqual(y.tolist(), [0, 2, 1, 4, 3, -1])


if __name__ == "__main__":
    unittest.main()
