"""Symmetries of chess: the transformed encoding equals the encoding of the transformed
FEN, each symmetry is its own inverse, the labels follow, and the averaged prediction
gives a board and its mirror the same answer."""

import unittest

import numpy as np

from analyzer.augment import (SYMMETRIES, augment, flip_label, symmetric_proba, transform, transform_fen,
                              usable_symmetries)
from analyzer.fen import fen_to_vector, parse_line
from analyzer.labels import LABELS
from helpers import FIXTURE
from my_torch import Network

CASTLING_AND_EN_PASSANT = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b Kq e3 0 1"


def fixture_fens():
    with open(FIXTURE) as f:
        return [CASTLING_AND_EN_PASSANT] + [parse_line(line)[0] for line in f.read().splitlines()]


class TestAugment(unittest.TestCase):
    def test_transformed_encoding_matches_the_transformed_fen(self):
        fens = fixture_fens()
        X = np.stack([fen_to_vector(fen) for fen in fens])
        for symmetry in SYMMETRIES:
            with self.subTest(symmetry=symmetry):
                expected = np.stack([fen_to_vector(transform_fen(fen, symmetry)) for fen in fens])
                np.testing.assert_array_equal(transform(X, symmetry), expected)
                np.testing.assert_array_equal(transform(transform(X, symmetry), symmetry), X)

    def test_transformed_fens(self):
        self.assertEqual(transform_fen(CASTLING_AND_EN_PASSANT, "colors"),
                         "r3k2r/pppbbppp/2n2q1P/1P2p3/3pn3/BN2PNP1/P1PPQPB1/R3K2R w Qk e6 0 1")
        self.assertEqual(transform_fen(CASTLING_AND_EN_PASSANT, "files"),
                         "r2k3r/1bpqpp1p/1pnp2nb/3NP3/3P2p1/p1Q2N2/PPPBBPPP/R2K3R b Qk d3 0 1")

    def test_labels_follow_the_symmetries(self):
        self.assertEqual([flip_label(label) for label in LABELS],
                         ["Nothing", "Check Black", "Check White", "Checkmate Black", "Checkmate White"])
        X = np.zeros((6, 773), dtype=np.uint8)
        y = np.array([0, 1, 2, 3, 4, -1])
        X4, y4 = augment(X, y, LABELS, ["colors", "files", "colors+files"])
        self.assertEqual(X4.shape, (24, 773))
        self.assertEqual(y4.tolist(), [0, 1, 2, 3, 4, -1, 0, 2, 1, 4, 3, -1, 0, 1, 2, 3, 4, -1, 0, 2, 1, 4, 3, -1])

    def test_averaged_prediction_is_the_same_for_a_board_and_its_mirror(self):
        net = Network({"input_size": 773, "layers": [{"size": 16, "activation": "relu"},
                                                     {"size": 5, "activation": "softmax"}], "seed": 1})
        X = np.stack([fen_to_vector(fen) for fen in fixture_fens()])
        proba = symmetric_proba(net, X, LABELS)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
        mirrored = symmetric_proba(net, transform(X, "colors"), LABELS)
        swap = [LABELS.index(flip_label(label)) for label in LABELS]
        np.testing.assert_allclose(mirrored[:, swap], proba, rtol=1e-5)

    def test_labels_without_their_color_swapped_counterpart_keep_the_colors(self):
        self.assertEqual(usable_symmetries(LABELS + ["Stalemate"]), SYMMETRIES)
        self.assertEqual(usable_symmetries(["Nothing", "Check White"]), ("identity", "files"))


if __name__ == "__main__":
    unittest.main()
