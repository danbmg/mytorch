"""Generated stalemate data: random positions are legal and labelled like the move generator says (T2.9)."""

import random
import unittest

from bonus.gen_stalemate import classify, place
from bonus.movegen import status


class TestGenStalemate(unittest.TestCase):
    def test_generated_positions_are_legal_and_correctly_labelled(self):
        rng = random.Random(0)
        boards = [b for b in (place(rng) for _ in range(300)) if b is not None]
        self.assertGreater(len(boards), 100)
        for board in boards:
            self.assertEqual(classify(board)[1], status(board.to_fen()))


if __name__ == "__main__":
    unittest.main()
