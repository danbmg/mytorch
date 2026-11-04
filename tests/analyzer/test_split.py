"""Deduplication and seeded stratified split of the dataset (T0.3)."""

import unittest

import numpy as np

from analyzer.split import deduplicate, stratified_split


class TestSplit(unittest.TestCase):
    def test_dedup_and_stratified_split(self):
        kept, duplicates, conflicts = deduplicate(["a", "b", "a", "c", "b", "a"], [0, 1, 0, 2, 3, 0])
        self.assertEqual((kept.tolist(), duplicates, conflicts), ([0, 3], 3, 1))
        groups = np.repeat([0, 1, 2], [100, 50, 10])
        parts = stratified_split(groups, (0.8, 0.1, 0.1), seed=0)
        self.assertEqual(sorted(np.concatenate(parts).tolist()), list(range(160)))
        self.assertEqual([np.bincount(groups[p]).tolist() for p in parts], [[80, 40, 8], [10, 5, 1], [10, 5, 1]])
        np.testing.assert_array_equal(parts[0], stratified_split(groups, (0.8, 0.1, 0.1), seed=0)[0])


if __name__ == "__main__":
    unittest.main()
