"""Confusion matrix, per-class precision / recall / F1, macro F1 and the text report (T2.5)."""

import unittest

import numpy as np

from my_torch.metrics import confusion_matrix, macro_f1, per_class, report


class TestMetrics(unittest.TestCase):
    def test_confusion_and_per_class(self):
        cm = confusion_matrix([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 1, 0], 3)
        np.testing.assert_array_equal(cm, [[2, 1, 0], [0, 2, 0], [1, 0, 0]])
        stats = per_class(cm)
        np.testing.assert_allclose(stats["precision"], [2 / 3, 2 / 3, 0])
        np.testing.assert_allclose(stats["recall"], [2 / 3, 1, 0])
        self.assertAlmostEqual(macro_f1(cm), (2 / 3 + 0.8 + 0) / 3)
        self.assertIn("accuracy", report(cm, ["a", "b", "c"]))


if __name__ == "__main__":
    unittest.main()
