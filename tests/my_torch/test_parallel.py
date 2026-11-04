"""Parallel trainings: worker processes give the results of the in-process run, a failed
training does not stop the others (T2.3)."""

import unittest

import numpy as np

from helpers import xor_config
from my_torch.parallel import train_many


class TestParallel(unittest.TestCase):
    def test_processes_give_the_same_results_as_in_process(self):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(200, 2)), (rng.normal(size=200) > 0).astype(int)
        broken = xor_config(layers=[{"size": -1, "activation": "relu"}, {"size": 2, "activation": "softmax"}])
        configs = [xor_config(seed=0, training={"epochs": 5}), xor_config(seed=1, training={"epochs": 5}), broken]
        seen = []
        here = train_many(configs, X, Y, X, Y, workers=0)
        spawned = train_many(configs, X, Y, X, Y, workers=2, on_result=seen.append)
        self.assertEqual(sorted(r["index"] for r in seen), [0, 1, 2])
        for a, b in zip(here[:2], spawned[:2]):
            self.assertAlmostEqual(a["val_loss"], b["val_loss"], places=5)
        self.assertIn("layers[0].size", spawned[2]["error"])  # a failed training does not stop the others


if __name__ == "__main__":
    unittest.main()
