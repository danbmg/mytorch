"""Hyperparameter search: dotted paths, architecture genes, search spaces, random and genetic searches (T2.4)."""

import unittest

import numpy as np

from helpers import xor_config
from my_torch import hpo, normalize_config


class TestHpo(unittest.TestCase):
    base = normalize_config(xor_config())

    def test_set_path_and_apply_genes(self):
        cfg = {"layers": [{"d": 0}, {"d": 0}, {"d": 0}], "training": None}
        for path, value in (("layers.hidden.d", 0.5), ("training.l2", 1e-3), ("layers.*.size", 4), ("layers.0.size", 9)):
            hpo.set_path(cfg, path, value)
        self.assertEqual(cfg, {"layers": [{"d": 0.5, "size": 9}, {"d": 0.5, "size": 4}, {"d": 0, "size": 4}],
                               "training": {"l2": 1e-3}})
        with self.assertRaises(hpo.SpaceError):
            hpo.set_path(cfg, "layers.first.d", 1)
        cfg = normalize_config(hpo.apply_genes(self.base, {"hidden.layers": 3, "hidden.size": 64, "hidden.ratio": 0.5,
                                                           "hidden.activation": "relu", "optimizer.name": "momentum"}))
        self.assertEqual([(layer["size"], layer["init"]) for layer in cfg["layers"]], [(64, "he"), (32, "he"), (16, "he"), (2, "xavier")])
        self.assertEqual(cfg["optimizer"], {"name": "momentum", "lr": 0.05, "momentum": 0.9})

    def test_space(self):
        space = {"a": {"choice": [1, 2]}, "b": {"uniform": [0, 1]}, "c": {"log_uniform": [1e-4, 1e-2]}, "e": {"value": "x"}}
        hpo.check_space(space)
        rng = np.random.default_rng(0)
        for _ in range(50):
            g = {key: hpo.sample(spec, rng) for key, spec in space.items()}
            m = hpo.mutate(g, space, rng, rate=1.0)
            self.assertTrue(all(0 <= x["b"] <= 1 and 1e-4 <= x["c"] <= 1e-2 for x in (g, m)))
            self.assertNotEqual(m["a"], g["a"])
        self.assertEqual(len(hpo.grid({"x": {"choice": [1, 2, 3]}, "y": {"choice": ["p", "q"]}, "z": {"value": 0}})), 6)
        for bad in ({}, {"a": {"pick": [1]}}, {"a": {"uniform": [2, 1]}}, {"a": {"log_uniform": [0, 1]}}, {"hidden.depth": {"value": 1}}):
            with self.subTest(space=bad), self.assertRaises(hpo.SpaceError):
                hpo.check_space(bad)

    def test_random_and_genetic_searches(self):
        calls = []

        def evaluate(configs):  # val_loss = distance of the learning rate to 0.003, no training
            calls.append(len(configs))
            return [{"config": c, "val_loss": abs(np.log10(normalize_config(c)["optimizer"]["lr"] / 0.003))} for c in configs]

        space = {"optimizer.lr": {"log_uniform": [1e-4, 1e-1]}, "hidden.size": {"choice": [4, 8]}}
        self.assertEqual(len(hpo.random_search(space, self.base, evaluate, 20, np.random.default_rng(0))), 20)
        calls.clear()
        best = []
        results = hpo.genetic_search(space, self.base, evaluate, np.random.default_rng(1), population=8, generations=6,
                                     on_generation=lambda g, ranked: best.append(ranked[0]["val_loss"]))
        self.assertEqual(best, sorted(best, reverse=True))  # elitism: the best is never lost
        self.assertEqual((len(best), calls[0], sum(calls)), (6, 8, len(results)))
        self.assertTrue(all(n <= 6 for n in calls[1:]))  # the 2 elites are not trained again
        self.assertEqual(hpo.score({"error": "x"}), float("inf"))


if __name__ == "__main__":
    unittest.main()
