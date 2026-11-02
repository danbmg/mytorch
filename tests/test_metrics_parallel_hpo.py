import os
import shutil
import tempfile
import unittest

import numpy as np

from my_torch import hpo, load_network, normalize_config
from my_torch.metrics import balanced_accuracy, confusion_matrix, macro_f1, per_class, report
from my_torch.parallel import train_many
from tests.helpers import XOR_X, XOR_Y, small_config


class TestMetrics(unittest.TestCase):
    def test_confusion_and_per_class(self):
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 1, 1, 1, 0]
        cm = confusion_matrix(y_true, y_pred, 3)
        np.testing.assert_array_equal(cm, [[2, 1, 0], [0, 2, 0], [1, 0, 0]])
        stats = per_class(cm)
        np.testing.assert_allclose(stats["precision"], [2 / 3, 2 / 3, 0])
        np.testing.assert_allclose(stats["recall"], [2 / 3, 1, 0])
        np.testing.assert_allclose(stats["f1"], [2 / 3, 0.8, 0])
        self.assertAlmostEqual(macro_f1(cm), (2 / 3 + 0.8) / 3)
        self.assertAlmostEqual(balanced_accuracy(cm), (2 / 3 + 1 + 0) / 3)
        text = report(cm, ["a", "b", "c"])
        self.assertIn("macro f1", text)
        self.assertIn("confusion matrix", text)


def regression_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    Y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:150], Y[:150], X[150:], Y[150:]


def classifier(seed, lr=0.01):
    return {"input_size": 4, "layers": [{"size": 8, "activation": "relu"}, {"size": 2, "activation": "softmax"}],
            "optimizer": {"name": "adam", "lr": lr}, "training": {"epochs": 5, "batch_size": 32}, "seed": seed}


class TestParallel(unittest.TestCase):
    def test_processes_give_the_same_results_as_in_process(self):
        X, Y, Xv, Yv = regression_data()
        broken = classifier(2)
        broken["layers"][0]["size"] = -1
        configs = [classifier(0), classifier(1), broken]
        seen = []
        here = train_many(configs, X, Y, Xv, Yv, workers=0)
        tmp = tempfile.mkdtemp()
        try:
            paths = [os.path.join(tmp, f"{i}.nn") for i in range(3)]
            spawned = train_many(configs, X, Y, Xv, Yv, workers=2, save_paths=paths, on_result=seen.append)
            self.assertEqual(sorted(r["index"] for r in seen), [0, 1, 2])
            for a, b in zip(here[:2], spawned[:2]):
                self.assertAlmostEqual(a["val_loss"], b["val_loss"], places=5)
                self.assertEqual(a["history"]["epoch"], b["history"]["epoch"])
                self.assertEqual(len(a["confusion"]), 2)
            # a failing training is reported without stopping the others
            self.assertIn("layers[0].size", spawned[2]["error"])
            self.assertAlmostEqual(load_network(paths[0]).evaluate(Xv, Yv)["loss"], spawned[0]["val_loss"], places=5)
        finally:
            shutil.rmtree(tmp)


class TestHpo(unittest.TestCase):
    def setUp(self):
        self.base = normalize_config(small_config())

    def test_set_path(self):
        cfg = {"layers": [{"d": 0}, {"d": 0}, {"d": 0}], "training": None}
        hpo.set_path(cfg, "layers.hidden.d", 0.5)
        hpo.set_path(cfg, "training.l2", 1e-3)
        hpo.set_path(cfg, "layers.*.size", 4)
        hpo.set_path(cfg, "layers.0.size", 9)
        self.assertEqual(cfg["layers"], [{"d": 0.5, "size": 9}, {"d": 0.5, "size": 4}, {"d": 0, "size": 4}])
        self.assertEqual(cfg["training"], {"l2": 1e-3})
        with self.assertRaises(hpo.SpaceError):
            hpo.set_path(cfg, "layers.first.d", 1)

    def test_apply_genes(self):
        cfg = hpo.apply_genes(self.base, {"hidden.layers": 3, "hidden.size": 64, "hidden.ratio": 0.5,
                                          "hidden.activation": "relu", "optimizer.name": "momentum",
                                          "optimizer.lr": 0.2, "training.batch_size": 2})
        normalized = normalize_config(cfg)
        self.assertEqual([layer["size"] for layer in normalized["layers"]], [64, 32, 16, 2])
        self.assertEqual(normalized["layers"][-1]["activation"], "softmax")
        self.assertEqual(normalized["layers"][0]["init"], "he")
        self.assertEqual(normalized["optimizer"], {"name": "momentum", "lr": 0.2, "momentum": 0.9, "nesterov": False})
        self.assertEqual(normalized["training"]["batch_size"], 2)
        self.assertEqual(len(hpo.apply_genes(self.base, {"hidden.layers": 0})["layers"]), 1)

    def test_space_sampling_and_grid(self):
        space = {"a": {"choice": [1, 2]}, "b": {"uniform": [0, 1]}, "c": {"log_uniform": [1e-4, 1e-2]},
                 "d": {"int_uniform": [3, 5]}, "e": {"value": "x"}}
        hpo.check_space(space)
        rng = np.random.default_rng(0)
        for _ in range(50):
            g = hpo.random_genes(space, rng)
            self.assertIn(g["a"], (1, 2))
            self.assertTrue(0 <= g["b"] <= 1 and 1e-4 <= g["c"] <= 1e-2 and 3 <= g["d"] <= 5)
            m = hpo.mutate(g, space, rng, rate=1.0)
            self.assertTrue(0 <= m["b"] <= 1 and 1e-4 <= m["c"] <= 1e-2 and 3 <= m["d"] <= 5)
            self.assertNotEqual(m["a"], g["a"])
        child = hpo.crossover({"a": 1, "b": 2}, {"a": 3, "b": 4}, rng)
        self.assertIn(child["a"], (1, 3))
        self.assertEqual(len(hpo.grid({"x": {"choice": [1, 2, 3]}, "y": {"choice": ["p", "q"]}, "z": {"value": 0}})), 6)
        with self.assertRaises(hpo.SpaceError):
            hpo.grid(space)
        for bad in ({}, {"a": {"pick": [1]}}, {"a": {"uniform": [2, 1]}}, {"a": {"log_uniform": [0, 1]}},
                    {"hidden.depth": {"choice": [1]}}):
            with self.assertRaises(hpo.SpaceError):
                hpo.check_space(bad)

    def fake_evaluate(self, calls):
        """val_loss = distance of the learning rate to 0.003; no training."""
        def evaluate(configs):
            calls.append(len(configs))
            return [{"config": c, "val_loss": abs(np.log10(normalize_config(c)["optimizer"]["lr"] / 0.003)),
                     "val_acc": 0.5} for c in configs]
        return evaluate

    def test_random_search(self):
        calls = []
        space = {"optimizer.lr": {"log_uniform": [1e-4, 1e-1]}}
        results = hpo.random_search(space, self.base, self.fake_evaluate(calls), 20, np.random.default_rng(0))
        self.assertEqual(calls, [20])
        self.assertEqual(len({r["genes"]["optimizer.lr"] for r in results}), 20)

    def test_genetic_search_keeps_its_best_and_skips_known_individuals(self):
        calls = []
        space = {"optimizer.lr": {"log_uniform": [1e-4, 1e-1]}, "hidden.size": {"choice": [4, 8]}}
        best_by_generation = []
        results = hpo.genetic_search(space, self.base, self.fake_evaluate(calls), np.random.default_rng(1),
                                     population=8, generations=6, elite=2,
                                     on_generation=lambda g, ranked: best_by_generation.append(ranked[0]["val_loss"]))
        self.assertEqual(len(best_by_generation), 6)
        # elitism: the best individual is never lost
        self.assertEqual(best_by_generation, sorted(best_by_generation, reverse=True))
        self.assertEqual(calls[0], 8)
        self.assertTrue(all(n <= 6 for n in calls[1:]))  # the 2 elites are not trained again
        self.assertEqual(sum(calls), len(results))

    def test_score(self):
        self.assertEqual(hpo.score({"error": "x"}), float("inf"))
        self.assertEqual(hpo.score({"val_acc": 0.9}, "val_acc"), -0.9)
        self.assertEqual(hpo.best([{"val_acc": 0.5}, {"val_acc": 0.9}], "val_acc"), {"val_acc": 0.9})


class TestXorThroughParallel(unittest.TestCase):
    def test_xor_in_process(self):
        result = train_many([small_config()], XOR_X, XOR_Y, XOR_X, XOR_Y, workers=0)[0]
        self.assertEqual(result["val_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
