"""Metrics, parallel trainings, hyperparameter search and my_torch_generator."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from generator.main import main as generator
from my_torch import hpo, load_network, normalize_config
from my_torch.metrics import confusion_matrix, macro_f1, per_class, report
from my_torch.parallel import train_many
from tests.test_my_torch import XOR_X, XOR_Y, xor_config


class TestMetrics(unittest.TestCase):
    def test_confusion_and_per_class(self):
        cm = confusion_matrix([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 1, 0], 3)
        np.testing.assert_array_equal(cm, [[2, 1, 0], [0, 2, 0], [1, 0, 0]])
        stats = per_class(cm)
        np.testing.assert_allclose(stats["precision"], [2 / 3, 2 / 3, 0])
        np.testing.assert_allclose(stats["recall"], [2 / 3, 1, 0])
        self.assertAlmostEqual(macro_f1(cm), (2 / 3 + 0.8 + 0) / 3)
        self.assertIn("accuracy", report(cm, ["a", "b", "c"]))


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


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.old_cwd, self.dir = os.getcwd(), tempfile.mkdtemp()
        os.chdir(self.dir)
        for name, content in (("basic.conf", json.dumps(xor_config(seed=10))), ("bad.conf", '{"input_size": 2,')):
            with open(name, "w") as f:
                f.write(content)

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.dir)

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return generator(list(args)), out.getvalue(), err.getvalue()

    def test_generates_reproducible_networks(self):
        self.assertEqual(self.run_main("basic.conf", "3"), (0, "", ""))
        nets = [load_network(f"basic_{i}.nn") for i in (1, 2, 3)]
        self.assertEqual([n.seed for n in nets], [10, 11, 12])
        first = nets[0].layers[0].W
        self.run_main("basic.conf", "1")
        np.testing.assert_array_equal(load_network("basic_1.nn").layers[0].W, first)
        self.assertEqual(nets[0].predict(XOR_X).shape, XOR_Y.shape)

    def test_errors_and_help(self):
        for args, message in (((), "pairs"), (("basic.conf", "0"), "positive integer"), (("missing.conf", "1"), "No such file"),
                              (("bad.conf", "1"), "invalid JSON at line 1"), (("basic.conf", "1", "bad.conf", "1"), "bad.conf")):
            with self.subTest(args=args):
                code, out, err = self.run_main(*args)
                self.assertEqual((code, out), (84, ""))
                self.assertIn(message, err)
        self.assertEqual([f for f in os.listdir() if f.endswith(".nn")], [])  # every file is checked before writing
        self.assertIn("config_file_1 nb_1", self.run_main("--help")[1])


if __name__ == "__main__":
    unittest.main()
