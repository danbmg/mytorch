import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from generator.main import main
from my_torch import load_network
from tests.helpers import small_config


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.dir = tempfile.mkdtemp()
        os.chdir(self.dir)
        self.write("basic_network.conf", small_config(seed=10))

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.dir)

    def write(self, name, content):
        with open(name, "w") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_generates_loadable_networks(self):
        code, out, err = self.run_main("basic_network.conf", "3")
        self.assertEqual((code, out, err), (0, "", ""))
        names = sorted(f for f in os.listdir() if f.endswith(".nn"))
        self.assertEqual(names, ["basic_network_1.nn", "basic_network_2.nn", "basic_network_3.nn"])
        nets = [load_network(n) for n in names]
        self.assertEqual([n.seed for n in nets], [10, 11, 12])
        self.assertFalse(np.array_equal(nets[0].layers[0].W, nets[1].layers[0].W))
        self.assertEqual(nets[0].meta["epochs_trained"], 0)

    def test_several_configs_and_reproducibility(self):
        self.write("other.conf", small_config(seed=None))
        self.assertEqual(self.run_main("basic_network.conf", "1", "other.conf", "2")[0], 0)
        first = load_network("basic_network_1.nn").layers[0].W
        others = [load_network(f"other_{i}.nn") for i in (1, 2)]
        self.assertNotEqual(others[0].seed, others[1].seed)
        self.run_main("basic_network.conf", "1")
        np.testing.assert_array_equal(load_network("basic_network_1.nn").layers[0].W, first)

    def test_errors(self):
        self.write("bad_json.conf", '{"input_size": 2,')
        self.write("bad_config.conf", small_config(layers=[{"size": 2, "activation": "nope"}]))
        cases = [
            ((), "pairs"),
            (("basic_network.conf",), "pairs"),
            (("basic_network.conf", "0"), "positive integer"),
            (("basic_network.conf", "two"), "positive integer"),
            (("missing.conf", "1"), "no such file"),
            (("bad_json.conf", "1"), "invalid JSON at line 1"),
            (("bad_config.conf", "1"), "layers[0].activation"),
            # the first pair is valid but nothing is written: everything is checked first
            (("basic_network.conf", "1", "bad_config.conf", "1"), "bad_config.conf"),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                code, out, err = self.run_main(*args)
                self.assertEqual(code, 84)
                self.assertEqual(out, "")
                self.assertIn(message, err)
        self.assertEqual([f for f in os.listdir() if f.endswith(".nn")], [])

    def test_help(self):
        code, out, err = self.run_main("--help")
        self.assertEqual(code, 0)
        self.assertIn("USAGE", out)
        self.assertIn("config_file_1 nb_1", out)


if __name__ == "__main__":
    unittest.main()
