"""my_torch_generator: reproducible blank networks, errors on stderr with exit 84, help (T1.7)."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from generator.main import main
from helpers import XOR_X, XOR_Y, run_main, xor_config
from my_torch import load_network


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

    def test_generates_reproducible_networks(self):
        self.assertEqual(run_main(main, "basic.conf", "3"), (0, "", ""))
        nets = [load_network(f"basic_{i}.nn") for i in (1, 2, 3)]
        self.assertEqual([n.seed for n in nets], [10, 11, 12])
        first = nets[0].layers[0].W
        run_main(main, "basic.conf", "1")
        np.testing.assert_array_equal(load_network("basic_1.nn").layers[0].W, first)
        self.assertEqual(nets[0].predict(XOR_X).shape, XOR_Y.shape)

    def test_errors_and_help(self):
        for args, message in (((), "pairs"), (("basic.conf", "0"), "positive integer"), (("missing.conf", "1"), "No such file"),
                              (("bad.conf", "1"), "invalid JSON at line 1"), (("basic.conf", "1", "bad.conf", "1"), "bad.conf")):
            with self.subTest(args=args):
                code, out, err = run_main(main, *args)
                self.assertEqual((code, out), (84, ""))
                self.assertIn(message, err)
        self.assertEqual([f for f in os.listdir() if f.endswith(".nn")], [])  # every file is checked before writing
        self.assertIn("config_file_1 nb_1", run_main(main, "--help")[1])


if __name__ == "__main__":
    unittest.main()
