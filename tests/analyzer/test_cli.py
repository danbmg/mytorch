"""my_torch_analyzer command line: errors on stderr with exit 84 and nothing on stdout,
predict in order, train with --save or in place with a backup, reproducible (T1.9, T1.10, T2.7)."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from analyzer.boards import read_boards
from analyzer.labels import LABELS
from analyzer.main import main
from helpers import FIXTURE, run_main
from my_torch import Network, load_network

KINGS = "8/8/8/8/8/8/k7/K7 b - - 0 1"


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.old_cwd, self.dir = os.getcwd(), tempfile.mkdtemp()
        os.chdir(self.dir)
        shutil.copy(FIXTURE, "boards.txt")
        Network(self.config()).save("net.nn")

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.dir)

    def config(self, labels=LABELS, input_size=773):
        return {"input_size": input_size, "layers": [{"size": 16, "activation": "relu"}, {"size": len(labels), "activation": "softmax"}],
                "optimizer": {"name": "adam", "lr": 0.01}, "training": {"epochs": 3, "batch_size": 8}, "labels": labels, "seed": 3}

    def write(self, name, lines):
        with open(name, "w") as f:
            f.write("\n".join(lines) + "\n")

    def read(self, name):
        with open(name, "rb") as f:
            return f.read()

    def test_errors_go_to_stderr_with_84(self):
        self.write("bad.txt", [KINGS, "", "8/8/8/8/8/8/k7/K7 x - - 0 1"])
        self.write("unlabelled.txt", [KINGS + " Nothing", KINGS])
        Network(self.config(input_size=10)).save("small.nn")
        before = self.read("net.nn")
        cases = [((), "choose a mode"),
                 (("--predict", "--train", "net.nn", "boards.txt"), "not both"),
                 (("--predict", "--save", "x.nn", "net.nn", "boards.txt"), "--save only works with --train"),
                 (("--train", "net.nn", "boards.txt", "--save"), "--save needs a value"),
                 (("--train", "--epochs", "0", "net.nn", "boards.txt"), "positive integer"),
                 (("--predict", "--verbose", "net.nn", "boards.txt"), "unknown option"),
                 (("--predict", "net.nn"), "got 1 file argument"),
                 (("--predict", "missing.nn", "boards.txt"), "no such file"),
                 (("--predict", "boards.txt", "boards.txt"), "not a my_torch network file"),
                 (("--predict", "small.nn", "boards.txt"), "takes 10 inputs"),
                 (("--predict", "net.nn", "missing.txt"), "No such file"),
                 (("--predict", "net.nn", "bad.txt"), "bad.txt:3: side to move"),
                 (("--train", "net.nn", "unlabelled.txt"), "unlabelled.txt:2: no expected output"),
                 (("--train", "--save", "nodir/out.nn", "net.nn", "boards.txt"), "cannot save")]
        for args, message in cases:
            with self.subTest(args=args):
                code, out, err = run_main(main, *args)
                self.assertEqual((code, out), (84, ""))
                self.assertTrue(err.startswith("my_torch_analyzer: ") and message in err, err)
        self.assertEqual(self.read("net.nn"), before)
        self.assertIn("LOADFILE CHESSFILE", run_main(main, "--help")[1])

    def test_predict_one_label_per_board_in_order(self):
        code, out, err = run_main(main, "--predict", "net.nn", "boards.txt")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.splitlines(), [LABELS[i] for i in load_network("net.nn").predict(read_boards("boards.txt")[0])])
        Network(self.config(labels=LABELS + ["Stalemate"])).save("six.nn")  # the labels come from the network
        self.assertTrue(set(run_main(main, "--predict", "six.nn", "boards.txt")[1].splitlines()) <= set(LABELS + ["Stalemate"]))

    def test_train_save_backup_and_reproducibility(self):
        before = self.read("net.nn")
        code, out, err = run_main(main, "--train", "--save", "a.nn", "net.nn", "boards.txt")
        self.assertEqual((code, out, self.read("net.nn")), (0, "", before))
        self.assertIn("epoch   3/3", err)
        self.assertEqual(load_network("a.nn").meta["epochs_trained"], 3)
        code, _, err = run_main(main, "--train", "--epochs", "2", "net.nn", "boards.txt")
        self.assertIn("previous version kept as net.nn.bak", err)
        self.assertEqual(self.read("net.nn.bak"), before)
        with open(FIXTURE) as f:
            self.write("more.txt", f.read().splitlines() * 5)  # 120 boards: 10% held out for validation
        for name in ("b.nn", "c.nn"):
            code, _, err = run_main(main, "--train", "--save", name, "a.nn", "more.txt")
        self.assertIn("training on 108, validating on 12", err)
        for (wb, _), (wc, _) in zip(load_network("b.nn").get_weights(), load_network("c.nn").get_weights()):
            np.testing.assert_array_equal(wb, wc)


if __name__ == "__main__":
    unittest.main()
