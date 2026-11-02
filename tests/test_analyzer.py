"""The chess side: FEN parsing and encoding, dataset helpers, my_torch_analyzer command line."""

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from analyzer import dataset
from analyzer.dataset import ChessFileError, deduplicate, parse_boards, read_boards, stratified_split
from analyzer.fen import FenError, fen_to_vector, parse_fen, position_key, split_line
from analyzer.labels import LABELS
from analyzer.main import main
from my_torch import Network, load_network

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "boards.txt")
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KINGS = "8/8/8/8/8/8/k7/K7 b - - 0 1"


class TestFen(unittest.TestCase):
    def test_encoding(self):
        x = fen_to_vector(START)
        self.assertEqual((x.shape, x.dtype, int(x.sum())), ((773,), np.uint8, 32 + 1 + 4))
        self.assertEqual((x[5 * 64 + 60], x[11 * 64 + 4]), (1, 1))  # white king e1 (row 7, col 4), black king e8
        self.assertEqual(fen_to_vector("r3k3/8/8/8/8/8/8/4K2R b Kq - 0 1")[768:].tolist(), [0, 1, 0, 0, 1])

    def test_invalid_fens_name_the_problem(self):
        for fen, message in (("8/8/8/8/8/8/K6k w - - 0 1", "8 ranks"), ("9/8/8/8/8/8/8/K6k w - - 0 1", "invalid character '9'"),
                             ("8/8/8/8/8/8/8/K6kp w - - 0 1", "9 squares"), ("8/8/8/8/8/8/8/K7 w - - 0 1", "one black king"),
                             ("8/8/8/8/8/8/8/K6k x - - 0 1", "side to move"), ("8/8/8/8/8/8/8/K6k w KK - 0 1", "castling"),
                             ("8/8/8/8/8/8/8/K6k w - e4 0 1", "en passant"), ("8/8/8/8/8/8/8/K6k w - - -1 1", "halfmove"),
                             ("8/8/8/8/8/8/8/K6k w -", "6 fields")):
            with self.subTest(fen=fen), self.assertRaisesRegex(FenError, message):
                parse_fen(fen)

    def test_lines_and_keys(self):
        self.assertEqual(split_line(START + " Check White\r"), (START, "Check White"))
        self.assertEqual(split_line("8/8/8/8/8/8/k7/K7 b - - Nothing"), ("8/8/8/8/8/8/k7/K7 b - -", "Nothing"))
        self.assertEqual(split_line(START), (START, None))
        self.assertEqual(position_key("8/8/8/8/8/8/k7/K7 b Kq - 3 40"), position_key("44/8/8/8/8/8/k7/K7 b qK - 0 1"))


class TestDataset(unittest.TestCase):
    def test_fixture_chunks_and_labels(self):
        X, y = read_boards(FIXTURE, LABELS, require_labels=True)
        with mock.patch.object(dataset, "ENCODE_CHUNK", 5):
            X5, _ = read_boards(FIXTURE, LABELS)
        np.testing.assert_array_equal(X, X5)
        self.assertEqual((X.shape, y[0]), ((24, 773), 1))
        lines = [KINGS + " check  white", "  ", KINGS, KINGS + " Stalemate"]
        self.assertEqual(parse_boards(lines, labels=LABELS)[1].tolist(), [1, -1, -1])
        for bad, message in ((lines[:3], "f.txt:3: no expected output"), (lines[3:], "f.txt:1: unknown label 'Stalemate'")):
            with self.assertRaisesRegex(ChessFileError, message):
                parse_boards(bad, "f.txt", LABELS, require_labels=True)
        with self.assertRaisesRegex(ChessFileError, "no chessboard"):
            parse_boards(["", " "])

    def test_dedup_and_stratified_split(self):
        kept, duplicates, conflicts = deduplicate(["a", "b", "a", "c", "b", "a"], [0, 1, 0, 2, 3, 0])
        self.assertEqual((kept.tolist(), duplicates, conflicts), ([0, 3], 3, 1))
        groups = np.repeat([0, 1, 2], [100, 50, 10])
        parts = stratified_split(groups, (0.8, 0.1, 0.1), seed=0)
        self.assertEqual(sorted(np.concatenate(parts).tolist()), list(range(160)))
        self.assertEqual([np.bincount(groups[p]).tolist() for p in parts], [[80, 40, 8], [10, 5, 1], [10, 5, 1]])
        np.testing.assert_array_equal(parts[0], stratified_split(groups, (0.8, 0.1, 0.1), seed=0)[0])


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

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return main(list(args)), out.getvalue(), err.getvalue()

    def test_errors_go_to_stderr_with_84(self):
        self.write("bad.txt", [KINGS, "", "8/8/8/8/8/8/k7/K7 x - - 0 1"])
        self.write("unlabelled.txt", [KINGS + " Nothing", KINGS])
        Network(self.config(input_size=10)).save("small.nn")
        before = open("net.nn", "rb").read()
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
                code, out, err = self.run_main(*args)
                self.assertEqual((code, out), (84, ""))
                self.assertTrue(err.startswith("my_torch_analyzer: ") and message in err, err)
        self.assertEqual(open("net.nn", "rb").read(), before)
        self.assertIn("LOADFILE CHESSFILE", self.run_main("--help")[1])

    def test_predict_one_label_per_board_in_order(self):
        code, out, err = self.run_main("--predict", "net.nn", "boards.txt")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.splitlines(), [LABELS[i] for i in load_network("net.nn").predict(read_boards("boards.txt")[0])])
        Network(self.config(labels=LABELS + ["Stalemate"])).save("six.nn")  # the labels come from the network
        self.assertTrue(set(self.run_main("--predict", "six.nn", "boards.txt")[1].splitlines()) <= set(LABELS + ["Stalemate"]))

    def test_train_save_backup_and_reproducibility(self):
        before = open("net.nn", "rb").read()
        code, out, err = self.run_main("--train", "--save", "a.nn", "net.nn", "boards.txt")
        self.assertEqual((code, out, open("net.nn", "rb").read()), (0, "", before))
        self.assertIn("epoch   3/3", err)
        self.assertEqual(load_network("a.nn").meta["epochs_trained"], 3)
        code, _, err = self.run_main("--train", "--epochs", "2", "net.nn", "boards.txt")
        self.assertIn("previous version kept as net.nn.bak", err)
        self.assertEqual(open("net.nn.bak", "rb").read(), before)
        with open(FIXTURE) as f:
            self.write("more.txt", f.read().splitlines() * 5)  # 120 boards: 10% held out for validation
        for name in ("b.nn", "c.nn"):
            code, _, err = self.run_main("--train", "--save", name, "a.nn", "more.txt")
        self.assertIn("training on 108, validating on 12", err)
        for (wb, _), (wc, _) in zip(load_network("b.nn").get_weights(), load_network("c.nn").get_weights()):
            np.testing.assert_array_equal(wb, wc)


if __name__ == "__main__":
    unittest.main()
