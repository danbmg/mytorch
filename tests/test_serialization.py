import json
import os
import shutil
import stat
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

from my_torch import Network, NetworkFormatError, load_network, read_header
from my_torch import serialization
from tests.helpers import XOR_X, XOR_Y, small_config


def rewrite_header(path, change):
    """Rewrite the JSON header of a saved network through `change(header)`."""
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    header = json.loads(bytes(arrays["header"]).decode())
    change(header)
    arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    with open(path, "wb") as f:
        np.savez(f, **arrays)


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "net.nn")
        self.net = Network(small_config(labels=["zero", "one"],
                                        layers=[{"size": 8, "activation": "tanh", "dropout": 0.1},
                                                {"size": 2, "activation": "softmax"}]))
        self.net.fit(XOR_X, XOR_Y, XOR_X, XOR_Y, epochs=30)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_round_trip_gives_identical_predictions(self):
        self.net.save(self.path)
        loaded = Network.load(self.path)
        np.testing.assert_array_equal(loaded.predict_proba(XOR_X), self.net.predict_proba(XOR_X))
        self.assertEqual(loaded.config, self.net.config)
        self.assertEqual(loaded.labels, ["zero", "one"])
        self.assertEqual(loaded.meta, self.net.meta)
        self.assertEqual(loaded.history, self.net.history)
        # training can go on after loading
        loaded.fit(XOR_X, XOR_Y, epochs=2)
        self.assertEqual(loaded.meta["epochs_trained"], 32)

    def test_exact_file_name_and_no_leftover(self):
        self.net.save(self.path)
        self.assertEqual(os.listdir(self.dir), ["net.nn"])

    def test_content_is_inspectable_but_not_all_text(self):
        self.net.save(self.path)
        with zipfile.ZipFile(self.path) as z:
            self.assertEqual(sorted(z.namelist()), ["W0.npy", "W1.npy", "b0.npy", "b1.npy", "header.npy"])
        header = read_header(self.path)
        self.assertEqual(header["format_version"], serialization.FORMAT_VERSION)
        self.assertEqual([layer["activation"] for layer in header["config"]["layers"]], ["tanh", "softmax"])
        self.assertEqual(header["config"]["optimizer"]["lr"], 0.05)
        self.assertEqual(header["layers"][0]["params"], 2 * 8 + 8)
        with np.load(self.path) as data:
            self.assertEqual(data["W0"].dtype, np.float32)

    def test_backup_keeps_the_previous_network(self):
        self.net.save(self.path)
        before = self.net.predict_proba(XOR_X)
        self.net.fit(XOR_X, XOR_Y, epochs=20)
        self.net.save(self.path, backup=True)
        np.testing.assert_array_equal(load_network(self.path + ".bak").predict_proba(XOR_X), before)
        np.testing.assert_array_equal(load_network(self.path).predict_proba(XOR_X), self.net.predict_proba(XOR_X))

    def test_failed_save_leaves_the_old_file_intact(self):
        self.net.save(self.path)
        with open(self.path, "rb") as f:
            original = f.read()
        with mock.patch.object(serialization.np, "savez", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.net.save(self.path)
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), original)
        self.assertEqual(os.listdir(self.dir), ["net.nn"])

    def test_permissions_of_the_replaced_file_are_kept(self):
        self.net.save(self.path)
        os.chmod(self.path, 0o640)
        self.net.save(self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o640)

    def test_unusable_files(self):
        empty = os.path.join(self.dir, "empty.nn")
        open(empty, "w").close()
        text = os.path.join(self.dir, "text.nn")
        with open(text, "w") as f:
            f.write("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")
        npy = os.path.join(self.dir, "array.nn")
        with open(npy, "wb") as f:
            np.save(f, np.zeros(3))
        other_npz = os.path.join(self.dir, "other.nn")
        with open(other_npz, "wb") as f:
            np.savez(f, a=np.zeros(3))
        self.net.save(self.path)
        truncated = os.path.join(self.dir, "truncated.nn")
        with open(self.path, "rb") as src, open(truncated, "wb") as dst:
            dst.write(src.read()[:500])
        for path in (os.path.join(self.dir, "missing.nn"), self.dir, empty, text, npy, other_npz, truncated):
            with self.subTest(path=os.path.basename(path)):
                with self.assertRaises(NetworkFormatError):
                    load_network(path)

    def test_newer_format_version_is_refused(self):
        self.net.save(self.path)
        rewrite_header(self.path, lambda h: h.update(format_version=serialization.FORMAT_VERSION + 1))
        with self.assertRaisesRegex(NetworkFormatError, "newer"):
            load_network(self.path)

    def test_weights_that_do_not_match_the_config(self):
        self.net.save(self.path)
        rewrite_header(self.path, lambda h: h["config"]["layers"][0].update(size=9))
        with self.assertRaises(NetworkFormatError):
            load_network(self.path)

    def test_invalid_config_in_header(self):
        self.net.save(self.path)
        rewrite_header(self.path, lambda h: h["config"]["layers"][0].update(activation="nope"))
        with self.assertRaisesRegex(NetworkFormatError, "invalid network config"):
            load_network(self.path)


if __name__ == "__main__":
    unittest.main()
