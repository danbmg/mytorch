"""The .nn file: exact round trip, binary weights, atomic write with backup, unusable files refused (T1.5)."""

import json
import os
import shutil
import stat
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

from helpers import XOR_X, XOR_Y, xor_config
from my_torch import Network, NetworkFormatError, load_network, read_header
from my_torch.serialization import FORMAT_VERSION, writer


class TestSaveFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "net.nn")
        self.net = Network(xor_config(labels=["zero", "one"]))
        self.net.fit(XOR_X, XOR_Y, epochs=30)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def read(self, path):
        with open(path, "rb") as f:
            return f.read()

    def rewrite_header(self, change):
        with np.load(self.path) as data:
            arrays = {k: data[k] for k in data.files}
        header = json.loads(bytes(arrays["header"]).decode())
        change(header)
        arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
        with open(self.path, "wb") as f:
            np.savez(f, **arrays)

    def test_round_trip_exact_name_and_binary_weights(self):
        self.net.save(self.path)
        self.assertEqual(os.listdir(self.dir), ["net.nn"])  # no x.nn.npz, no temporary file left
        loaded = load_network(self.path)
        np.testing.assert_array_equal(loaded.predict_proba(XOR_X), self.net.predict_proba(XOR_X))
        self.assertEqual((loaded.labels, loaded.meta["epochs_trained"], loaded.config), (["zero", "one"], 30, self.net.config))
        header = read_header(self.path)
        self.assertEqual([layer["activation"] for layer in header["config"]["layers"]], ["tanh", "softmax"])
        with zipfile.ZipFile(self.path) as z:
            self.assertEqual(sorted(z.namelist()), ["W0.npy", "W1.npy", "b0.npy", "b1.npy", "header.npy"])
            with self.assertRaises(UnicodeDecodeError):
                z.read("W0.npy").decode("utf-8")  # weights are binary, not text

    def test_backup_permissions_and_failed_save(self):
        self.net.save(self.path)
        os.chmod(self.path, 0o640)
        original = self.read(self.path)
        Network(xor_config(seed=1)).save(self.path, backup=True)
        self.assertEqual(self.read(self.path + ".bak"), original)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o640)
        current = self.read(self.path)
        with mock.patch.object(writer.np, "savez", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.net.save(self.path)
        self.assertEqual(self.read(self.path), current)  # atomic: the old file is intact
        self.assertEqual(sorted(os.listdir(self.dir)), ["net.nn", "net.nn.bak"])

    def test_unusable_files(self):
        for name, content in (("empty.nn", b""), ("text.nn", b"8/8/8/8/8/8/k7/K7 b - - 0 1\n")):
            with open(os.path.join(self.dir, name), "wb") as f:
                f.write(content)
        other = os.path.join(self.dir, "other.nn")
        with open(other, "wb") as f:
            np.savez(f, a=np.zeros(3))
        for path in ("missing.nn", self.dir, "empty.nn", "text.nn", other):
            with self.subTest(path=path), self.assertRaises(NetworkFormatError):
                load_network(os.path.join(self.dir, path))
        for change, message in ((lambda h: h.update(format_version=FORMAT_VERSION + 1), "newer"),
                                (lambda h: h["config"]["layers"][0].update(activation="nope"), "invalid network config"),
                                (lambda h: h["config"]["layers"][0].update(size=9), "corrupted weights")):
            self.net.save(self.path)
            self.rewrite_header(change)
            with self.subTest(message=message), self.assertRaisesRegex(NetworkFormatError, message):
                load_network(self.path)


if __name__ == "__main__":
    unittest.main()
