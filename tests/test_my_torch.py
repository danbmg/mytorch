"""The neural network module: gradients, training, regularisation, config, save files."""

import io
import json
import os
import shutil
import stat
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

from my_torch import ConfigError, Network, NetworkFormatError, TrainingError, load_network, normalize_config, read_header
from my_torch import activations, serialization
from my_torch.network import flush_subnormals
from my_torch.schedules import Schedule

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])


def xor_config(**overrides):
    return {"input_size": 2, "layers": [{"size": 8, "activation": "tanh"}, {"size": 2, "activation": "softmax"}],
            "optimizer": {"name": "adam", "lr": 0.05}, "training": {"epochs": 300, "batch_size": 4}, "seed": 0,
            **overrides}


def numeric_grad(f, x, eps=1e-6):
    """Central finite differences of the scalar function f() with respect to the array x."""
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        plus = f()
        x[i] = old - eps
        grad[i] = (plus - f()) / (2 * eps)
        x[i] = old
    return grad


def rel_error(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12))


class TestGradients(unittest.TestCase):
    """Backpropagation checked against finite differences, in float64."""

    def check(self, hidden, output="softmax", loss="cross_entropy", n_out=3, l2=0.0):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(7, 5))
        Y = rng.integers(3, size=7) if loss == "cross_entropy" else rng.normal(size=(7, n_out))
        net = Network({"input_size": 5, "layers": [{"size": 6, "activation": hidden}, {"size": 4, "activation": hidden},
                                                   {"size": n_out, "activation": output}],
                       "loss": loss, "dtype": "float64", "seed": 7})
        Y = net._targets(Y, len(X))
        total = lambda: net._loss(net._forward(X), Y) + 0.5 * l2 * sum((layer.W ** 2).sum() for layer in net.layers)  # noqa: E731
        net._backward(net._forward(X), Y)
        analytic = [(layer.dW + l2 * layer.W, layer.db) for layer in net.layers]
        for layer, (dW, db) in zip(net.layers, analytic):
            self.assertLess(rel_error(dW, numeric_grad(total, layer.W)), 1e-5)
            self.assertLess(rel_error(db, numeric_grad(total, layer.b)), 1e-5)

    def test_every_hidden_activation_with_softmax_cross_entropy(self):
        for hidden in ("relu", "leaky_relu", "elu", "sigmoid", "tanh", "linear"):
            with self.subTest(hidden=hidden):
                self.check(hidden)

    def test_l2_and_mse(self):
        self.check("tanh", l2=0.1)
        self.check("tanh", output="linear", loss="mse", n_out=2)
        self.check("sigmoid", output="softmax", loss="mse")  # full softmax Jacobian

    def test_activations_stay_finite_on_extreme_values(self):
        z = np.array([[-1000.0, 0.0, 1000.0]])
        for name in activations.ACTIVATIONS:
            self.assertTrue(np.isfinite(activations.forward(name, z, 0.01)).all(), name)
        np.testing.assert_allclose(activations.forward("softmax", z).sum(), 1.0)

    def test_init_keeps_the_variance(self):
        for act, init, expected in (("relu", "he", 2 / 400), ("tanh", "xavier", 2 / 800), ("linear", "lecun", 1 / 400)):
            net = Network({"input_size": 400, "layers": [{"size": 400, "activation": act}], "loss": "mse", "seed": 0})
            self.assertEqual(net.config["layers"][0]["init"], init)  # chosen by "auto"
            self.assertAlmostEqual(net.layers[0].W.var() / expected, 1.0, delta=0.05)


class TestTraining(unittest.TestCase):
    def test_every_optimizer_learns_xor(self):
        for name, lr in {"sgd": 0.5, "momentum": 0.1, "rmsprop": 0.01, "adam": 0.05}.items():
            with self.subTest(optimizer=name):
                net = Network(xor_config(optimizer={"name": name, "lr": lr}, training={"epochs": 500, "batch_size": 4}))
                net.fit(XOR_X, XOR_Y)
                np.testing.assert_array_equal(net.predict(XOR_X), XOR_Y)

    def test_same_seed_same_network_and_resume(self):
        a, b = Network(xor_config()), Network(xor_config())
        a.fit(XOR_X, XOR_Y, epochs=20)
        b.fit(XOR_X, XOR_Y, epochs=20)
        np.testing.assert_array_equal(a.layers[0].W, b.layers[0].W)
        history = a.fit(XOR_X, XOR_Y, epochs=5, batch_size=2)
        self.assertEqual((history["epoch"], a.meta["epochs_trained"], len(a.history["epoch"])), ([21, 22, 23, 24, 25], 25, 25))

    def test_inputs_and_errors(self):
        net = Network(xor_config())
        net.fit(XOR_X.astype(np.uint8), XOR_Y, epochs=2)  # uint8 inputs are converted batch by batch
        for X, Y, message in ((XOR_X[:, :1], XOR_Y, "shape"), (XOR_X, XOR_Y[:3], "targets"), (XOR_X, XOR_Y + 5, "class")):
            with self.assertRaisesRegex(ValueError, message):
                net.fit(X, Y)
        with self.assertRaisesRegex(ValueError, "unknown training setting"):
            net.fit(XOR_X, XOR_Y, epoch=3)
        linear = xor_config(layers=[{"size": 1, "activation": "linear"}], loss="mse", optimizer={"name": "sgd", "lr": 1e6})
        with self.assertRaisesRegex(TrainingError, "diverged"):
            Network(linear).fit(XOR_X * 100, XOR_Y.reshape(-1, 1) * 100, epochs=50)

    def test_verbose_progress(self):
        log = io.StringIO()
        Network(xor_config()).fit(XOR_X, XOR_Y, XOR_X, XOR_Y, epochs=2, verbose=True, log=log)
        self.assertIn("epoch   2/2", log.getvalue())
        self.assertIn("val_acc", log.getvalue())

    def test_schedules(self):
        lr = lambda spec, epoch: Schedule(normalize_config(xor_config(lr_schedule=spec))["lr_schedule"], 1.0, 10).lr(epoch)  # noqa: E731
        self.assertEqual(lr({"name": "step", "step_size": 3, "gamma": 0.5}, 7), 0.25)
        self.assertAlmostEqual(lr({"name": "exponential", "gamma": 0.9}, 2), 0.81)
        self.assertAlmostEqual(lr({"name": "cosine"}, 5), 0.5)
        plateau = Schedule(normalize_config(xor_config(lr_schedule="plateau"))["lr_schedule"], 1.0, 10)
        self.assertEqual([plateau.lr(e, loss) for e, loss in enumerate([None, 1.0, 1.0, 1.0])], [1.0, 1.0, 1.0, 0.5])


class TestRegularisation(unittest.TestCase):
    def test_early_stopping_stops_and_restores_the_best_weights(self):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(40, 2)), rng.integers(2, size=40)  # noise: validation stops improving fast
        net = Network(xor_config(layers=[{"size": 64, "activation": "relu"}, {"size": 2, "activation": "softmax"}],
                                 training={"epochs": 200, "batch_size": 8, "early_stopping": {"patience": 3}}))
        history = net.fit(X[:20], Y[:20], X[20:], Y[20:])
        self.assertTrue(net.meta["stopped_early"])
        self.assertEqual(len(history["epoch"]), net.meta["best_epoch"] + 3)
        self.assertAlmostEqual(net.evaluate(X[20:], Y[20:])["loss"], net.meta["best_val_loss"])

    def test_l2_shrinks_the_weights(self):
        norms = []
        for l2 in (0.0, 0.1):
            net = Network(xor_config(training={"epochs": 200, "batch_size": 4, "l2": l2}))
            net.fit(XOR_X, XOR_Y)
            norms.append(np.abs(net.layers[0].W).sum())
        self.assertLess(norms[1], norms[0])

    def test_inverted_dropout_only_in_training(self):
        net = Network(xor_config(layers=[{"size": 2000, "activation": "relu", "dropout": 0.5},
                                         {"size": 2, "activation": "softmax"}]))
        layer, x = net.layers[0], np.ones((50, 2))
        np.testing.assert_array_equal(layer.forward(x), layer.forward(x))  # inference: no dropout
        layer.forward(x, training=True, rng=np.random.default_rng(0))
        self.assertAlmostEqual((layer.mask == 0).mean(), 0.5, delta=0.02)
        self.assertAlmostEqual(layer.mask.mean(), 1.0, delta=0.02)  # kept units x2: same expected output

    def test_gradient_clipping_and_subnormals(self):
        net = Network(xor_config(training={"epochs": 50, "batch_size": 4, "grad_clip": 1e-3}))
        before = net.layers[0].W.copy()
        net.fit(XOR_X, XOR_Y, epochs=1)
        self.assertLess(np.abs(net.layers[0].W - before).max(), 0.06)  # adam: steps <= lr whatever the gradient
        a = np.array([1e-40, 1.0, -1e-39], dtype=np.float32)
        flush_subnormals([a])
        np.testing.assert_array_equal(a, [0, 1, 0])


class TestConfig(unittest.TestCase):
    def test_defaults_short_forms_and_copy(self):
        config = xor_config(optimizer="rmsprop", lr_schedule="step", training={"early_stopping": True})
        cfg = normalize_config(config)
        self.assertEqual(cfg["optimizer"], {"name": "rmsprop", "lr": 0.001, "rho": 0.9, "eps": 1e-8})
        self.assertEqual(cfg["lr_schedule"], {"name": "step", "step_size": 10, "gamma": 0.5})
        self.assertEqual(cfg["training"]["early_stopping"], {"patience": 5, "min_delta": 0.0, "restore_best": True})
        self.assertEqual(normalize_config(cfg), cfg)
        self.assertEqual(config["optimizer"], "rmsprop")

    def test_errors_name_the_faulty_key(self):
        cases = [
            (xor_config(layers=[{"size": 2, "activation": "relux"}]), "layers[0].activation: unknown 'relux'"),
            (xor_config(layers=[{"size": 0, "activation": "relu"}]), "layers[0].size"),
            (xor_config(layers=[{"size": 2, "activation": "softmax", "dropout": 0.5}]), "output layer cannot use dropout"),
            (xor_config(layers=[{"size": 2, "activation": "relu", "alpha": 1}]), "unknown key(s) alpha"),
            (xor_config(layers=[{"size": 2, "activation": "sigmoid"}]), "'cross_entropy' needs a 'softmax'"),
            (xor_config(optimizer={"name": "adam", "lr": -1}), "optimizer.lr"),
            (xor_config(optimizer={"name": "adam", "beta3": 1}), "unknown key(s) beta3"),
            (xor_config(lr_schedule={"name": "step", "step_size": 0}), "lr_schedule.step_size"),
            (xor_config(training={"batch_size": "big"}), "training.batch_size"),
            (xor_config(training={"early_stopping": {"patience": 0}}), "training.early_stopping.patience"),
            (xor_config(labels=["a"]), "1 labels for an output layer of size 2"),
            ({"layers": []}, "missing 'input_size'"),
            (xor_config(colour="red"), "unknown key(s) colour"),
        ]
        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ConfigError) as ctx:
                    normalize_config(config)
                self.assertIn(message, str(ctx.exception))


class TestSaveFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "net.nn")
        self.net = Network(xor_config(labels=["zero", "one"]))
        self.net.fit(XOR_X, XOR_Y, epochs=30)

    def tearDown(self):
        shutil.rmtree(self.dir)

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
        original = open(self.path, "rb").read()
        Network(xor_config(seed=1)).save(self.path, backup=True)
        self.assertEqual(open(self.path + ".bak", "rb").read(), original)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o640)
        current = open(self.path, "rb").read()
        with mock.patch.object(serialization.np, "savez", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.net.save(self.path)
        self.assertEqual(open(self.path, "rb").read(), current)  # atomic: the old file is intact
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
        for change, message in ((lambda h: h.update(format_version=serialization.FORMAT_VERSION + 1), "newer"),
                                (lambda h: h["config"]["layers"][0].update(activation="nope"), "invalid network config"),
                                (lambda h: h["config"]["layers"][0].update(size=9), "corrupted weights")):
            self.net.save(self.path)
            self.rewrite_header(change)
            with self.subTest(message=message), self.assertRaisesRegex(NetworkFormatError, message):
                load_network(self.path)


if __name__ == "__main__":
    unittest.main()
