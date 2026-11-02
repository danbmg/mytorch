import io
import unittest

import numpy as np

from my_torch import Network, TrainingError
from my_torch.init import init_weights
from my_torch.network import _clip_global_norm, flush_subnormals
from my_torch.schedules import make_schedule
from tests.helpers import XOR_X, XOR_Y, numeric_grad, rel_error, small_config


def network_gradient_check(config, X, Y, l2=0.0):
    """Largest relative error between backprop and finite differences, over all weights."""
    net = Network(config)
    Yt = net._targets(Y, len(X))

    def total_loss():
        out = net._forward(X)
        penalty = 0.5 * l2 * sum(float((layer.W ** 2).sum()) for layer in net.layers)
        return net._loss(out, Yt) + penalty

    out = net._forward(X)
    net._backward(out, Yt)
    analytic = net._grads(l2)
    errors = []
    for k, param in enumerate(net._params()):
        errors.append(rel_error(analytic[k], numeric_grad(total_loss, param)))
    return max(errors)


class TestGradients(unittest.TestCase):
    """Backpropagation checked against finite differences, in float64 (DoD T1.1: < 1e-5)."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(7, 5))
        self.Y = rng.integers(3, size=7)

    def config(self, hidden, output="softmax", loss="cross_entropy", n_out=3, dropout=0.0):
        return {
            "input_size": 5,
            "layers": [{"size": 6, "activation": hidden, "dropout": dropout},
                       {"size": 4, "activation": hidden},
                       {"size": n_out, "activation": output}],
            "loss": loss,
            "dtype": "float64",
            "seed": 7,
        }

    def test_every_hidden_activation_with_softmax_cross_entropy(self):
        for hidden in ("relu", "leaky_relu", "elu", "sigmoid", "tanh", "linear"):
            with self.subTest(hidden=hidden):
                self.assertLess(network_gradient_check(self.config(hidden), self.X, self.Y), 1e-5)

    def test_with_l2(self):
        self.assertLess(network_gradient_check(self.config("tanh"), self.X, self.Y, l2=0.1), 1e-5)

    def test_mse_with_linear_output(self):
        Y = np.random.default_rng(4).normal(size=(7, 2))
        cfg = self.config("tanh", output="linear", loss="mse", n_out=2)
        self.assertLess(network_gradient_check(cfg, self.X, Y), 1e-5)

    def test_mse_with_softmax_output_uses_the_full_softmax_jacobian(self):
        cfg = self.config("tanh", output="softmax", loss="mse")
        self.assertLess(network_gradient_check(cfg, self.X, self.Y), 1e-5)

    def test_binary_cross_entropy_with_sigmoid(self):
        Y = (np.random.default_rng(5).random((7, 2)) > 0.5).astype(float)
        cfg = self.config("tanh", output="sigmoid", loss="binary_cross_entropy", n_out=2)
        self.assertLess(network_gradient_check(cfg, self.X, Y), 1e-5)


class TestXor(unittest.TestCase):
    """DoD T1.4: every optimiser learns XOR."""

    def test_all_optimizers_converge(self):
        settings = {"sgd": 0.5, "momentum": 0.1, "rmsprop": 0.01, "adam": 0.05}
        for name, lr in settings.items():
            with self.subTest(optimizer=name):
                net = Network(small_config(optimizer={"name": name, "lr": lr},
                                           training={"epochs": 500, "batch_size": 4}))
                net.fit(XOR_X, XOR_Y)
                np.testing.assert_array_equal(net.predict(XOR_X), XOR_Y)

    def test_nesterov_momentum(self):
        net = Network(small_config(optimizer={"name": "momentum", "lr": 0.1, "nesterov": True},
                                   training={"epochs": 500, "batch_size": 4}))
        net.fit(XOR_X, XOR_Y)
        np.testing.assert_array_equal(net.predict(XOR_X), XOR_Y)

    def test_single_sigmoid_output(self):
        cfg = small_config(layers=[{"size": 8, "activation": "tanh"}, {"size": 1, "activation": "sigmoid"}],
                           loss="binary_cross_entropy")
        net = Network(cfg)
        net.fit(XOR_X, XOR_Y)
        np.testing.assert_array_equal(net.predict(XOR_X), XOR_Y)


class TestTraining(unittest.TestCase):
    def test_same_seed_same_network(self):
        a, b = Network(small_config()), Network(small_config())
        a.fit(XOR_X, XOR_Y, epochs=20)
        b.fit(XOR_X, XOR_Y, epochs=20)
        for (wa, ba), (wb, bb) in zip(a.get_weights(), b.get_weights()):
            np.testing.assert_array_equal(wa, wb)
            np.testing.assert_array_equal(ba, bb)
        c = Network(small_config(seed=1))
        self.assertFalse(np.array_equal(a.layers[0].W, c.layers[0].W))

    def test_overrides_batch_size_epochs_and_history(self):
        net = Network(small_config(optimizer={"name": "sgd", "lr": 0.3}))
        history = net.fit(XOR_X, XOR_Y, epochs=3, batch_size=2)
        self.assertEqual(history["epoch"], [1, 2, 3])
        self.assertEqual(history["lr"], [0.3, 0.3, 0.3])
        net.fit(XOR_X, XOR_Y, epochs=2)
        self.assertEqual(net.meta["epochs_trained"], 5)
        self.assertEqual(net.history["epoch"], [1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            net.fit(XOR_X, XOR_Y, epochz=3)

    def test_input_and_target_checks(self):
        net = Network(small_config())
        with self.assertRaises(ValueError):
            net.predict(np.zeros((3, 5)))
        with self.assertRaises(ValueError):
            net.fit(XOR_X, XOR_Y[:3])
        with self.assertRaises(ValueError):
            net.fit(XOR_X, np.array([0, 1, 2, 0]))
        self.assertEqual(net.predict(np.zeros((0, 2))).shape, (0,))

    def test_divergence_is_reported(self):
        cfg = {"input_size": 2, "layers": [{"size": 1, "activation": "linear"}], "loss": "mse",
               "optimizer": {"name": "sgd", "lr": 1e6}, "training": {"epochs": 50, "batch_size": 4}, "seed": 0}
        with self.assertRaises(TrainingError):
            Network(cfg).fit(XOR_X * 100, XOR_Y.reshape(-1, 1) * 100)

    def test_verbose_progress(self):
        log = io.StringIO()
        net = Network(small_config())
        net.fit(XOR_X, XOR_Y, XOR_X, XOR_Y, epochs=2, verbose=True, log=log)
        lines = log.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("val_acc", lines[0])

    def test_uint8_inputs_are_accepted(self):
        net = Network(small_config())
        net.fit(XOR_X.astype(np.uint8), XOR_Y, epochs=2)
        self.assertEqual(net.predict(XOR_X.astype(np.uint8)).shape, (4,))


class TestRegularisation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # random labels: nothing to learn, the validation loss can only get worse
        self.X, self.Y = rng.normal(size=(64, 10)), rng.integers(2, size=64)
        self.Xv, self.Yv = rng.normal(size=(64, 10)), rng.integers(2, size=64)

    def config(self, **training):
        return {"input_size": 10,
                "layers": [{"size": 64, "activation": "relu", "dropout": 0.3}, {"size": 2, "activation": "softmax"}],
                "optimizer": {"name": "adam", "lr": 0.01}, "training": {"epochs": 200, "batch_size": 16, **training},
                "seed": 0}

    def test_early_stopping_stops_and_restores_the_best_weights(self):
        net = Network(self.config(early_stopping={"patience": 3}))
        history = net.fit(self.X, self.Y, self.Xv, self.Yv)
        self.assertLess(len(history["epoch"]), 200)
        self.assertTrue(net.meta["stopped_early"])
        best = int(np.argmin(history["val_loss"]))
        self.assertEqual(net.meta["best_epoch"], history["epoch"][best])
        self.assertAlmostEqual(net.evaluate(self.Xv, self.Yv)["loss"], history["val_loss"][best], places=5)

    def test_early_stopping_without_validation_is_disabled(self):
        log = io.StringIO()
        net = Network(self.config(early_stopping=True))
        history = net.fit(self.X, self.Y, epochs=5, verbose=True, log=log)
        self.assertEqual(len(history["epoch"]), 5)
        self.assertIn("disabled", log.getvalue())

    def test_l2_shrinks_the_weights(self):
        norms = []
        for l2 in (0.0, 0.05):
            net = Network(self.config(l2=l2))
            net.fit(self.X, self.Y, epochs=50)
            norms.append(float((net.layers[0].W ** 2).sum()))
        self.assertLess(norms[1], norms[0] / 2)

    def test_dropout_only_in_training(self):
        net = Network(self.config())
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(net.forward(self.X), net.forward(self.X))
        self.assertFalse(np.array_equal(net.forward(self.X, training=True, rng=rng), net.forward(self.X)))

    def test_inverted_dropout_keeps_the_expected_activation(self):
        net = Network(self.config())
        layer = net.layers[0]
        x = np.ones((20000, 10), dtype=np.float32)
        kept = layer.forward(x, training=True, rng=np.random.default_rng(1))
        plain = layer.forward(x)
        self.assertAlmostEqual(float(kept.mean()), float(plain.mean()), delta=0.02 * float(plain.mean()))

    def test_mc_dropout(self):
        net = Network(self.config())
        mean, std = net.predict_mc(self.X, n_samples=20)
        self.assertEqual(mean.shape, (64, 2))
        np.testing.assert_allclose(mean.sum(axis=1), 1, rtol=1e-5)
        self.assertGreater(float(std.mean()), 0)
        cfg = self.config()
        cfg["layers"][0]["dropout"] = 0.0
        _, std = Network(cfg).predict_mc(self.X, n_samples=5)
        self.assertEqual(float(std.max()), 0.0)

    def test_gradient_clipping(self):
        grads = [np.full(4, 3.0), np.full(3, 4.0)]
        norm = _clip_global_norm(grads, 1.0)
        self.assertAlmostEqual(norm, float(np.sqrt(4 * 9 + 3 * 16)))
        self.assertAlmostEqual(float(np.sqrt(sum((g * g).sum() for g in grads))), 1.0, places=6)
        small = [np.full(2, 0.1)]
        _clip_global_norm(small, 1.0)
        np.testing.assert_array_equal(small[0], [0.1, 0.1])

    def test_flush_subnormals(self):
        a = np.array([1e-40, -1e-41, 1e-30, 0.5, 0.0], dtype=np.float32)
        flush_subnormals([a])
        np.testing.assert_array_equal(a, np.array([0, 0, 1e-30, 0.5, 0], dtype=np.float32))


class TestSchedules(unittest.TestCase):
    def test_values(self):
        step = make_schedule({"name": "step", "step_size": 2, "gamma": 0.5}, 1.0)
        self.assertEqual([step.lr(e) for e in range(5)], [1, 1, 0.5, 0.5, 0.25])
        exp = make_schedule({"name": "exponential", "gamma": 0.9}, 1.0)
        self.assertAlmostEqual(exp.lr(3), 0.729)
        cos = make_schedule({"name": "cosine", "lr_min": 0.0}, 1.0, epochs=10)
        self.assertAlmostEqual(cos.lr(0), 1.0)
        self.assertAlmostEqual(cos.lr(5), 0.5)
        self.assertAlmostEqual(cos.lr(10), 0.0)
        const = make_schedule({"name": "constant"}, 0.1)
        self.assertEqual(const.lr(100), 0.1)

    def test_plateau(self):
        plateau = make_schedule({"name": "plateau", "factor": 0.5, "patience": 2}, 1.0)
        lrs = [plateau.lr(e, loss) for e, loss in enumerate([None, 1.0, 0.9, 0.95, 0.95, 0.95, 0.95])]
        self.assertEqual(lrs, [1, 1, 1, 1, 0.5, 0.5, 0.25])

    def test_schedule_is_applied_during_training(self):
        net = Network(small_config(lr_schedule={"name": "step", "step_size": 1, "gamma": 0.5}))
        history = net.fit(XOR_X, XOR_Y, epochs=3)
        self.assertEqual(history["lr"], [0.05, 0.025, 0.0125])


class TestInit(unittest.TestCase):
    """DoD T1.3: weight variance matches each formula."""

    def test_variances(self):
        rng = np.random.default_rng(0)
        n_in, n_out = 1000, 500
        expected = {"he": 2 / n_in, "xavier": 2 / (n_in + n_out), "lecun": 1 / n_in, "uniform": 0.05 ** 2 / 3}
        for name, var in expected.items():
            with self.subTest(init=name):
                W = init_weights(name, n_in, n_out, rng)
                self.assertAlmostEqual(float(W.var()), var, delta=0.03 * var)
                self.assertAlmostEqual(float(W.mean()), 0.0, delta=0.01)

    def test_selection_by_config(self):
        cfg = {"input_size": 4, "layers": [{"size": 3, "activation": "relu"}, {"size": 3, "activation": "tanh"},
                                           {"size": 2, "activation": "linear", "init": "uniform"}], "loss": "mse"}
        net = Network(cfg)
        self.assertEqual([layer["init"] for layer in net.config["layers"]], ["he", "xavier", "uniform"])
        cfg["init"] = "lecun"
        self.assertEqual([layer["init"] for layer in Network(cfg).config["layers"]], ["lecun", "lecun", "uniform"])
        self.assertTrue(np.all(Network(cfg).layers[0].b == 0))


if __name__ == "__main__":
    unittest.main()
