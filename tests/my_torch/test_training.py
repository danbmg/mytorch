"""Mini-batch training: every optimizer learns XOR, same seed gives the same network,
training resumes, bad inputs are refused, the learning rate schedules (T1.4, T2.2)."""

import io
import unittest

import numpy as np

from helpers import XOR_X, XOR_Y, xor_config
from my_torch import Network, TrainingError, normalize_config
from my_torch.schedules import Schedule


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


if __name__ == "__main__":
    unittest.main()
