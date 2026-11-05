"""Regularization: early stopping, L2, inverted dropout, gradient clipping, subnormal flush (T2.1, T2.2)."""

import unittest

import numpy as np

from helpers import XOR_X, XOR_Y, xor_config
from my_torch import Network
from my_torch.training.batch import flush_subnormals


class TestRegularization(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
