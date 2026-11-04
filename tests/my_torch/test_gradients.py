"""Backpropagation checked against finite differences, in float64 (T1.1, T1.2, T1.3)."""

import unittest

import numpy as np

from helpers import numeric_grad, rel_error
from my_torch import Network, activations
from my_torch.data import targets


class TestGradients(unittest.TestCase):
    def check(self, hidden, output="softmax", loss="cross_entropy", n_out=3, l2=0.0):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(7, 5))
        Y = rng.integers(3, size=7) if loss == "cross_entropy" else rng.normal(size=(7, n_out))
        net = Network({"input_size": 5, "layers": [{"size": 6, "activation": hidden}, {"size": 4, "activation": hidden},
                                                   {"size": n_out, "activation": output}],
                       "loss": loss, "dtype": "float64", "seed": 7})
        Y = targets(net, Y, len(X))
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


if __name__ == "__main__":
    unittest.main()
