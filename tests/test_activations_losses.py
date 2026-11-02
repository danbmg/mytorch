import unittest

import numpy as np

from my_torch.activations import ACTIVATIONS, get_activation
from my_torch.losses import get_loss
from tests.helpers import numeric_grad, rel_error


def away_from_zero(shape, rng):
    """Values with |z| > 0.1: finite differences are wrong on the kink of ReLU at 0."""
    z = rng.uniform(0.1, 3, size=shape)
    return z * rng.choice([-1, 1], size=shape)


class TestActivations(unittest.TestCase):
    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        for name in ACTIVATIONS:
            with self.subTest(activation=name):
                act = get_activation(name)
                z = away_from_zero((4, 5), rng)
                upstream = rng.normal(size=(4, 5))
                a = act.forward(z)
                analytic = act.backward(z, a, upstream)
                numeric = numeric_grad(lambda: float((act.forward(z) * upstream).sum()), z)
                self.assertLess(rel_error(analytic, numeric), 1e-6)

    def test_known_values(self):
        z = np.array([[-2.0, 0.0, 3.0]])
        np.testing.assert_allclose(get_activation("relu").forward(z), [[0, 0, 3]])
        np.testing.assert_allclose(get_activation("leaky_relu", alpha=0.1).forward(z), [[-0.2, 0, 3]])
        np.testing.assert_allclose(get_activation("sigmoid").forward(z), 1 / (1 + np.exp(-z)))
        np.testing.assert_allclose(get_activation("tanh").forward(z), np.tanh(z))
        np.testing.assert_allclose(get_activation("linear").forward(z), z)
        np.testing.assert_allclose(get_activation("elu").forward(z), [[np.exp(-2) - 1, 0, 3]])

    def test_softmax_rows_are_probabilities_even_for_extreme_logits(self):
        z = np.array([[1000.0, -1000.0, 0.0], [-1e4, -1e4, -1e4], [3.0, 1.0, 0.2]], dtype=np.float32)
        p = get_activation("softmax").forward(z)
        self.assertTrue(np.all(np.isfinite(p)))
        np.testing.assert_allclose(p.sum(axis=1), 1, rtol=1e-6)

    def test_sigmoid_extreme_values_stay_finite(self):
        z = np.array([[-1e4, 1e4]], dtype=np.float32)
        a = get_activation("sigmoid").forward(z)
        np.testing.assert_allclose(a, [[0, 1]])

    def test_unknown_activation(self):
        with self.assertRaises(ValueError):
            get_activation("heaviside")


class TestLosses(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.logits = rng.normal(size=(6, 4))
        self.y = np.eye(4)[rng.integers(4, size=6)]
        self.softmax = get_activation("softmax")

    def test_cross_entropy_fused_gradient(self):
        loss = get_loss("cross_entropy")

        def value():
            return loss.value(self.softmax.forward(self.logits), self.y, logits=self.logits)

        out = self.softmax.forward(self.logits)
        self.assertLess(rel_error(loss.fused_grad(out, self.y), numeric_grad(value, self.logits)), 1e-6)

    def test_cross_entropy_gradient_wrt_probabilities(self):
        loss = get_loss("cross_entropy")
        p = self.softmax.forward(self.logits)
        numeric = numeric_grad(lambda: loss.value(p, self.y), p)
        self.assertLess(rel_error(loss.grad(p, self.y), numeric), 1e-6)

    def test_cross_entropy_no_nan_on_extreme_logits(self):
        loss = get_loss("cross_entropy")
        logits = np.array([[1e4, -1e4, 0], [-1e4, 1e4, 0]], dtype=np.float32)
        y = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
        out = self.softmax.forward(logits)
        for v in (loss.value(out, y, logits=logits), loss.value(out, y)):
            self.assertTrue(np.isfinite(v))
        self.assertAlmostEqual(loss.value(out, y, logits=logits), 1e4, delta=1)
        self.assertTrue(np.all(np.isfinite(loss.fused_grad(out, y))))

    def test_binary_cross_entropy(self):
        loss = get_loss("binary_cross_entropy")
        sigmoid = get_activation("sigmoid")
        y = (self.logits > 0.3).astype(float)
        numeric = numeric_grad(lambda: loss.value(sigmoid.forward(self.logits), y, logits=self.logits),
                               self.logits)
        self.assertLess(rel_error(loss.fused_grad(sigmoid.forward(self.logits), y), numeric), 1e-6)
        p = sigmoid.forward(self.logits)
        self.assertLess(rel_error(loss.grad(p, y), numeric_grad(lambda: loss.value(p, y), p)), 1e-6)
        extreme = np.array([[1e4, -1e4]])
        self.assertTrue(np.isfinite(loss.value(sigmoid.forward(extreme), np.array([[0.0, 1.0]]), logits=extreme)))

    def test_mse(self):
        loss = get_loss("mse")
        out = self.logits.copy()
        self.assertAlmostEqual(loss.value(out, self.y), float(np.mean((out - self.y) ** 2)))
        self.assertLess(rel_error(loss.grad(out, self.y), numeric_grad(lambda: loss.value(out, self.y), out)), 1e-6)


if __name__ == "__main__":
    unittest.main()
