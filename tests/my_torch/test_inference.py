"""Prediction on 0/1 inputs: adding the active rows of the weights gives the matrix
product, empty rows and all, and predict_proba() takes that path only for 0/1 values."""

import unittest

import numpy as np

from my_torch import Network
from my_torch.layers import sum_active_rows


def binary_network():
    return Network({"input_size": 40, "layers": [{"size": 16, "activation": "relu"},
                                                 {"size": 3, "activation": "softmax"}], "seed": 2})


class TestInference(unittest.TestCase):
    def test_sum_of_active_rows_is_the_matrix_product(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(40, 7)).astype(np.float32)
        for density in (0.0, 0.05, 0.5, 1.0):
            with self.subTest(density=density):
                X = (rng.random((50, 40)) < density).astype(np.uint8)
                X[3] = 0  # a row without any active input
                np.testing.assert_allclose(sum_active_rows(X, W), X.astype(np.float32) @ W, rtol=1e-5, atol=1e-6)
        self.assertEqual(sum_active_rows(np.zeros((0, 40), dtype=np.uint8), W).shape, (0, 7))

    def test_predict_proba_gives_the_outputs_of_forward(self):
        net = binary_network()
        rng = np.random.default_rng(1)
        X = (rng.random((30, 40)) < 0.2).astype(np.uint8)
        np.testing.assert_allclose(net.predict_proba(X), net.forward(X), rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(net.predict_proba(X.astype(bool)), net.forward(X), rtol=1e-5, atol=1e-7)
        counts = X * 3  # uint8 but not 0/1: the matrix product
        np.testing.assert_allclose(net.predict_proba(counts), net.forward(counts), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
