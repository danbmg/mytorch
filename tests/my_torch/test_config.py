"""Network configuration: defaults, short forms, and errors that name the faulty key (T1.5)."""

import unittest

from helpers import xor_config
from my_torch import ConfigError, normalize_config


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
            (xor_config(color="red"), "unknown key(s) color"),
        ]
        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ConfigError) as ctx:
                    normalize_config(config)
                self.assertIn(message, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
