import copy
import unittest

from my_torch import ConfigError, normalize_config
from tests.helpers import small_config


class TestConfig(unittest.TestCase):
    def test_defaults_are_filled_and_normalising_twice_changes_nothing(self):
        cfg = normalize_config({"input_size": 3, "layers": [{"size": 2, "activation": "softmax"}]})
        self.assertEqual(cfg["loss"], "cross_entropy")
        self.assertEqual(cfg["optimizer"], {"name": "adam", "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8})
        self.assertEqual(cfg["lr_schedule"], {"name": "constant"})
        self.assertEqual(cfg["training"]["batch_size"], 256)
        self.assertEqual(cfg["layers"][0]["init"], "xavier")
        self.assertEqual(normalize_config(cfg), cfg)

    def test_short_forms(self):
        cfg = normalize_config(small_config(optimizer="sgd", lr_schedule="cosine",
                                            training={"early_stopping": True}))
        self.assertEqual(cfg["optimizer"], {"name": "sgd", "lr": 0.01})
        self.assertEqual(cfg["lr_schedule"], {"name": "cosine", "t_max": None, "lr_min": 0.0})
        self.assertEqual(cfg["training"]["early_stopping"],
                         {"patience": 5, "min_delta": 0.0, "restore_best": True})

    def test_input_is_not_modified(self):
        cfg = small_config()
        before = copy.deepcopy(cfg)
        normalize_config(cfg)
        self.assertEqual(cfg, before)

    def test_errors_name_the_faulty_key(self):
        cases = {
            "input_size": {"input_size": 0},
            "layers[0].activation": {"layers": [{"size": 2, "activation": "relux"}, {"size": 2, "activation": "softmax"}]},
            "layers[0].size": {"layers": [{"size": -1, "activation": "relu"}, {"size": 2, "activation": "softmax"}]},
            "layers[1].dropout": {"layers": [{"size": 2, "activation": "relu"}, {"size": 2, "activation": "softmax", "dropout": 0.5}]},
            "layers[0].dropout": {"layers": [{"size": 2, "activation": "relu", "dropout": 1.0}, {"size": 2, "activation": "softmax"}]},
            "layers[0]": {"layers": [{"size": 2, "activation": "relu", "alpha": 0.1}, {"size": 2, "activation": "softmax"}]},
            "loss": {"layers": [{"size": 2, "activation": "relu"}]},
            "labels": {"labels": ["a"]},
            "config": {"colour": "blue"},
            "optimizer": {"optimizer": {"name": "adam", "beta3": 0.5}},
            "optimizer.lr": {"optimizer": {"name": "adam", "lr": 0}},
            "optimizer.name": {"optimizer": "adamw"},
            "lr_schedule": {"lr_schedule": {"name": "step", "rate": 2}},
            "training.epochs": {"training": {"epochs": 0}},
            "training.early_stopping.patience": {"training": {"early_stopping": {"patience": 1.5}}},
            "seed": {"seed": -1},
            "dtype": {"dtype": "float16"},
        }
        for where, change in cases.items():
            with self.subTest(where=where):
                with self.assertRaises(ConfigError) as ctx:
                    normalize_config(small_config(**change))
                self.assertTrue(str(ctx.exception).startswith(where), str(ctx.exception))

    def test_missing_keys(self):
        for key in ("input_size", "layers"):
            cfg = small_config()
            del cfg[key]
            with self.assertRaises(ConfigError):
                normalize_config(cfg)
        with self.assertRaises(ConfigError):
            normalize_config([1, 2])

    def test_activation_parameters(self):
        cfg = normalize_config(small_config(layers=[{"size": 4, "activation": "leaky_relu", "alpha": 0.2},
                                                    {"size": 2, "activation": "softmax"}]))
        self.assertEqual(cfg["layers"][0]["alpha"], 0.2)


if __name__ == "__main__":
    unittest.main()
