"""The registries: every entry builds and runs, an unknown name is refused with the choices listed."""

import re
import unittest

import numpy as np

from my_torch import ConfigError, activations
from my_torch.activations import ACTIVATIONS
from my_torch.errors import lookup
from my_torch.initializers import INITS
from my_torch.optimizers import OPTIMIZERS, Optimizer
from my_torch.schedules import SCHEDULES, Schedule


class TestRegistries(unittest.TestCase):
    def test_unknown_names_are_refused_with_the_choices(self):
        for registry, name, where in ((ACTIVATIONS, "relux", "layers[0].activation"), (INITS, "random", "init"),
                                      (OPTIMIZERS, "adamw", "optimizer"), (SCHEDULES, "linear", "lr_schedule")):
            with self.subTest(name=name), self.assertRaisesRegex(ConfigError, re.escape(f"{where}: unknown '{name}' (choices: ")):
                lookup(registry, name, where)
        with self.assertRaises(ConfigError):
            Optimizer("adamw", 0.1)
        with self.assertRaises(ConfigError):
            Schedule({"name": "linear"}, 1.0, 10)
        with self.assertRaises(ConfigError):
            activations.forward("nope", np.zeros((1, 1)))

    def test_every_activation_runs_forward_and_backward(self):
        z = np.random.default_rng(0).normal(size=(3, 4))
        for name, (forward, backward) in ACTIVATIONS.items():
            with self.subTest(activation=name):
                a = forward(z, 0.1)
                self.assertEqual((a.shape, backward(z, a, np.ones_like(z), 0.1).shape), (z.shape, z.shape))
                np.testing.assert_array_equal(activations.forward(name, z, 0.1), a)

    def test_every_initializer_gives_the_right_shape(self):
        rng = np.random.default_rng(0)
        for name, init in INITS.items():
            self.assertEqual(init(5, 3, rng).shape, (5, 3), name)

    def test_every_optimizer_moves_the_parameters_against_the_gradient(self):
        for name, cls in OPTIMIZERS.items():
            with self.subTest(optimizer=name):
                opt = Optimizer(name, 0.1)
                self.assertIsInstance(opt, cls)
                p = np.ones(3)
                opt.step([p], [np.ones(3)])
                self.assertTrue((p < 1).all())
                self.assertEqual(len(opt.state()), cls.MOMENTS)

    def test_every_schedule_starts_at_lr0(self):
        for name, cls in SCHEDULES.items():
            with self.subTest(schedule=name):
                schedule = Schedule({"name": name, **cls.DEFAULTS}, 0.5, 10)
                self.assertIsInstance(schedule, cls)
                self.assertAlmostEqual(schedule.lr(0), 0.5)


if __name__ == "__main__":
    unittest.main()
