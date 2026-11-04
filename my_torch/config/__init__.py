"""Network configuration: validation and defaults.

normalize_config() returns a copy with every missing setting filled in, so a
saved network lists every hyperparameter it used. An invalid config raises
ConfigError naming the faulty key: "layers[0].activation: unknown 'relux' (...)".
checks.py holds the generic checks, normalize.py the defaults and the sections.
"""

from ..errors import ConfigError
from .normalize import EARLY_STOPPING, KEYS, TRAINING, normalize_config, normalize_training

__all__ = ["ConfigError", "EARLY_STOPPING", "KEYS", "TRAINING", "normalize_config", "normalize_training"]
