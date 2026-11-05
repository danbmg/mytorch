"""The errors of the module, and the lookup in its registries.

Every configurable choice (activation, initialization, loss, optimizer, learning
rate schedule) is a registry: a dict name -> implementation. One function reports
an unknown name the same way for all of them, naming the choices, so a typo in a
configuration is refused before anything is trained or written.
"""


class ConfigError(ValueError):
    """Invalid network configuration; the message names the faulty key."""


class TrainingError(RuntimeError):
    """The training cannot go on (the loss diverged)."""


class NetworkFormatError(ValueError):
    """A .nn file that cannot be read."""


def lookup(registry, name, where):
    """registry[name], or ConfigError "where: unknown 'name' (choices: ...)"."""
    if name in registry:
        return registry[name]
    raise ConfigError(f"{where}: unknown '{name}' (choices: {', '.join(registry)})")
