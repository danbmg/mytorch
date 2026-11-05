"""Generic checks of a configuration value, each raising ConfigError with the
path of the faulty key. Shared by every section of normalize.py so that all the
error messages have the same shape: "where: what is wrong (what is allowed)"."""

import numbers

from ..errors import ConfigError


def check(ok, where, message):
    if not ok:
        raise ConfigError(f"{where}: {message}")


def is_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def is_int(v, low=1):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool) and v >= low


def check_keys(spec, allowed, where):
    check(isinstance(spec, dict), where, "must be an object")
    unknown = sorted(set(spec) - set(allowed))
    check(not unknown, where, f"unknown key(s) {', '.join(unknown)} (allowed: {', '.join(allowed)})")


def fill(spec, defaults, where):
    """spec completed with defaults; each value must have the type of its default."""
    check_keys(spec, defaults, where)
    out = {**defaults, **spec}
    for key, default in defaults.items():
        value = out[key]
        here = f"{where}.{key}"
        if isinstance(default, bool):
            check(isinstance(value, bool), here, "must be true or false")
        elif isinstance(default, int) or default is None:
            check(is_int(value) or (default is None and value is None), here, "must be a positive integer")
        else:
            check(is_number(value) and value >= 0, here, "must be a number >= 0")
    return out


def named(spec, registry, default, where):
    """"adam" or {"name": "adam", ...} -> ("adam", {settings}); the name must be in the registry."""
    spec = {"name": default} if spec is None else {"name": spec} if isinstance(spec, str) else spec
    check(isinstance(spec, dict), where, 'must be a name or an object like {"name": ...}')
    spec = dict(spec)
    name = spec.pop("name", None)
    check(name in registry, f"{where}.name", f"unknown '{name}' (choices: {', '.join(registry)})")
    return name, spec
