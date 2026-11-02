"""Network configuration: validation and defaults.

A configuration is a JSON-compatible dict (see docs/nn_module.md).
`normalize_config()` checks it and fills every missing setting with its
default, so that a saved network lists every hyperparameter actually used.
Errors raise `ConfigError` with the path of the faulty key.
"""

import copy
import inspect
import numbers

from .activations import ACTIVATIONS
from .init import INITS, resolve
from .losses import LOSSES
from .optimizers import OPTIMIZERS
from .schedules import SCHEDULES


class ConfigError(ValueError):
    pass


DTYPES = ("float32", "float64")

DEFAULT_LR = {"sgd": 0.01, "momentum": 0.01, "rmsprop": 0.001, "adam": 0.001}

DEFAULT_TRAINING = {
    "epochs": 20,
    "batch_size": 256,
    "l2": 0.0,
    "grad_clip": None,
    "early_stopping": None,
    "shuffle": True,
    # train loss/accuracy are re-measured each epoch, without dropout, on this many samples
    "eval_train_samples": 20000,
}

DEFAULT_EARLY_STOPPING = {"patience": 5, "min_delta": 0.0, "restore_best": True}

TOP_LEVEL_KEYS = (
    "name", "description", "input_size", "layers", "init", "loss", "dtype",
    "optimizer", "lr_schedule", "training", "labels", "seed",
)


def _defaults(cls, skip):
    """Keyword arguments of cls.__init__ and their defaults."""
    params = inspect.signature(cls.__init__).parameters
    keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {n: p.default for n, p in params.items()
            if p.kind in keyword and n not in skip and n != "self"}


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_int(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _check(condition, where, message):
    if not condition:
        raise ConfigError(f"{where}: {message}")


def _check_keys(spec, allowed, where):
    unknown = sorted(set(spec) - set(allowed))
    _check(not unknown, where, f"unknown key(s) {', '.join(unknown)} (allowed: {', '.join(allowed)})")


def _fill_params(spec, defaults, where):
    """Fill `spec` with `defaults`, checking each value has the default's type."""
    _check_keys(spec, defaults, where)
    out = {}
    for key, default in defaults.items():
        value = spec.get(key, default)
        if isinstance(default, bool):
            _check(isinstance(value, bool), f"{where}.{key}", "must be true or false")
        elif isinstance(default, int):
            _check(_is_int(value) and value > 0, f"{where}.{key}", "must be a positive integer")
        elif isinstance(default, float):
            _check(_is_number(value), f"{where}.{key}", "must be a number")
        elif default is None:
            _check(value is None or (_is_int(value) and value > 0), f"{where}.{key}",
                   "must be null or a positive integer")
        out[key] = value
    return out


def _named_spec(spec, registry, default, where):
    """Accept "adam" or {"name": "adam", ...}; return a dict with a valid name."""
    if spec is None:
        spec = {"name": default}
    elif isinstance(spec, str):
        spec = {"name": spec}
    _check(isinstance(spec, dict), where, 'must be a name or an object like {"name": ...}')
    name = spec.get("name")
    _check(name in registry, f"{where}.name", f"unknown '{name}' (choices: {', '.join(registry)})")
    return dict(spec)


def _normalize_layers(layers, default_init, where="layers"):
    _check(isinstance(layers, list) and layers, where, "must be a non-empty list of layers")
    out = []
    for i, layer in enumerate(layers):
        here = f"{where}[{i}]"
        _check(isinstance(layer, dict), here, 'must be an object like {"size": 128, "activation": "relu"}')
        _check("size" in layer, here, "missing 'size'")
        _check(_is_int(layer["size"]) and layer["size"] > 0, f"{here}.size", "must be a positive integer")
        act = layer.get("activation")
        _check(act is not None, here, f"missing 'activation' (choices: {', '.join(ACTIVATIONS)})")
        _check(act in ACTIVATIONS, f"{here}.activation", f"unknown '{act}' (choices: {', '.join(ACTIVATIONS)})")
        act_defaults = _defaults(ACTIVATIONS[act], skip=())
        allowed = ("size", "activation", "dropout", "init", *act_defaults)
        _check_keys(layer, allowed, here)
        act_params = _fill_params({k: layer[k] for k in act_defaults if k in layer}, act_defaults, here)
        dropout = layer.get("dropout", 0.0)
        _check(_is_number(dropout) and 0 <= dropout < 1, f"{here}.dropout", "must be a number in [0, 1)")
        if i == len(layers) - 1:
            _check(dropout == 0, f"{here}.dropout", "the output layer cannot use dropout")
        init = layer.get("init", default_init)
        _check(init == "auto" or init in INITS, f"{here}.init", f"unknown '{init}' (choices: auto, {', '.join(INITS)})")
        out.append({"size": int(layer["size"]), "activation": act, **act_params,
                    "dropout": float(dropout), "init": resolve(init, act)})
    return out


def _normalize_training(spec, where="training"):
    spec = {} if spec is None else spec
    _check(isinstance(spec, dict), where, "must be an object")
    _check_keys(spec, DEFAULT_TRAINING, where)
    out = {**DEFAULT_TRAINING, **spec}
    for key in ("epochs", "batch_size"):
        _check(_is_int(out[key]) and out[key] > 0, f"{where}.{key}", "must be a positive integer")
    _check(_is_number(out["l2"]) and out["l2"] >= 0, f"{where}.l2", "must be a number >= 0")
    _check(out["grad_clip"] is None or (_is_number(out["grad_clip"]) and out["grad_clip"] > 0),
           f"{where}.grad_clip", "must be null or a number > 0")
    _check(isinstance(out["shuffle"], bool), f"{where}.shuffle", "must be true or false")
    _check(_is_int(out["eval_train_samples"]) and out["eval_train_samples"] >= 0,
           f"{where}.eval_train_samples", "must be an integer >= 0")
    es = out["early_stopping"]
    if es is True:
        es = {}
    if es is None or es is False:
        out["early_stopping"] = None
    else:
        _check(isinstance(es, dict), f"{where}.early_stopping", "must be null, true or an object")
        filled = _fill_params(es, DEFAULT_EARLY_STOPPING, f"{where}.early_stopping")
        _check(filled["min_delta"] >= 0, f"{where}.early_stopping.min_delta", "must be >= 0")
        out["early_stopping"] = filled
    return out


def normalize_config(config):
    """Return a validated copy of `config` with every default filled in."""
    _check(isinstance(config, dict), "config", "must be a JSON object")
    cfg = copy.deepcopy(config)
    _check_keys(cfg, TOP_LEVEL_KEYS, "config")
    out = {}
    if "name" in cfg:
        _check(isinstance(cfg["name"], str), "name", "must be a string")
        out["name"] = cfg["name"]
    if "description" in cfg:
        out["description"] = cfg["description"]

    _check("input_size" in cfg, "config", "missing 'input_size'")
    _check(_is_int(cfg["input_size"]) and cfg["input_size"] > 0, "input_size", "must be a positive integer")
    out["input_size"] = int(cfg["input_size"])

    default_init = cfg.get("init", "auto")
    _check(default_init == "auto" or default_init in INITS, "init",
           f"unknown '{default_init}' (choices: auto, {', '.join(INITS)})")
    out["init"] = default_init
    _check("layers" in cfg, "config", "missing 'layers'")
    out["layers"] = _normalize_layers(cfg["layers"], default_init)

    loss = cfg.get("loss", "cross_entropy")
    _check(loss in LOSSES, "loss", f"unknown '{loss}' (choices: {', '.join(LOSSES)})")
    fused = LOSSES[loss].fused_activation
    last = out["layers"][-1]["activation"]
    _check(fused is None or last == fused, "loss", f"'{loss}' needs a '{fused}' output layer (got '{last}')")
    out["loss"] = loss

    dtype = cfg.get("dtype", "float32")
    _check(dtype in DTYPES, "dtype", f"must be one of {', '.join(DTYPES)}")
    out["dtype"] = dtype

    opt = _named_spec(cfg.get("optimizer"), OPTIMIZERS, "adam", "optimizer")
    name = opt.pop("name")
    defaults = _defaults(OPTIMIZERS[name], skip=("lr",))
    lr = opt.pop("lr", DEFAULT_LR[name])
    _check(_is_number(lr) and lr > 0, "optimizer.lr", "must be a number > 0")
    out["optimizer"] = {"name": name, "lr": float(lr), **_fill_params(opt, defaults, "optimizer")}

    sched = _named_spec(cfg.get("lr_schedule"), SCHEDULES, "constant", "lr_schedule")
    name = sched.pop("name")
    defaults = _defaults(SCHEDULES[name], skip=("lr0", "epochs"))
    out["lr_schedule"] = {"name": name, **_fill_params(sched, defaults, "lr_schedule")}

    out["training"] = _normalize_training(cfg.get("training"))

    labels = cfg.get("labels")
    if labels is not None:
        _check(isinstance(labels, list) and all(isinstance(x, str) for x in labels), "labels",
               "must be a list of strings")
        _check(len(set(labels)) == len(labels), "labels", "must be unique")
        n_out = out["layers"][-1]["size"]
        _check(len(labels) == n_out, "labels", f"{len(labels)} labels for an output layer of size {n_out}")
    out["labels"] = labels

    seed = cfg.get("seed")
    _check(seed is None or (_is_int(seed) and seed >= 0), "seed", "must be null or an integer >= 0")
    out["seed"] = seed
    return out
