"""Network configuration: validation and defaults.

normalize_config() returns a copy with every missing setting filled in, so a
saved network lists every hyperparameter it used. An invalid config raises
ConfigError naming the faulty key: "layers[0].activation: unknown 'relux' (...)".
"""

import copy
import numbers

from .activations import ACTIVATIONS, ALPHA
from .errors import ConfigError, lookup
from .initializers import INITS, resolve
from .losses import LOSSES
from .optimizers import DEFAULT_LR, OPTIMIZERS
from .schedules import SCHEDULES


TRAINING = {
    "epochs": 20,
    "batch_size": 256,
    "l2": 0.0,
    "grad_clip": None,
    "early_stopping": None,
    "shuffle": True,
    # train loss/accuracy re-measured each epoch, dropout off, on this many samples (0: running average)
    "eval_train_samples": 20000,
}
EARLY_STOPPING = {"patience": 5, "min_delta": 0.0, "restore_best": True}
KEYS = ("name", "description", "input_size", "layers", "init", "loss", "dtype",
        "optimizer", "lr_schedule", "training", "labels", "seed")


def _check(ok, where, message):
    if not ok:
        raise ConfigError(f"{where}: {message}")


def _num(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _int(v, low=1):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool) and v >= low


def _keys(spec, allowed, where):
    _check(isinstance(spec, dict), where, "must be an object")
    unknown = sorted(set(spec) - set(allowed))
    _check(not unknown, where, f"unknown key(s) {', '.join(unknown)} (allowed: {', '.join(allowed)})")


def _fill(spec, defaults, where):
    """spec completed with defaults; each value must have the type of its default."""
    _keys(spec, defaults, where)
    out = {**defaults, **spec}
    for key, default in defaults.items():
        value, here = out[key], f"{where}.{key}"
        if isinstance(default, bool):
            _check(isinstance(value, bool), here, "must be true or false")
        elif isinstance(default, int) or default is None:
            _check(_int(value) or (default is None and value is None), here, "must be a positive integer")
        else:
            _check(_num(value) and value >= 0, here, "must be a number >= 0")
    return out


def _named(spec, registry, default, where):
    """"adam" or {"name": "adam", ...} -> {"name": "adam", **settings with defaults}."""
    spec = {"name": default} if spec is None else {"name": spec} if isinstance(spec, str) else spec
    _check(isinstance(spec, dict), where, 'must be a name or an object like {"name": ...}')
    spec = dict(spec)
    name = spec.pop("name", None)
    _check(name in registry, f"{where}.name", f"unknown '{name}' (choices: {', '.join(registry)})")
    return name, spec


def _layers(layers, default_init):
    _check(isinstance(layers, list) and layers, "layers", "must be a non-empty list of layers")
    out = []
    for i, layer in enumerate(layers):
        where = f"layers[{i}]"
        _check(isinstance(layer, dict), where, 'must be an object like {"size": 128, "activation": "relu"}')
        act = layer.get("activation")
        lookup(ACTIVATIONS, act, f"{where}.activation")
        alpha = {"alpha": ALPHA[act]} if act in ALPHA else {}
        _keys(layer, ("size", "activation", "dropout", "init", *alpha), where)
        _check(_int(layer.get("size")), f"{where}.size", "must be a positive integer")
        dropout = layer.get("dropout", 0.0)
        _check(_num(dropout) and 0 <= dropout < 1, f"{where}.dropout", "must be a number in [0, 1)")
        _check(dropout == 0 or i < len(layers) - 1, f"{where}.dropout", "the output layer cannot use dropout")
        init = layer.get("init", default_init)
        _check(init == "auto" or init in INITS, f"{where}.init", f"unknown '{init}' (choices: auto, {', '.join(INITS)})")
        alpha = _fill({k: layer[k] for k in alpha if k in layer}, alpha, where)
        out.append({"size": layer["size"], "activation": act, **alpha, "dropout": float(dropout),
                    "init": resolve(init, act)})
    return out


def normalize_training(spec):
    spec = {} if spec is None else spec
    _keys(spec, TRAINING, "training")
    out = {**TRAINING, **spec}
    for key in ("epochs", "batch_size"):
        _check(_int(out[key]), f"training.{key}", "must be a positive integer")
    _check(_num(out["l2"]) and out["l2"] >= 0, "training.l2", "must be a number >= 0")
    _check(out["grad_clip"] is None or (_num(out["grad_clip"]) and out["grad_clip"] > 0),
           "training.grad_clip", "must be null or a number > 0")
    _check(isinstance(out["shuffle"], bool), "training.shuffle", "must be true or false")
    _check(_int(out["eval_train_samples"], 0), "training.eval_train_samples", "must be an integer >= 0")
    es = out["early_stopping"]
    out["early_stopping"] = None if es in (None, False) else _fill({} if es is True else es, EARLY_STOPPING,
                                                                    "training.early_stopping")
    return out


def normalize_config(config):
    """Validated copy of `config`, with every default filled in."""
    _keys(config, KEYS, "config")
    cfg = copy.deepcopy(config)
    for key in ("input_size", "layers"):
        _check(key in cfg, "config", f"missing '{key}'")
    _check(_int(cfg["input_size"]), "input_size", "must be a positive integer")
    init = cfg.get("init", "auto")
    _check(init == "auto" or init in INITS, "init", f"unknown '{init}' (choices: auto, {', '.join(INITS)})")
    out = {k: cfg[k] for k in ("name", "description") if k in cfg}
    out.update(input_size=cfg["input_size"], init=init, layers=_layers(cfg["layers"], init))

    loss, last = cfg.get("loss", "cross_entropy"), out["layers"][-1]["activation"]
    _check(loss in LOSSES, "loss", f"unknown '{loss}' (choices: {', '.join(LOSSES)})")
    _check(loss != "cross_entropy" or last == "softmax", "loss", f"'cross_entropy' needs a 'softmax' output layer (got '{last}')")
    dtype = cfg.get("dtype", "float32")
    _check(dtype in ("float32", "float64"), "dtype", "must be float32 or float64")
    out.update(loss=loss, dtype=dtype)

    name, opt = _named(cfg.get("optimizer"), OPTIMIZERS, "adam", "optimizer")
    lr = opt.pop("lr", DEFAULT_LR[name])
    _check(_num(lr) and lr > 0, "optimizer.lr", "must be a number > 0")
    out["optimizer"] = {"name": name, "lr": float(lr), **_fill(opt, OPTIMIZERS[name], "optimizer")}
    name, sched = _named(cfg.get("lr_schedule"), SCHEDULES, "constant", "lr_schedule")
    out["lr_schedule"] = {"name": name, **_fill(sched, SCHEDULES[name], "lr_schedule")}
    out["training"] = normalize_training(cfg.get("training"))

    labels = cfg.get("labels")
    if labels is not None:
        _check(isinstance(labels, list) and all(isinstance(x, str) for x in labels) and len(set(labels)) == len(labels),
               "labels", "must be a list of distinct strings")
        n_out = out["layers"][-1]["size"]
        _check(len(labels) == n_out, "labels", f"{len(labels)} labels for an output layer of size {n_out}")
    seed = cfg.get("seed")
    _check(seed is None or _int(seed, 0), "seed", "must be null or an integer >= 0")
    out.update(labels=labels, seed=seed)
    return out
