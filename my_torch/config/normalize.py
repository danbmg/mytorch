"""The sections of a configuration and their defaults: layers, loss and dtype,
optimizer, learning rate schedule, training, labels and seed. Every default
lives here, once, and the registries say which names are valid."""

import copy

from ..activations import ACTIVATIONS, ALPHA
from ..errors import lookup
from ..initializers import INITS, resolve
from ..losses import LOSSES
from ..optimizers import OPTIMIZERS
from ..schedules import SCHEDULES
from .checks import check, check_keys, fill, is_int, is_number, named

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


def _init(init, where):
    check(init == "auto" or init in INITS, where, f"unknown '{init}' (choices: auto, {', '.join(INITS)})")
    return init


def _layer(layer, where, last, default_init):
    check(isinstance(layer, dict), where, 'must be an object like {"size": 128, "activation": "relu"}')
    act = layer.get("activation")
    lookup(ACTIVATIONS, act, f"{where}.activation")
    alpha = {"alpha": ALPHA[act]} if act in ALPHA else {}
    check_keys(layer, ("size", "activation", "dropout", "init", *alpha), where)
    check(is_int(layer.get("size")), f"{where}.size", "must be a positive integer")
    dropout = layer.get("dropout", 0.0)
    check(is_number(dropout) and 0 <= dropout < 1, f"{where}.dropout", "must be a number in [0, 1)")
    check(dropout == 0 or not last, f"{where}.dropout", "the output layer cannot use dropout")
    init = _init(layer.get("init", default_init), f"{where}.init")
    alpha = fill({k: layer[k] for k in alpha if k in layer}, alpha, where)
    return {"size": layer["size"], "activation": act, **alpha, "dropout": float(dropout), "init": resolve(init, act)}


def _layers(layers, default_init):
    check(isinstance(layers, list) and layers, "layers", "must be a non-empty list of layers")
    return [_layer(layer, f"layers[{i}]", i == len(layers) - 1, default_init) for i, layer in enumerate(layers)]


def _loss_and_dtype(cfg, last_activation):
    loss = cfg.get("loss", "cross_entropy")
    check(loss in LOSSES, "loss", f"unknown '{loss}' (choices: {', '.join(LOSSES)})")
    check(loss != "cross_entropy" or last_activation == "softmax", "loss",
          f"'cross_entropy' needs a 'softmax' output layer (got '{last_activation}')")
    dtype = cfg.get("dtype", "float32")
    check(dtype in ("float32", "float64"), "dtype", "must be float32 or float64")
    return {"loss": loss, "dtype": dtype}


def _optimizer(spec):
    name, opt = named(spec, OPTIMIZERS, "adam", "optimizer")
    lr = opt.pop("lr", OPTIMIZERS[name].LR)
    check(is_number(lr) and lr > 0, "optimizer.lr", "must be a number > 0")
    return {"name": name, "lr": float(lr), **fill(opt, OPTIMIZERS[name].DEFAULTS, "optimizer")}


def _schedule(spec):
    name, sched = named(spec, SCHEDULES, "constant", "lr_schedule")
    return {"name": name, **fill(sched, SCHEDULES[name].DEFAULTS, "lr_schedule")}


def _labels(labels, n_out):
    if labels is not None:
        check(isinstance(labels, list) and all(isinstance(x, str) for x in labels) and len(set(labels)) == len(labels),
              "labels", "must be a list of distinct strings")
        check(len(labels) == n_out, "labels", f"{len(labels)} labels for an output layer of size {n_out}")
    return labels


def normalize_training(spec):
    spec = {} if spec is None else spec
    check_keys(spec, TRAINING, "training")
    out = {**TRAINING, **spec}
    for key in ("epochs", "batch_size"):
        check(is_int(out[key]), f"training.{key}", "must be a positive integer")
    check(is_number(out["l2"]) and out["l2"] >= 0, "training.l2", "must be a number >= 0")
    check(out["grad_clip"] is None or (is_number(out["grad_clip"]) and out["grad_clip"] > 0),
          "training.grad_clip", "must be null or a number > 0")
    check(isinstance(out["shuffle"], bool), "training.shuffle", "must be true or false")
    check(is_int(out["eval_train_samples"], 0), "training.eval_train_samples", "must be an integer >= 0")
    es = out["early_stopping"]
    out["early_stopping"] = None if es in (None, False) else fill({} if es is True else es, EARLY_STOPPING,
                                                                  "training.early_stopping")
    return out


def normalize_config(config):
    """Validated copy of `config`, with every default filled in."""
    check_keys(config, KEYS, "config")
    cfg = copy.deepcopy(config)
    for key in ("input_size", "layers"):
        check(key in cfg, "config", f"missing '{key}'")
    check(is_int(cfg["input_size"]), "input_size", "must be a positive integer")
    init = _init(cfg.get("init", "auto"), "init")
    out = {k: cfg[k] for k in ("name", "description") if k in cfg}
    out.update(input_size=cfg["input_size"], init=init, layers=_layers(cfg["layers"], init))
    out.update(_loss_and_dtype(cfg, out["layers"][-1]["activation"]))
    out["optimizer"] = _optimizer(cfg.get("optimizer"))
    out["lr_schedule"] = _schedule(cfg.get("lr_schedule"))
    out["training"] = normalize_training(cfg.get("training"))
    labels, seed = _labels(cfg.get("labels"), out["layers"][-1]["size"]), cfg.get("seed")
    check(seed is None or is_int(seed, 0), "seed", "must be null or an integer >= 0")
    out.update(labels=labels, seed=seed)
    return out
