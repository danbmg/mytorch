"""From genes to a network configuration: dotted paths written into the config,
and the architecture genes (hidden.*) that rebuild the hidden layers. The base
config is never modified: every trial gets its own deep copy."""

import copy

from .space import SpaceError


def _indices(items, key, path):
    if key == "*":
        return range(len(items))
    if key == "hidden":
        return range(len(items) - 1)
    if key.isdigit():
        return [int(key)]
    raise SpaceError(f"{path}: '{key}' is not a list index (use a number, * or hidden)")


def set_path(config, path, value):
    """config[a][b]... = value for path "a.b...". In a list, "*" is every item,
    "hidden" every item but the last, a number one item: "layers.hidden.dropout"."""
    *keys, last = path.split(".")
    targets = [config]
    for key in keys:
        following = []
        for target in targets:
            if isinstance(target, list):
                following += [target[i] for i in _indices(target, key, path)]
            else:
                if not isinstance(target.get(key), (dict, list)):
                    target[key] = {}
                following.append(target[key])
        targets = following
    for target in targets:
        for i in (_indices(target, last, path) if isinstance(target, list) else [last]):
            target[i] = copy.deepcopy(value)


def _hidden_layers(config, hidden):
    """The hidden layers rebuilt from the hidden.* genes; missing genes come from the first layer."""
    first = config["layers"][0] if len(config["layers"]) > 1 else {}
    size = hidden.get("size", first.get("size", 128))
    layers = []
    for _ in range(int(hidden.get("layers", len(config["layers"]) - 1))):
        layers.append({"size": max(int(round(size)), 1),
                       "activation": hidden.get("activation", first.get("activation", "relu")),
                       "dropout": hidden.get("dropout", first.get("dropout", 0.0))})
        size *= hidden.get("ratio", 1.0)
    return layers


def _apply_gene(config, key, value):
    section = key.split(".")[0]
    current = config.get(section, {})
    if key in ("optimizer.name", "lr_schedule.name") and current.get("name") != value:
        # another optimizer / schedule: the settings of the old one do not apply, the lr does
        config[section] = {"name": value}
        if "lr" in current:
            config[section]["lr"] = current["lr"]
    elif not key.startswith("hidden."):
        set_path(config, key, value)


def apply_genes(base, genes):
    """Network config = base config with the genes applied."""
    config = copy.deepcopy(base)
    hidden = {k[len("hidden."):]: v for k, v in genes.items() if k.startswith("hidden.")}
    if hidden:
        config["layers"] = _hidden_layers(config, hidden) + [config["layers"][-1]]
    for key, value in genes.items():
        _apply_gene(config, key, value)
    return config
