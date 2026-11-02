"""my_torch_generator: create blank (untrained) networks from configuration files."""

import json
import os
import sys

from my_torch import ConfigError, Network, normalize_config

USAGE = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i    Configuration file containing description of a neural network we want to
                     generate (JSON, see docs/nn_module.md).
    nb_i             Number of neural networks to generate based on the configuration file.

    The networks are written in the current directory, named after their configuration
    file: basic_network.conf 3 -> basic_network_1.nn basic_network_2.nn basic_network_3.nn.
    If the configuration has a "seed", network i uses seed + i - 1: generating again gives
    the same weights. Without a seed, every network gets a random one (kept in its file)."""


class UsageError(Exception):
    pass


def read_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise UsageError(f"{path}: no such file") from None
    except IsADirectoryError:
        raise UsageError(f"{path}: is a directory") from None
    except OSError as e:
        raise UsageError(f"{path}: {e.strerror}") from None
    except UnicodeDecodeError:
        raise UsageError(f"{path}: not a text file") from None
    except json.JSONDecodeError as e:
        raise UsageError(f"{path}: invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}") from None
    try:
        normalize_config(config)
    except ConfigError as e:
        raise UsageError(f"{path}: {e}") from None
    return config


def parse_args(args):
    if len(args) == 0 or len(args) % 2:
        raise UsageError("expected pairs of arguments: config_file nb (see --help)")
    jobs = []
    for path, count in zip(args[::2], args[1::2]):
        if not count.isdigit() or int(count) < 1:
            raise UsageError(f"{count}: the number of networks must be a positive integer")
        jobs.append((path, read_config(path), int(count)))
    return jobs


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    try:
        # every file is checked before anything is written
        jobs = parse_args(args)
        for path, config, count in jobs:
            name = os.path.splitext(os.path.basename(path))[0]
            base_seed = config.get("seed")
            for i in range(1, count + 1):
                seed = base_seed + i - 1 if base_seed is not None else None
                Network(config, seed=seed).save(f"{name}_{i}.nn")
    except UsageError as e:
        print(f"my_torch_generator: {e}", file=sys.stderr)
        return 84
    except OSError as e:
        print(f"my_torch_generator: cannot write network: {e.strerror}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main())
