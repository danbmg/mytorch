"""my_torch_generator: create blank (untrained) networks from configuration files.

Every configuration file is read and validated before anything is written, so
a bad second file never leaves half of the networks behind. Network i of a
configuration with a seed uses seed + i - 1: generating again gives the same
weights, and the i networks of one file differ from each other.
"""

import json
import os
import sys

from analyzer.errors import EXIT_FAILURE
from my_torch import ConfigError, Network, normalize_config

USAGE = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i    Configuration file containing description of a neural network we want to
                     generate (JSON, see README.md).
    nb_i             Number of neural networks to generate based on the configuration file.

    The networks are written in the current directory, named after their configuration
    file: basic.conf 3 -> basic_1.nn basic_2.nn basic_3.nn. If the configuration has a
    "seed", network i uses seed + i - 1: generating again gives the same weights."""


def read_config(path):
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        normalize_config(config)
        return config
    except OSError as e:
        raise ValueError(f"{path}: {e.strerror}") from None
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not a text file") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}") from None
    except ConfigError as e:
        raise ValueError(f"{path}: {e}") from None


def parse_jobs(args):
    """[config, nb, config, nb...] -> [(path, config, count)], every file checked before anything is written."""
    if not args or len(args) % 2:
        raise ValueError("expected pairs of arguments: config_file nb (see --help)")
    jobs = []
    for path, count in zip(args[::2], args[1::2]):
        if not count.isdigit() or int(count) < 1:
            raise ValueError(f"{count}: the number of networks must be a positive integer")
        jobs.append((path, read_config(path), int(count)))
    return jobs


def write_networks(path, config, count):
    """<name>_1.nn ... <name>_<count>.nn in the current directory, name = the config file without extension."""
    name, seed = os.path.splitext(os.path.basename(path))[0], config.get("seed")
    for i in range(1, count + 1):
        Network(config, seed=None if seed is None else seed + i - 1).save(f"{name}_{i}.nn")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args[:1] in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    try:
        for path, config, count in parse_jobs(args):
            write_networks(path, config, count)
    except ValueError as e:
        print(f"my_torch_generator: {e}", file=sys.stderr)
        return EXIT_FAILURE
    except OSError as e:
        print(f"my_torch_generator: cannot write network: {e.strerror}", file=sys.stderr)
        return EXIT_FAILURE
    return 0
