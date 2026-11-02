"""my_torch: a small neural network library written from scratch with numpy.

It knows nothing about chess: it builds, trains, saves and loads fully
connected networks described by a JSON configuration.
"""

import os as _os
import sys as _sys

# Threads of the BLAS library behind numpy's matrix products. On mini-batches
# of a few hundred rows, 1 thread is faster than 12 (synchronisation costs more
# than it saves, see docs/nn_module.md), and it leaves the other cores to
# parallel.py. Only effective if numpy is not imported yet; MY_TORCH_THREADS=auto
# keeps numpy's default, an explicit OPENBLAS_NUM_THREADS etc. always wins.
_threads = _os.environ.get("MY_TORCH_THREADS", "1")
if "numpy" not in _sys.modules and _threads != "auto":
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_var, _threads)

from .config import ConfigError, normalize_config
from .network import Network, TrainingError
from .serialization import NetworkFormatError, load_network, read_header, save_network

__all__ = [
    "ConfigError",
    "Network",
    "NetworkFormatError",
    "TrainingError",
    "load_network",
    "normalize_config",
    "read_header",
    "save_network",
]
