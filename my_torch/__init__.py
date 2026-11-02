"""my_torch: a small neural network library written from scratch with numpy.

It knows nothing about chess: it builds, trains, saves and loads fully connected
networks described by a JSON configuration.
"""

import os as _os
import sys as _sys

# BLAS threads behind numpy's matrix products. On mini-batches of a few hundred
# rows, 1 thread is about as fast as 12 and leaves the other cores to parallel.py.
# Only effective before numpy is imported; MY_TORCH_THREADS=auto keeps numpy's
# default, an explicit OPENBLAS_NUM_THREADS etc. always wins.
_threads = _os.environ.get("MY_TORCH_THREADS", "1")
if "numpy" not in _sys.modules and _threads != "auto":
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_var, _threads)

from .config import ConfigError, normalize_config  # noqa: E402
from .network import Network, TrainingError  # noqa: E402
from .serialization import NetworkFormatError, load_network, read_header, save_network  # noqa: E402

__all__ = ["ConfigError", "Network", "NetworkFormatError", "TrainingError", "load_network",
           "normalize_config", "read_header", "save_network"]
