"""my_torch: a small neural network library written from scratch with numpy.

It knows nothing about chess: it builds, trains, saves and loads fully connected
networks described by a JSON configuration.

numpy's matrix products run on its default number of BLAS threads (all the
cores); OPENBLAS_NUM_THREADS changes it. parallel.train_many() gives each of its
worker processes its own number of threads.
"""

from .config import normalize_config
from .errors import ConfigError, NetworkFormatError, TrainingError
from .network import Network, load_network
from .serialization import read_header, save_network

__all__ = ["ConfigError", "Network", "NetworkFormatError", "TrainingError", "load_network",
           "normalize_config", "read_header", "save_network"]
