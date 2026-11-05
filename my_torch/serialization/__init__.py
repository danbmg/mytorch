"""Network files (.nn): a numpy .npz archive (zip) holding

- `header`: UTF-8 JSON, readable as is (`unzip -p FILE.nn header.npy`): format
  version, full config (layers, activation of each layer, optimizer, learning
  rate and schedule, regularization, seed), labels, training metadata and history;
- `W0, b0, W1, b1...`: weights and biases in raw binary float32 (exact, 3x smaller
  than text, fast to load).

writer.py writes atomically (next to the target, then renamed over it, so an
interrupted save never leaves a half-written network); reader.py checks the
archive and the format version and returns the header and the arrays, from
which Network.load() rebuilds the network.
"""

from ..errors import NetworkFormatError
from .reader import read_header, read_network_file
from .writer import FORMAT, FORMAT_VERSION, save_network

__all__ = ["FORMAT", "FORMAT_VERSION", "NetworkFormatError", "read_header", "read_network_file", "save_network"]
