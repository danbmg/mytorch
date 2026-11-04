"""Network files (.nn): a numpy .npz archive (zip) holding

- `header`: UTF-8 JSON, readable as is (`unzip -p FILE.nn header.npy`): format
  version, full config (layers, activation of each layer, optimizer, learning
  rate and schedule, regularisation, seed), labels, training metadata and history;
- `W0, b0, W1, b1...`: weights and biases in raw binary float32 (exact, 3x smaller
  than text, fast to load).

writer.py writes atomically (next to the target, then renamed over it, so an
interrupted save never leaves a half-written network); reader.py validates the
archive and the format version before rebuilding the network.
"""

from ..errors import NetworkFormatError
from .reader import load_network, read_header
from .writer import FORMAT, FORMAT_VERSION, save_network

__all__ = ["FORMAT", "FORMAT_VERSION", "NetworkFormatError", "load_network", "read_header", "save_network"]
