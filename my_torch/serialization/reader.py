"""Reading a .nn file: the archive is checked, then the header (format and
version), then the network is rebuilt from its config and weights.

Every failure is a NetworkFormatError naming the file and what is wrong, so
the analyzer can report it without a traceback. A file of a newer format
version than this program is refused rather than misread; older versions
would be converted in upgrade() (none yet).
"""

import io
import json
import os
import zipfile

import numpy as np

from ..errors import ConfigError, NetworkFormatError
from ..network import Network
from .writer import FORMAT, FORMAT_VERSION


def _archive(path):
    """The .npz archive, read in memory first (a few MB): no file left open if it is broken."""
    if not os.path.isfile(path):
        raise NetworkFormatError(f"{path}: {'is a directory' if os.path.isdir(path) else 'no such file'}")
    try:
        with open(path, "rb") as f:
            data = np.load(io.BytesIO(f.read()), allow_pickle=False)
    except OSError as e:
        raise NetworkFormatError(f"{path}: {e.strerror or e}") from None
    except (ValueError, EOFError, zipfile.BadZipFile) as e:
        raise NetworkFormatError(f"{path}: not a my_torch network file ({e})") from None
    if not hasattr(data, "files") or "header" not in data.files:
        raise NetworkFormatError(f"{path}: not a my_torch network file")
    return data


def _header(path, data):
    """The header of the archive, checked: a my_torch file of a version this program reads."""
    try:
        header = json.loads(bytes(data["header"]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise NetworkFormatError(f"{path}: corrupted header ({e})") from None
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        raise NetworkFormatError(f"{path}: not a my_torch network file")
    version = header.get("format_version")
    if not isinstance(version, int) or version < 1:
        raise NetworkFormatError(f"{path}: invalid format version {version!r}")
    if version > FORMAT_VERSION:
        raise NetworkFormatError(f"{path}: format version {version} is newer than this program"
                                 f" (supports up to {FORMAT_VERSION})")
    return upgrade(header)


def upgrade(header):
    """Where headers of older format versions will be converted to the current one (none yet)."""
    return header


def _open(path):
    data = _archive(path)
    return data, _header(path, data)


def read_header(path):
    """Header of a .nn file, without building the network."""
    return _open(path)[1]


def _weights(path, data, net):
    try:
        net.set_weights([(data[f"W{i}"], data[f"b{i}"]) for i in range(len(net.layers))])
    except KeyError as e:
        raise NetworkFormatError(f"{path}: missing weights {e}") from None
    except (ValueError, OSError, zipfile.BadZipFile) as e:
        raise NetworkFormatError(f"{path}: corrupted weights ({e})") from None


def load_network(path):
    """Rebuild a Network from a .nn file. Raises NetworkFormatError if the file is unusable."""
    data, header = _open(path)
    with data:
        try:
            net = Network(header["config"])
        except (KeyError, ConfigError) as e:
            raise NetworkFormatError(f"{path}: invalid network config ({e})") from None
        _weights(path, data, net)
    net.meta.update(header.get("meta") or {})
    net.history = header.get("history") or {}
    return net
