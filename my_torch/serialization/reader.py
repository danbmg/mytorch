"""Reading a .nn file: the archive is checked, then the header (format and
version), then every weight array is read in memory.

Every failure is a NetworkFormatError naming the file and what is wrong, so
the analyzer can report it without a traceback. A file of a newer format
version than this program is refused rather than misread; older versions
would be converted in upgrade() (none yet). This module only reads files:
Network.load() builds the network from what it returns.
"""

import io
import json
import os
import zipfile

import numpy as np

from ..errors import NetworkFormatError
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


def read_header(path):
    """Header of a .nn file, without reading the weights."""
    with _archive(path) as data:
        return _header(path, data)


def read_network_file(path):
    """(header, arrays) of a .nn file: the checked header and {"W0": ..., "b0": ...}."""
    with _archive(path) as data:
        header = _header(path, data)
        try:
            arrays = {name: data[name] for name in data.files if name != "header"}
        except (ValueError, EOFError, OSError, zipfile.BadZipFile) as e:
            raise NetworkFormatError(f"{path}: corrupted weights ({e})") from None
    return header, arrays
