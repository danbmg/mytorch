"""Network files (.nn): a numpy .npz archive (zip) holding

- `header`: UTF-8 JSON, readable as is (`unzip -p FILE.nn header.npy`): format
  version, full config (layers, activation of each layer, optimizer, learning
  rate and schedule, regularisation, seed), labels, training metadata and history;
- `W0, b0, W1, b1...`: weights and biases in raw binary float32 (exact, 3x smaller
  than text, fast to load).

Writes are atomic: written next to the target, then renamed over it, so an
interrupted save never leaves a half-written network.
"""

import datetime
import io
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np

from .errors import ConfigError, NetworkFormatError
from .network import Network

FORMAT, FORMAT_VERSION = "my_torch.nn", 1


def save_network(net, path, backup=False):
    """Write `net` to `path` atomically. backup=True keeps the replaced file as `path`.bak."""
    header = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": net.config, "labels": net.labels, "meta": net.meta,
        "layers": [{"name": f"W{i}/b{i}", "inputs": layer.W.shape[0], "outputs": layer.W.shape[1],
                    "activation": layer.activation, "params": layer.n_params} for i, layer in enumerate(net.layers)],
        "history": net.history,
    }
    arrays = {"header": np.frombuffer(json.dumps(header, indent=2).encode("utf-8"), dtype=np.uint8)}
    for i, layer in enumerate(net.layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = layer.W, layer.b
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".nn.tmp", dir=os.path.dirname(os.path.abspath(path)))
    try:
        # a file object, not a name: np.savez("x.nn") would write "x.nn.npz"
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        # mkstemp creates a private file: give it the permissions of the file it replaces, else the usual ones
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o666 & ~umask)
        if backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _open(path):
    if not os.path.isfile(path):
        raise NetworkFormatError(f"{path}: {'is a directory' if os.path.isdir(path) else 'no such file'}")
    try:
        # read in memory first (a few MB): no file left open if the archive is broken
        with open(path, "rb") as f:
            data = np.load(io.BytesIO(f.read()), allow_pickle=False)
    except OSError as e:
        raise NetworkFormatError(f"{path}: {e.strerror or e}") from None
    except (ValueError, EOFError, zipfile.BadZipFile) as e:
        raise NetworkFormatError(f"{path}: not a my_torch network file ({e})") from None
    if not hasattr(data, "files") or "header" not in data.files:
        raise NetworkFormatError(f"{path}: not a my_torch network file")
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
    return data, _upgrade(header)


def _upgrade(header):
    """Where headers of older format versions will be converted to the current one (none yet)."""
    return header


def read_header(path):
    """Header of a .nn file, without building the network."""
    return _open(path)[1]


def load_network(path):
    """Rebuild a Network from a .nn file. Raises NetworkFormatError if the file is unusable."""
    data, header = _open(path)
    with data:
        try:
            net = Network(header["config"])
        except (KeyError, ConfigError) as e:
            raise NetworkFormatError(f"{path}: invalid network config ({e})") from None
        try:
            net.set_weights([(data[f"W{i}"], data[f"b{i}"]) for i in range(len(net.layers))])
        except KeyError as e:
            raise NetworkFormatError(f"{path}: missing weights {e}") from None
        except (ValueError, OSError, zipfile.BadZipFile) as e:
            raise NetworkFormatError(f"{path}: corrupted weights ({e})") from None
    net.meta.update(header.get("meta") or {})
    net.history = header.get("history") or {}
    return net
