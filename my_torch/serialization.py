"""Network files (.nn).

A .nn file is a numpy .npz archive (a zip) holding:
- `header`: UTF-8 JSON, readable as is (`unzip -p FILE.nn header.npy`): format
  version, full config (layers, activation of each layer, init, optimizer,
  learning rate and schedule, regularisation, seed), labels, training metadata
  and history;
- `W0, b0, W1, b1, ...`: weights and biases of each layer, raw binary
  float32/float64 (exact values, compact, fast to load).

Writes are atomic: the file is written next to its target then renamed over
it, so an interrupted save never leaves a half-written network behind.
"""

import datetime
import io
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np

from .config import ConfigError
from .network import Network

FORMAT = "my_torch.nn"
FORMAT_VERSION = 1


class NetworkFormatError(ValueError):
    pass


def _header(net):
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": net.config,
        "labels": net.labels,
        "meta": net.meta,
        "layers": [
            {"name": f"W{i}/b{i}", "inputs": layer.n_in, "outputs": layer.n_out,
             "activation": spec["activation"], "params": layer.n_params}
            for i, (layer, spec) in enumerate(zip(net.layers, net.config["layers"]))
        ],
        "history": net.history,
    }


def _file_mode(path):
    """Permissions for the new file: those of the file it replaces, else the umask default."""
    if os.path.exists(path):
        return os.stat(path).st_mode & 0o777
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def save_network(net, path, backup=False):
    """Write `net` to `path` atomically. backup=True: keep the old file as `path`.bak."""
    header = json.dumps(_header(net), indent=2).encode("utf-8")
    arrays = {"header": np.frombuffer(header, dtype=np.uint8)}
    for i, layer in enumerate(net.layers):
        arrays[f"W{i}"] = layer.W
        arrays[f"b{i}"] = layer.b
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".nn.tmp", dir=directory)
    try:
        # a file object, not a name: np.savez("x.nn") would write "x.nn.npz"
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        os.chmod(tmp, _file_mode(path))
        if backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_header(path):
    """Header of a .nn file, without building the network."""
    with _open(path) as data:
        return _parse_header(path, data)


def _open(path):
    if not os.path.exists(path):
        raise NetworkFormatError(f"{path}: no such file")
    if os.path.isdir(path):
        raise NetworkFormatError(f"{path}: is a directory")
    try:
        # read in memory first (about 1 MB): no file left open when the archive is broken
        with open(path, "rb") as f:
            raw = io.BytesIO(f.read())
    except OSError as e:
        raise NetworkFormatError(f"{path}: {e.strerror}") from None
    try:
        data = np.load(raw, allow_pickle=False)
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as e:
        raise NetworkFormatError(f"{path}: not a my_torch network file ({e})") from None
    if not hasattr(data, "files"):
        # a plain .npy array, not an archive
        raise NetworkFormatError(f"{path}: not a my_torch network file")
    return data


def _parse_header(path, data):
    if "header" not in data.files:
        raise NetworkFormatError(f"{path}: not a my_torch network file (no header)")
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
    return _upgrade(header)


def _upgrade(header):
    """Convert headers of older format versions to the current one (none yet)."""
    return header


def load_network(path):
    """Rebuild a Network from a .nn file. Raises NetworkFormatError if the file is unusable."""
    data = _open(path)
    with data:
        header = _parse_header(path, data)
        try:
            net = Network(header["config"])
        except (KeyError, ConfigError) as e:
            raise NetworkFormatError(f"{path}: invalid network config ({e})") from None
        weights = []
        for i in range(len(net.layers)):
            try:
                weights.append((data[f"W{i}"], data[f"b{i}"]))
            except KeyError:
                raise NetworkFormatError(f"{path}: missing weights of layer {i}") from None
            except (ValueError, OSError, zipfile.BadZipFile) as e:
                raise NetworkFormatError(f"{path}: corrupted weights of layer {i} ({e})") from None
        try:
            net.set_weights(weights)
        except ValueError as e:
            raise NetworkFormatError(f"{path}: {e}") from None
    net.meta.update(header.get("meta") or {})
    net.history = header.get("history") or {}
    return net
