"""Writing a .nn file: the JSON header, the weight arrays, and an atomic replace.

np.savez gets a file object, not a name: given "x.nn" it would write "x.nn.npz".
The temporary file takes the permissions of the file it replaces, so a shared
network stays shared; backup=True keeps the replaced file as `path`.bak.
"""

import datetime
import json
import os
import shutil
import tempfile

import numpy as np

FORMAT = "my_torch.nn"  # written in every header: a file without it is not a network
FORMAT_VERSION = 1  # +1 at every change of the layout, so that old files can be converted (reader.upgrade)


def header(net):
    """Everything about the network but the weights, as a JSON-ready dict."""
    return {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": net.config, "labels": net.labels, "meta": net.meta,
        "layers": [layer_summary(i, layer) for i, layer in enumerate(net.layers)],
        "history": net.history,
    }


def layer_summary(i, layer):
    """What the header says about layer i: its arrays, its size and its activation."""
    return {"name": f"W{i}/b{i}", "inputs": layer.W.shape[0], "outputs": layer.W.shape[1],
            "activation": layer.activation, "params": layer.n_params}


def arrays(net):
    """The .npz members: the header as bytes, then W0, b0, W1, b1..."""
    out = {"header": np.frombuffer(json.dumps(header(net), indent=2).encode("utf-8"), dtype=np.uint8)}
    for i, layer in enumerate(net.layers):
        out[f"W{i}"] = layer.W
        out[f"b{i}"] = layer.b
    return out


def _replace(tmp, path, backup):
    """Give tmp the permissions of path (else the usual ones), keep a backup if asked, rename over path."""
    # mkstemp creates a private file: give it the permissions of the file it replaces, else the usual ones
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o666 & ~umask)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)


def save_network(net, path, backup=False):
    """Write `net` to `path` atomically. backup=True keeps the replaced file as `path`.bak."""
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".nn.tmp", dir=os.path.dirname(os.path.abspath(path)))
    try:
        # a file object, not a name: np.savez("x.nn") would write "x.nn.npz"
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays(net))
        _replace(tmp, path, backup)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
