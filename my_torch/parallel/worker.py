"""What one worker process does: find the shared arrays, train one network,
report its metrics at the best epoch (or the error that stopped it).

_DATA and _SHM are the one piece of module state of my_torch: a spawned worker
can only receive its arrays through its initializer, which has nowhere else to
put them. They are filled in the worker (or for the duration of an in-process
run) and never read by anything else.
"""

import time
from multiprocessing import shared_memory

import numpy as np

from ..errors import TrainingError
from ..metrics import confusion_matrix, macro_f1
from ..network import Network

_DATA = {}  # arrays seen by the trainings of this process
_SHM = []


def init_worker(specs):
    """Map the shared blocks {key: (name, shape, dtype)} as numpy arrays in _DATA."""
    for key, (name, shape, dtype) in specs.items():
        try:
            # Python >= 3.13: the parent owns the block, the worker must not unlink it on exit
            shm = shared_memory.SharedMemory(name=name, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(name=name)
        _SHM.append(shm)
        _DATA[key] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)


def _result(index, net, history, X_val, Y_val):
    """Metrics at the best epoch (the weights early stopping restored)."""
    best = net.meta["best_epoch"] if net.meta["best_epoch"] in history["epoch"] else history["epoch"][-1]
    i = history["epoch"].index(best)
    result = {"index": index, "config": net.config, "history": history, "epochs": len(history["epoch"]),
              "best_epoch": best, "train_loss": history["train_loss"][i], "train_acc": history["train_acc"][i],
              "n_params": net.n_params}
    if X_val is not None:
        scores = net.evaluate(X_val, Y_val)
        cm = confusion_matrix(Y_val, net.predict(X_val), net.output_size)
        result.update(val_loss=scores["loss"], val_acc=scores["accuracy"], val_macro_f1=macro_f1(cm))
    return result


def run(task):
    """(index, config, fit kwargs) -> result dict; a failed training gives an "error" key."""
    index, config, fit_kwargs = task
    t0 = time.perf_counter()
    X_val = _DATA.get("X_val")
    Y_val = _DATA.get("Y_val")
    try:
        net = Network(config)
        history = net.fit(_DATA["X"], _DATA["Y"], X_val, Y_val, **fit_kwargs)
        result = _result(index, net, history, X_val, Y_val)
    except (TrainingError, ValueError) as e:
        result = {"index": index, "config": config, "error": str(e)}
    result["time"] = time.perf_counter() - t0
    return result
