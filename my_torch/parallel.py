"""Train several networks at the same time, one process per CPU core.

The trainings of a hyperparameter search are independent from each other, so
they run in separate processes with no communication. The datasets are put
once in shared memory: every worker reads the same bytes instead of receiving
its own copy (a 570 000 x 773 uint8 training set weighs 440 MB).

Each worker uses `threads_per_worker` BLAS threads (default 1), so that
`workers` x `threads_per_worker` stays within the number of cores.
"""

import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np

from .metrics import confusion_matrix, macro_f1
from .network import Network, TrainingError

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

# arrays of the current worker, attached by _init_worker
_DATA = {}
_SHM = []


class SharedArray:
    """Copy of a numpy array in a named shared memory block."""

    def __init__(self, array):
        array = np.ascontiguousarray(array)
        self.shm = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
        np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)[...] = array
        self.spec = (self.shm.name, array.shape, array.dtype.str)

    def close(self):
        self.shm.close()
        self.shm.unlink()


def _attach(spec):
    name, shape, dtype = spec
    try:
        # Python >= 3.13: the parent owns the block, the worker must not unlink it on exit
        shm = shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
    return shm, np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)


def _init_worker(specs):
    for key, spec in specs.items():
        shm, array = _attach(spec)
        _SHM.append(shm)
        _DATA[key] = array


def _summary(net, history, X_val, Y_val):
    """Metrics of a trained network, taken at its best epoch."""
    epochs = history["epoch"]
    best = net.meta["best_epoch"] if net.meta["best_epoch"] in epochs else epochs[-1]
    i = epochs.index(best)
    result = {
        "epochs": len(epochs),
        "best_epoch": best,
        "train_loss": history["train_loss"][i],
        "train_acc": history["train_acc"][i],
        "n_params": net.n_params,
    }
    if X_val is not None:
        y_true = net._decide(net._targets(Y_val, len(X_val)))
        cm = confusion_matrix(y_true, net.predict(X_val), max(net.output_size, 2))
        scores = net.evaluate(X_val, Y_val)
        result.update(val_loss=scores["loss"], val_acc=scores["accuracy"],
                      val_macro_f1=macro_f1(cm), confusion=cm.tolist())
    return result


def _run(task):
    index, config, fit_kwargs, save_path = task
    t0 = time.perf_counter()
    try:
        net = Network(config)
        X_val, Y_val = _DATA.get("X_val"), _DATA.get("Y_val")
        history = net.fit(_DATA["X"], _DATA["Y"], X_val, Y_val, **fit_kwargs)
        result = {"index": index, "config": net.config, "history": history,
                  **_summary(net, history, X_val, Y_val)}
        if save_path:
            net.save(save_path)
    except (TrainingError, ValueError) as e:
        result = {"index": index, "config": config, "error": str(e)}
    result["time"] = time.perf_counter() - t0
    result["pid"] = os.getpid()
    return result


def train_many(configs, X, Y, X_val=None, Y_val=None, workers=None, threads_per_worker=1,
               fit_kwargs=None, save_paths=None, on_result=None):
    """Train one network per config and return one result dict per config, in order.

    workers: number of processes (None = one per core, 0 = in this process, one after
    the other). A training that fails (diverging loss, invalid config) gives a result
    with an "error" key instead of stopping the others. on_result(result) is called as
    soon as each training finishes.
    """
    fit_kwargs = dict(fit_kwargs or {})
    tasks = [(i, cfg, fit_kwargs, save_paths[i] if save_paths else None) for i, cfg in enumerate(configs)]
    arrays = {"X": X, "Y": Y}
    if X_val is not None:
        arrays.update(X_val=X_val, Y_val=Y_val)
    results = [None] * len(tasks)
    if not tasks:
        return results

    if workers == 0:
        _DATA.clear()
        _DATA.update(arrays)
        try:
            for task in tasks:
                result = _run(task)
                results[result["index"]] = result
                if on_result:
                    on_result(result)
        finally:
            _DATA.clear()
        return results

    workers = min(workers or os.cpu_count() or 1, len(tasks))
    shared = {}
    saved_env = {var: os.environ.get(var) for var in THREAD_VARS}
    try:
        for key, array in arrays.items():
            shared[key] = SharedArray(np.asarray(array))
        # read by numpy when it is imported in each new worker process
        for var in THREAD_VARS:
            os.environ[var] = str(threads_per_worker)
        # "spawn": fresh interpreters, so the thread setting above applies to them
        ctx = mp.get_context("spawn")
        specs = {key: s.spec for key, s in shared.items()}
        with ctx.Pool(workers, initializer=_init_worker, initargs=(specs,)) as pool:
            for result in pool.imap_unordered(_run, tasks):
                results[result["index"]] = result
                if on_result:
                    on_result(result)
    finally:
        for var, value in saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        for s in shared.values():
            s.close()
    return results
