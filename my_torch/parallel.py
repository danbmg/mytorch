"""Train several networks at the same time, one process per training.

The trainings of a hyperparameter search are independent: they run in separate
processes ("spawn": fresh interpreters, same behaviour on every Python version)
with no communication. The data is put once in shared memory, so every worker
reads the same bytes instead of receiving its own copy (440 MB for the chess
training set). Each worker uses `threads_per_worker` BLAS threads (default 1).
"""

import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np

from .metrics import confusion_matrix, macro_f1
from .errors import TrainingError
from .network import Network

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_DATA = {}  # arrays seen by the trainings of this process
_SHM = []


def _init_worker(specs):
    for key, (name, shape, dtype) in specs.items():
        try:
            # Python >= 3.13: the parent owns the block, the worker must not unlink it on exit
            shm = shared_memory.SharedMemory(name=name, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(name=name)
        _SHM.append(shm)
        _DATA[key] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)


def _run(task):
    index, config, fit_kwargs = task
    t0 = time.perf_counter()
    X_val, Y_val = _DATA.get("X_val"), _DATA.get("Y_val")
    try:
        net = Network(config)
        history = net.fit(_DATA["X"], _DATA["Y"], X_val, Y_val, **fit_kwargs)
        # metrics at the best epoch (the weights early stopping restored)
        best = net.meta["best_epoch"] if net.meta["best_epoch"] in history["epoch"] else history["epoch"][-1]
        i = history["epoch"].index(best)
        result = {"index": index, "config": net.config, "history": history, "epochs": len(history["epoch"]),
                  "best_epoch": best, "train_loss": history["train_loss"][i], "train_acc": history["train_acc"][i],
                  "n_params": net.n_params}
        if X_val is not None:
            scores = net.evaluate(X_val, Y_val)
            cm = confusion_matrix(Y_val, net.predict(X_val), net.output_size)
            result.update(val_loss=scores["loss"], val_acc=scores["accuracy"], val_macro_f1=macro_f1(cm))
    except (TrainingError, ValueError) as e:
        result = {"index": index, "config": config, "error": str(e)}
    result["time"] = time.perf_counter() - t0
    return result


def train_many(configs, X, Y, X_val=None, Y_val=None, workers=None, threads_per_worker=1,
               fit_kwargs=None, on_result=None):
    """Train one network per config; one result dict per config, in order.

    workers: number of processes (None: one per core, 0: one after the other in this
    process). A failed training (diverging loss, invalid config) gives a result with
    an "error" key without stopping the others. on_result(result) is called as soon
    as each training ends. Y_val: class indices.
    """
    tasks = [(i, cfg, dict(fit_kwargs or {})) for i, cfg in enumerate(configs)]
    arrays = {"X": X, "Y": Y, **({"X_val": X_val, "Y_val": Y_val} if X_val is not None else {})}
    results = [None] * len(tasks)

    def collect(result):
        results[result["index"]] = result
        if on_result:
            on_result(result)

    if workers == 0 or not tasks:
        _DATA.update(arrays)
        try:
            for task in tasks:
                collect(_run(task))
        finally:
            _DATA.clear()
        return results

    blocks, saved_env = [], {var: os.environ.get(var) for var in THREAD_VARS}
    try:
        specs = {}
        for key, array in arrays.items():
            array = np.ascontiguousarray(array)
            shm = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
            blocks.append(shm)
            np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)[...] = array
            specs[key] = (shm.name, array.shape, array.dtype.str)
        # read by numpy when it is imported in each new worker
        os.environ.update({var: str(threads_per_worker) for var in THREAD_VARS})
        workers = min(workers or os.cpu_count() or 1, len(tasks))
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(specs,)) as pool:
            for result in pool.imap_unordered(_run, tasks):
                collect(result)
    finally:
        for var, value in saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        for shm in blocks:
            shm.close()
            shm.unlink()
    return results
