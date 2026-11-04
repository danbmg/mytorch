"""Train several networks at the same time, one process per training.

The trainings of a hyperparameter search are independent: they run in separate
processes ("spawn": fresh interpreters, same behaviour on every Python version)
with no communication. The data is put once in shared memory, so every worker
reads the same bytes instead of receiving its own copy (440 MB for the chess
training set). Each worker uses `threads_per_worker` BLAS threads (default 1).
worker.py is what runs in each process.
"""

import contextlib
import multiprocessing as mp
import os
from multiprocessing import shared_memory

import numpy as np

from . import worker

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@contextlib.contextmanager
def _shared(arrays):
    """The arrays copied once into shared memory -> {key: (block name, shape, dtype)}; freed at exit."""
    blocks, specs = [], {}
    try:
        for key, array in arrays.items():
            array = np.ascontiguousarray(array)
            shm = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
            blocks.append(shm)
            np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)[...] = array
            specs[key] = (shm.name, array.shape, array.dtype.str)
        yield specs
    finally:
        for shm in blocks:
            shm.close()
            shm.unlink()


@contextlib.contextmanager
def _blas_threads(n):
    """OPENBLAS_NUM_THREADS etc. set to n (read by numpy when a new worker imports it), restored at exit."""
    saved = {var: os.environ.get(var) for var in THREAD_VARS}
    try:
        os.environ.update({var: str(n) for var in THREAD_VARS})
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def _in_process(tasks, arrays, collect):
    worker._DATA.update(arrays)
    try:
        for task in tasks:
            collect(worker.run(task))
    finally:
        worker._DATA.clear()


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
        _in_process(tasks, arrays, collect)
        return results
    with _shared(arrays) as specs, _blas_threads(threads_per_worker):
        workers = min(workers or os.cpu_count() or 1, len(tasks))
        with mp.get_context("spawn").Pool(workers, initializer=worker.init_worker, initargs=(specs,)) as pool:
            for result in pool.imap_unordered(worker.run, tasks):
                collect(result)
    return results
