"""Entry point of a training: the settings of this call, the data checked, then
the epoch loop of trainer.py. Kept apart from the loop so that Network.fit()
reads as three steps: settings, data, run."""

import sys
import time

import numpy as np

from ..config import TRAINING, normalize_training
from ..data import inputs, targets
from .progress import Progress
from .trainer import Training


def _settings(net, hp):
    """config["training"] overridden by the keyword arguments of this call."""
    unknown = sorted(set(hp) - set(TRAINING))
    if unknown:
        raise ValueError(f"unknown training setting(s): {', '.join(unknown)}")
    return normalize_training({**net.config["training"], **hp})


def _validation(net, X_val, Y_val):
    """(X_val, Y_val) checked and encoded, or (None, None) when there is no validation data."""
    if X_val is None or len(X_val) == 0:
        return None, None
    X_val = inputs(net, X_val)
    return X_val, targets(net, Y_val, len(X_val))


def fit(net, X, Y, X_val=None, Y_val=None, verbose=False, log=None, callback=None, **hp):
    """Train net on (X, Y), validating on (X_val, Y_val) after every epoch -> history of this call."""
    progress, cfg = Progress(log or sys.stderr, verbose), _settings(net, hp)
    X = inputs(net, X)
    Y = targets(net, Y, len(X))
    if len(X) == 0:
        raise ValueError("no training data")
    X_val, Y_val = _validation(net, X_val, Y_val)
    if cfg["early_stopping"] and X_val is None:
        progress.message("early stopping needs validation data: disabled")
    training, t_fit = Training(net, cfg, X, Y, X_val, Y_val, progress, callback), time.perf_counter()
    # a diverging training overflows: reported as a TrainingError, not as numpy warnings
    with np.errstate(over="ignore", invalid="ignore"):
        stopped = training.run()
    training.finish(stopped, time.perf_counter() - t_fit)
    return training.history
