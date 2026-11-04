"""Initial weights. Goal: keep the variance of the signal about constant from one
layer to the next, so that it neither vanishes nor explodes. Biases start at 0.

A registry name -> function; "auto" picks the initialisation each activation was
designed with, so a configuration only has to say it when it wants another one.
"""

import numpy as np


def he(n_in, n_out, rng):
    # Var = 2 / n_in: compensates the half of the signal that ReLU sets to 0
    return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))


def xavier(n_in, n_out, rng):
    # Var = 2 / (n_in + n_out): for symmetric activations (sigmoid, tanh, softmax)
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))


def lecun(n_in, n_out, rng):
    return rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out))


def uniform(n_in, n_out, rng):
    # naive baseline, kept for comparison
    return rng.uniform(-0.05, 0.05, size=(n_in, n_out))


INITS = {"he": he, "xavier": xavier, "lecun": lecun, "uniform": uniform}
# "auto": the initialisation each activation was designed with (default: xavier)
AUTO = {"relu": "he", "leaky_relu": "he", "elu": "he", "linear": "lecun"}


def resolve(init, activation):
    return AUTO.get(activation, "xavier") if init == "auto" else init
