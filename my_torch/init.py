"""Weight initialisation strategies.

The goal is to keep the variance of the signal roughly constant from one layer
to the next, so that it neither vanishes nor explodes in deep networks:

- He (Kaiming): Var(W) = 2 / n_in, for ReLU-like activations (they zero half the signal)
- Xavier (Glorot): Var(W) = 2 / (n_in + n_out), for sigmoid / tanh / softmax
- LeCun: Var(W) = 1 / n_in, for linear outputs
- uniform: U(-scale, scale), naive baseline kept for comparison

Biases always start at 0.
"""

import numpy as np


def he(n_in, n_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))


def xavier(n_in, n_out, rng):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))


def lecun(n_in, n_out, rng):
    return rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out))


def uniform(n_in, n_out, rng, scale=0.05):
    return rng.uniform(-scale, scale, size=(n_in, n_out))


INITS = {"he": he, "xavier": xavier, "lecun": lecun, "uniform": uniform}

# "auto": the initialisation each activation was designed with
AUTO = {
    "relu": "he",
    "leaky_relu": "he",
    "elu": "he",
    "sigmoid": "xavier",
    "tanh": "xavier",
    "softmax": "xavier",
    "linear": "lecun",
}


def resolve(name, activation):
    """Name of the initialisation actually used (resolves "auto")."""
    return AUTO.get(activation, "xavier") if name == "auto" else name


def init_weights(name, n_in, n_out, rng):
    if name not in INITS:
        raise ValueError(f"unknown init '{name}' (choices: auto, {', '.join(INITS)})")
    return INITS[name](n_in, n_out, rng)
