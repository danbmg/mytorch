"""Mini-batch training of a Network, apart from the model itself.

batch.py: one optimisation step (forward with dropout, loss, backward, L2,
gradient clipping, optimiser step) and the flush of subnormal floats;
early_stopping.py: the best validation epoch and its weights;
progress.py: the verbose output, on the stream given by the caller;
trainer.py: the epoch loop; fit.py: the entry point behind Network.fit().
"""
