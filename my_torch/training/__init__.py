"""Mini-batch training of a Network, apart from the model itself.

trainer.py: fit(), behind Network.fit(), and the epoch loop;
batch.py: one optimization step (forward with dropout, loss, backward, L2,
gradient clipping, optimizer step) and the flush of subnormal floats;
early_stopping.py: the best validation epoch and its weights;
progress.py: the verbose output, on the stream given by the caller.
"""
