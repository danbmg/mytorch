"""The verbose output of a training: one line per epoch, a percentage during the
epoch when the output is a terminal, and the early stopping messages.

This is the only place in my_torch that writes anything, and it writes on the
stream the caller gives (my_torch_analyzer passes stderr): the module itself
never touches stdout, so the predictions of the analyzer stay clean.
"""


class Progress:
    def __init__(self, log, enabled):
        self.log = log
        self.enabled = enabled
        self.tty = enabled and getattr(log, "isatty", lambda: False)()

    def message(self, text):
        if self.enabled:
            print(text, file=self.log)

    def batch(self, epoch, b, n_batches):
        """Percentage of the epoch done, rewritten on the same terminal line."""
        if self.tty and b % max(1, n_batches // 50) == 0:
            print(f"\r\033[Kepoch {epoch}: {100 * b // n_batches}%", end="", file=self.log, flush=True)

    def epoch(self, epoch, last_epoch, train, val, lr, seconds):
        """The line of one epoch: loss and accuracy on train, on validation if any, learning rate, time."""
        if not self.enabled:
            return
        line = f"epoch {epoch:>3}/{last_epoch}  loss {train['loss']:.4f}  acc {100 * train['accuracy']:6.2f}%"
        if val["loss"] is not None:
            line += f"  val_loss {val['loss']:.4f}  val_acc {100 * val['accuracy']:6.2f}%"
        line += f"  lr {lr:.2e}  {seconds:.1f}s"
        if self.tty:
            line = "\r\033[K" + line  # over the percentage of batch()
        print(line, file=self.log, flush=True)
