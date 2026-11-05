"""my_torch_analyzer: state of chessboards (check, checkmate...) predicted by a my_torch network.

main() only routes: --help, the mode, and every error turned into one line on
stderr with the exit status 84. A broken pipe (| head) is not an error, and a
Ctrl-C leaves no half-written network.
"""

import os
import sys

from my_torch import TrainingError

from .args import UsageError, parse_args
from .errors import EXIT_FAILURE, log, message
from .modes import predict, train
from .usage import USAGE


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if "-h" in args or "--help" in args:
        print(USAGE)
        return 0
    try:
        opts = parse_args(args)
        if opts.mode == "predict":
            predict(opts)
        else:
            train(opts)
        return 0
    except BrokenPipeError:
        # output piped into a command that stopped reading (| head): not an error
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (UsageError, ValueError, TrainingError, OSError, MemoryError) as e:
        log(f"my_torch_analyzer: {message(e)}")
    except KeyboardInterrupt:
        log("\nmy_torch_analyzer: interrupted, no network written")
    return EXIT_FAILURE
