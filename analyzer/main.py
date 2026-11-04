"""my_torch_analyzer: state of chessboards (check, checkmate...) predicted by a my_torch network.

main() only routes: --help, the mode, and every error turned into one line on
stderr with the exit status 84. A broken pipe (| head) is not an error, and a
Ctrl-C leaves no half-written network.
"""

import os
import sys

# one training at a time: 2 BLAS threads suits both our machines (docs/benchmarks.md); MY_TORCH_THREADS overrides
os.environ.setdefault("MY_TORCH_THREADS", "2")

from my_torch import TrainingError  # noqa: E402

from .args import UsageError, parse_args  # noqa: E402
from .errors import EXIT_FAILURE, log, message  # noqa: E402
from .modes import predict, train  # noqa: E402
from .usage import USAGE  # noqa: E402


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if "-h" in args or "--help" in args:
        print(USAGE)
        return 0
    try:
        opts = parse_args(args)
        (predict if opts.mode == "predict" else train)(opts)
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
