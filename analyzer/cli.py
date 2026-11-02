"""Command line of my_torch_analyzer."""

from types import SimpleNamespace

USAGE = """USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
    --train      Launch the neural network in training mode. Each chessboard in CHESSFILE
                 must contain inputs to send to the neural network in FEN notation and the
                 expected output separated by space. If specified, the newly trained neural
                 network will be saved in SAVEFILE. Otherwise, it will be saved in the
                 original LOADFILE (the previous version is kept as LOADFILE.bak).
    --predict    Launch the neural network in prediction mode. Each chessboard in CHESSFILE
                 must contain inputs to send to the neural network in FEN notation, and
                 optionally an expected output.
    --save       Save neural network into SAVEFILE. Only works in train mode.
    --epochs N   Train mode: at most N epochs instead of the number stored in the network.

    LOADFILE     File containing an artificial neural network
    CHESSFILE    File containing chessboards

    --predict writes one line per chessboard, in order, and nothing else on stdout:
    Nothing, Check White, Check Black, Checkmate White or Checkmate Black (the colour
    is the side giving the check). --train shows its progress on stderr and holds out
    10% of CHESSFILE to validate after each epoch. Errors: stderr, exit status 84."""


class UsageError(Exception):
    pass


def parse_args(args):
    """Command line arguments -> namespace (mode, save, epochs, loadfile, chessfile)."""
    opts, files, args = SimpleNamespace(mode=None, save=None, epochs=None), [], list(args)
    while args:
        arg = args.pop(0)
        if arg in ("--predict", "--train"):
            if opts.mode not in (None, arg[2:]):
                raise UsageError("choose either --predict or --train, not both")
            opts.mode = arg[2:]
        elif arg in ("--save", "--epochs"):
            if not args:
                raise UsageError(f"{arg} needs a value (see --help)")
            value = args.pop(0)
            if arg == "--epochs" and not (value.isdigit() and int(value) > 0):
                raise UsageError(f"--epochs: expected a positive integer, got '{value}'")
            setattr(opts, arg[2:], int(value) if arg == "--epochs" else value)
        elif arg.startswith("-") and arg != "-":
            raise UsageError(f"unknown option '{arg}' (see --help)")
        else:
            files.append(arg)
    if opts.mode is None:
        raise UsageError("choose a mode: --predict or --train (see --help)")
    for option in ("save", "epochs"):
        if getattr(opts, option) is not None and opts.mode != "train":
            raise UsageError(f"--{option} only works with --train")
    if len(files) != 2:
        raise UsageError(f"expected LOADFILE and CHESSFILE, got {len(files)} file argument(s) (see --help)")
    opts.loadfile, opts.chessfile = files
    return opts
