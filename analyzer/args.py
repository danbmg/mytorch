"""Command line arguments of my_torch_analyzer, parsed by hand: the subject's
syntax has a mode, two options and two positional files, and argparse would
print its own usage and exit with status 2, where the subject wants our USAGE
and the status 84. Every mistake raises UsageError with what to fix."""

from types import SimpleNamespace


class UsageError(Exception):
    pass


def _mode(opts, arg):
    if opts.mode not in (None, arg[2:]):
        raise UsageError("choose either --predict or --train, not both")
    opts.mode = arg[2:]


def _value(opts, arg, args):
    """--save SAVEFILE or --epochs N: the value is popped from the remaining args."""
    if not args:
        raise UsageError(f"{arg} needs a value (see --help)")
    value = args.pop(0)
    if arg == "--epochs" and not (value.isdigit() and int(value) > 0):
        raise UsageError(f"--epochs: expected a positive integer, got '{value}'")
    setattr(opts, arg[2:], int(value) if arg == "--epochs" else value)


def _check(opts, files):
    if opts.mode is None:
        raise UsageError("choose a mode: --predict or --train (see --help)")
    for option in ("save", "epochs"):
        if getattr(opts, option) is not None and opts.mode != "train":
            raise UsageError(f"--{option} only works with --train")
    if len(files) != 2:
        raise UsageError(f"expected LOADFILE and CHESSFILE, got {len(files)} file argument(s) (see --help)")


def parse_args(args):
    """Command line arguments -> namespace (mode, save, epochs, loadfile, chessfile)."""
    opts, files, args = SimpleNamespace(mode=None, save=None, epochs=None), [], list(args)
    while args:
        arg = args.pop(0)
        if arg in ("--predict", "--train"):
            _mode(opts, arg)
        elif arg in ("--save", "--epochs"):
            _value(opts, arg, args)
        elif arg.startswith("-") and arg != "-":
            raise UsageError(f"unknown option '{arg}' (see --help)")
        else:
            files.append(arg)
    _check(opts, files)
    opts.loadfile, opts.chessfile = files
    return opts
