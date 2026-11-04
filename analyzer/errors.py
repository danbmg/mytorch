"""Exit status and error messages of the command line programs.

Every failure is reported on stderr as "program: message" with the exit
status 84 the subject requires, never as a traceback. This is the one place
that defines the status and turns an exception into its message; generator
and the scripts import these two things from here and nothing else.
"""

import sys

EXIT_FAILURE = 84


def log(text):
    """One line on stderr, flushed at once (the training progress is interleaved with it)."""
    print(text, file=sys.stderr, flush=True)


def message(error):
    """The text to report for an exception caught by a command line program."""
    if isinstance(error, MemoryError):
        return "not enough memory for this file"
    if isinstance(error, OSError):
        return f"{error.filename + ': ' if error.filename else ''}{error.strerror or error}"
    return str(error)
