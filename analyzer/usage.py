"""The --help text of my_torch_analyzer, in the layout of the subject (USAGE,
DESCRIPTION), plus what the subject leaves unsaid: the exact output format of
--predict, the validation split of --train and the exit status of errors."""

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
