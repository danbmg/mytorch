"""Chess positions in FEN notation: parsing, validation, encoding as network inputs.

parse.py checks a FEN field by field; lines.py cuts a '<FEN> [label]' line and
gives a position its canonical key; encode.py turns a position into the 773
inputs. The names below are the ones the rest of the project uses.
"""

from .encode import CASTLING, INPUT_SIZE, active_inputs, encode_many, fen_to_vector
from .lines import parse_line, position_key, split_line
from .parse import PIECES, FenError, Position, parse_fen

__all__ = ["CASTLING", "INPUT_SIZE", "PIECES", "FenError", "Position", "active_inputs", "encode_many",
           "fen_to_vector", "parse_fen", "parse_line", "position_key", "split_line"]
