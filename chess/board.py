import string
from itertools import zip_longest

import chess

COL_NAMES = string.ascii_lowercase[:8]
ROW_NAMES = range(1, 9)

ALL_SQUARES = [ '{}{}'.format(col, row) for col in COL_NAMES \
                                        for row in ROW_NAMES ]

class SquareOutOfBounds(Exception):
    pass

class MalformedSquareName(Exception):
    pass

################################################################################

def square_name(x_pos, y_pos):
    name = "{}{}".format(COL_NAMES[x_pos], y_pos+1)
    if name not in ALL_SQUARES:
        raise SquareOutOfBounds
    return name

def square_coordinates(square):
    if not len(square) == 2:
        raise MalformedSquareName

    col_name = square[0]
    if col_name not in COL_NAMES:
        raise SquareOutOfBounds
    x_pos = COL_NAMES.index(col_name)

    y_pos = square[1] - 1
    if y_pos >= len(ROW_NAMES) or y_pos < 0:
        raise SquareOutOfBounds

    return x_pos, y_pos


################################################################################

class BlindBoard:
    '''TODO
    '''

    class Diff:
        emptied = []
        filled = []
        changed = []

    @staticmethod
    def diff_board(board_to, board_from):
        return board_to.diff(board_from)

    ########################################

    occupied_squares = dict()

    def __init__(self, occupied_squares=None):
        if occupied_squares is None:
            return
        for square, color in occupied_squares.items():
            # TODO: assert or exceptions?
            assert square in ALL_SQUARES
            assert color  in chess.Color
        self.occupied_squares = occupied_squares

    def __eq__(self, other):
        return self.occupied_squares == other.occupied_squares

    def diff(self, board_from):
        diff = BlindBoard.Diff()
        diff.emptied = \
            [ s for s in board_from.occupied_squares
                    if s not in self.occupied_squares ]

        diff.filled = \
            [ s for s in self.occupied_squares
                    if s not in board_from.occupied_squares ]

        diff.changed = \
            [ s for s in self.occupied_squares
                    if s in board_from.occupied_squares and
                       board_from.occupied_squares[s] ==
                           chess.Color.opposite(self.occupied_squares[s]) ]

        return diff

    def clear(self):
        self.occupied_squares = dict()

    def add_piece(self, square, color):
        assert square in ALL_SQUARES
        assert color  in chess.Color
        self.occupied_squares[square] = color


