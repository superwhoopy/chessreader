'''Chess Board internal representation module '''

import chess
import utils


################################################################################
# EXCEPTIONS
################################################################################

class SquareOutOfBounds(Exception):
    pass

class MalformedSquareName(Exception):
    pass

################################################################################
# SQUARE REPRESENTATION
################################################################################

# Squares are internally represented by a string with their classical
# coordinate, such as 'a1', 'e2', 'h8', etc. (lower-case letter followed by an
# integer) The following variables define these identifiers.

COL_NAMES = 'abcdefgh'
'''Names of all the columns on a chessboard'''

ROW_NAMES = range(1, 9)
'''Names of all the rows on a chessboard'''

ALL_SQUARES = [ '{}{}'.format(col, row) for col in COL_NAMES \
                                        for row in ROW_NAMES ]
'''Identifiers of all of the 64 squares on a chessboard'''



def square_name(x_pos, y_pos):
    '''Convert square coordinates into a square identifier

    Note: square 'a1' matches coordinates (0,0); other squares follow...

    Args:
        x_pos (int): x-coordinate of the square, in `range(0,7)`
        y_pos (int): y-coordinate of the square, in `range(0,7)`

    Returns:
        string: a square-identifier, member of `ALL_SQUARES`

    Raises:
        SquareOutOfBounds: when (x_pos, y_pos) is outside of the chessboard,
            i.e. if one of the coordinates is not in [0,7].
    '''
    if x_pos not in range(0,8) or y_pos not in range(0,8):
        raise SquareOutOfBounds
    name = "{}{}".format(COL_NAMES[x_pos], y_pos+1)
    return name



def square_coordinates(square):
    '''Convert a square identifier into a pair of coordinates

    Note: square 'a1' matches coordinates (0,0); other squares follow...

    Args:
        square (str): square identifier, must be in `ALL_SQUARES`

    Returns:
        int, int: a pair of (x,y) coordinates in `range(0,7), range(0,7)`
    '''
    if square not in ALL_SQUARES:
        raise MalformedSquareName

    col_name = square[0]
    row_name = square[1]

    x_pos = COL_NAMES.index(col_name)
    y_pos = row_name - 1

    return x_pos, y_pos



################################################################################
# BOARD REPRESENTATION
################################################################################

class BlindBoard:
    '''Semi-blind chessboard representation

    A "blind board" partially tracks the state of a chessboard:
    '''

    class Diff:
        emptied = {}
        filled  = {}
        changed = {}

        def get_single_emptied(self):
            assert len(self.emptied) == 1
            return next(iter(self.emptied))

        def get_single_filled(self):
            assert len(self.filled) == 1
            return next(iter(self.filled))

        def get_single_changed(self):
            assert len(self.changed) == 1
            return next(iter(self.changed))

        def __init__(self, emptied, filled, changed):
            self.emptied = emptied
            self.filled = filled
            self.changed = changed

        def __eq__(self, other):
            return self.emptied == other.emptied and \
                   self.filled  == other.filled and \
                   self.changed == other.changed

        def __str__(self):
            return "emp:{} fill:{} chgd:{}".format(str(self.emptied),
                                                   str(self.filled),
                                                   str(self.changed))

        def length(self):
            return [len(self.emptied), len(self.filled), len(self.changed)]


    @staticmethod
    def diff_board(board_to, board_from):
        return board_to.diff(board_from)

    ########################################

    occupied_squares = dict()

    def __init__(self, occupied_squares=None):
        utils.log.debug('create BlindBoard with {}'.format(occupied_squares))
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
        emptied = \
            { s for s in board_from.occupied_squares
                    if s not in self.occupied_squares }

        filled = \
            { s for s in self.occupied_squares
                    if s not in board_from.occupied_squares }

        changed = \
            { s for s in self.occupied_squares
                    if s in board_from.occupied_squares and
                       board_from.occupied_squares[s] ==
                           chess.Color.opposite(self.occupied_squares[s]) }

        return chess.board.BlindBoard.Diff(emptied, filled, changed)

    def clear(self):
        self.occupied_squares = dict()

    def add_piece(self, square, color):
        assert square in ALL_SQUARES
        assert color  in chess.Color
        self.occupied_squares[square] = color


