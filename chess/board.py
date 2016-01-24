'''Chess Board internal representation module

Squares are identified by a string with their usual name, from 'a1' to 'h8'. The
global `ALL_SQUARES` is a list of all of the square names.

The module also provides a `BlindBoard` class used to represent a board where
pieces are only distinguished by their color.
'''

import chess
import utils
import utils.log

################################################################################
# EXCEPTIONS
################################################################################

class SquareOutOfBounds(Exception):
    '''Raised when attempting to access a square by its coordinates'''
    pass

class MalformedSquareName(Exception):
    '''Raised when accessing a square with a wrong id'''
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
    y_pos = int(row_name) - 1

    return x_pos, y_pos



################################################################################
# BOARD REPRESENTATION
################################################################################



class BlindBoard:
    '''Semi-blind chessboard representation

    A blind board "sees" pieces and their color, but not their type; basically
    it only stores which squares are occupied, and the color of the piece
    standing on it.
    '''

    class Diff:
        '''Diff between two BlindBoard'''

        emptied = {}
        '''Set of squares that were emptied'''

        filled  = {}
        '''Set of squares that were filled'''

        changed = {}
        '''Set of squares that were occuppied and still are, but whose piece
        color has changed'''

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
            '''Get the size of the diff

            Returns: the size of the three sets `emptied`, `filled` and
               `changed`
            '''
            return len(self.emptied), len(self.filled), len(self.changed)


    @staticmethod
    def diff_board(board_to, board_from):
        '''Diff two BlindBoards

        Note: this static method is an alias for `board_to.diff(board_from)`.

        Params:
            board_to    (BlindBoard): ending position
            board_from  (BlindBoard): starting position

        Returns (BlindBoard.Diff): the diff describinig the changes from
            `board_from` to `board_to`
        '''
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


def build_start_pos_blind_board():
    'Return a blind board for the starting position'
    filled = {}
    for col in 'abcdefgh':
        for row in '12':
            square = '{}{}'.format(col,row)
            filled[square] = chess.Color.WHITE
        for row in '78':
            square = '{}{}'.format(col,row)
            filled[square] = chess.Color.BLACK
    return BlindBoard(filled)

BLIND_EMPTY = BlindBoard()
'''Empty board representation'''

BLIND_START = build_start_pos_blind_board()
'''Board with pieces in starting position'''

