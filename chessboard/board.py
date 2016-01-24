'''Chess Board internal representation module

Squares are identified by a string with their usual name, from 'a1' to 'h8'. The
global `ALL_SQUARES` is a list of all of the square names.

The module also provides a `BlindBoard` class used to represent a board where
pieces are only distinguished by their color.
'''
import chess
from chess import BaseBoard

from .. import utils


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


WHITE_START_SQUARES = range(16)
BLACK_START_SQUARES = range(48, 64)


################################################################################
# BOARD REPRESENTATION
################################################################################


# TODO BlindBoard could inherit from chess.SquareSet
# a SquareSet is a binary mask representing which squares are occpuied on a board
# we could override the constructor and add a second binary mask indicating the colors
# in fact a blindboard is exactly the same as the occupied_co attribute on BaseBoard objects

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
            assert square in chess.SQUARES
            assert color  in chess.COLORS
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
            {s for s in self.occupied_squares
             if s in board_from.occupied_squares and
             board_from.occupied_squares[s] == (not self.occupied_squares[s])
            }

        return BlindBoard.Diff(emptied, filled, changed)

    def clear(self):
        self.occupied_squares = dict()

    def add_piece(self, square, color):
        assert square in chess.SQUARES
        assert color  in chess.COLORS
        self.occupied_squares[square] = color

    @staticmethod
    def from_board(board):
        # create a BlindBoard from a chess.Board object
        occupied_squares = {}
        board.occupied_co[chess.COLORS.WHITE]

def build_start_pos_blind_board():
    'Return a blind board for the starting position'
    filled = {white_square: chess.WHITE for white_square in WHITE_START_SQUARES}
    filled.update({black_square: chess.BLACK for black_square in BLACK_START_SQUARES})

    return BlindBoard(filled)

EMPTY_BLINDBOARD = BlindBoard()
'''Empty board representation'''

START_BLINDBOARD = build_start_pos_blind_board()
'''Board with pieces in starting position'''

