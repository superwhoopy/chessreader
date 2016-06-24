'''Chess Board internal representation module

Squares are identified by a string with their usual name, from 'a1' to 'h8'. The
global `ALL_SQUARES` is a list of all of the square names.

The module also provides a `BlindBoard` class used to represent a board where
pieces are only distinguished by their color.
'''

from string import ascii_lowercase, ascii_uppercase

import chess
from chess import (BaseBoard, Piece, STARTING_BOARD_FEN, BLACK,
                   WHITE, PAWN, SQUARE_NAMES, BB_VOID, BB_ALL)



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


class BlindBoard(BaseBoard):
    '''
    Semi-blind chessboard representation

    A blind board "sees" pieces and their color, but not their type; basically
    it only stores which squares are occupied, and the color of the piece
    standing on it.

    We build it as a child class of `BaseBoard`.
    '''

    def __init__(self, fen=None):
        # create an empty board by default
        # (whereas the default for BaseBoard is a board in starting position)
        BaseBoard.__init__(self, fen)

    def __eq__(self, other):
        '''
        Two blindboards are identical if the positions of their black and white
        pieces are the same
        '''
        return all(self.occupied_co[color] == other.occupied_co[color]
                   for color in (BLACK, WHITE))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        def obfusc(char):
            return 'W' if char in ascii_uppercase else \
                   'b' if char in ascii_lowercase else \
                   char

        return ''.join( obfusc(char) for char in BaseBoard.__str__(self) )

    def set_piece_at(self, square, piece):
        '''
        In `BaseBoard`, this method expects a square and a Piece object.
        But for BlindBoards, we only need the second argument to be a color
        (we don't need the piece type).
        '''
        if piece in chess.COLORS:
            piece = Piece(PAWN, piece)
        elif not isinstance(piece, Piece):
            raise ValueError("`%s` is neither a `bool` nor a Piece object"
                             % str(piece))
        return BaseBoard.set_piece_at(self, square, piece)

    def move_piece(self, from_square, to_square):
        bb_from_square = 1 << from_square
        if not self.occupied & bb_from_square:
            raise ValueError("Starting square %d is empty" % from_square)
        color = self.occupied_co[WHITE] & bb_from_square > 0
        self.remove_piece_at(from_square)
        self.set_piece_at(to_square, color)

    def change_color_at(self, square):
        bb_square = 1 << square
        color = self.occupied_co[WHITE] & bb_square > 0  # because WHITE == True
        self.occupied_co[color] &= (bb_square ^ BB_ALL)
        self.occupied_co[not color] |= bb_square

    def diff(self, board_from):
        '''
        The object inherits from `BaseBoard` an `occupied` attribute which
        corresponds to a binary mask (encoded as an integer) indicating which
        cases are occupied on the board. We compute `emptied`, `filled` and
        `changed` as binary masks as well.
        '''
        emptied = ~self.occupied & board_from.occupied
        filled = self.occupied & ~board_from.occupied
        changed = self.occupied_co[WHITE] & board_from.occupied_co[BLACK]
        changed |= self.occupied_co[BLACK] & board_from.occupied_co[WHITE]

        return BlindBoard.Diff(emptied, filled, changed)

    @classmethod
    def from_dict(cls, occupied_squares):
        '''
        Build a BlindBoard from a dictionary with the structure {square: color}
        '''
        board = cls()
        for square, color in occupied_squares.items():
            board.set_piece_at(square, Piece(PAWN, color))
        return board

    @classmethod
    def from_board(cls, base_board):
        '''
        Takes as input a `Board` object and makes it 'blind' by turning all
        pieces into pawns.
        '''
        blindboard = cls()

        for color in (BLACK, WHITE):
            occupied_squares = BlindBoard.Diff.get_squares_from_mask(
                    base_board.occupied_co[color])
            for square in occupied_squares:
                blindboard.set_piece_at(square, color)

        return blindboard

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

    @staticmethod
    def get_starting_board():
        return BlindBoard(fen=STARTING_BOARD_FEN)

    ########################################


    class Diff:
        '''Diff between two BlindBoard'''

        def __init__(self, emptied, filled, changed):
            '''
            `emptied`, `filled` and `changed` are 64-bit bitsets, represented as
            integers, describing respectively the set of squares that were
            emptied, filled, or whose color has changed between two blindboards.
            '''
            self.emptied = emptied or BB_VOID
            self.filled = filled or BB_VOID
            self.changed = changed or BB_VOID

        def __eq__(self, other):
            return all(getattr(self, attr) == getattr(other, attr)
                       for attr in ('emptied', 'filled', 'changed'))

        def __str__(self):
            return "emptied:{} filled:{} changed:{}".format(*(
                {SQUARE_NAMES[k] for k in self.get_squares_from_mask(n)}
                for n in (self.emptied, self.filled, self.changed)))

        def length(self):
            '''Get the size of the diff

            Returns: the size of the three sets `emptied`, `filled` and
               `changed`
            '''
            return tuple(bin(n).count('1') for n in (self.emptied, self.filled,
                                                     self.changed))

        def get_emptied(self):
            return self.get_squares_from_mask(self.emptied)

        def get_filled(self):
            return self.get_squares_from_mask(self.filled)

        def get_changed(self):
            return self.get_squares_from_mask(self.changed)

        @staticmethod
        def get_squares_from_mask(mask):
            '''
            Takes as input an integer `n` and returns the indices of the
            set bits in its binary representation on 64 bits, as a set
            '''
            return set( i for i in chess.SQUARES if mask & (1<<i) )

