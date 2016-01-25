'''Chess Board internal representation module

Squares are identified by a string with their usual name, from 'a1' to 'h8'. The
global `ALL_SQUARES` is a list of all of the square names.

The module also provides a `BlindBoard` class used to represent a board where
pieces are only distinguished by their color.
'''

from string import ascii_lowercase, ascii_uppercase
from chess import (BaseBoard, Piece, STARTING_BOARD_FEN, BLACK,
                   WHITE, PAWN, SQUARE_NAMES, BB_H8, BB_VOID)



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
        Two blindboards are identical if the positions of their black and white pieces are the same
        '''
        return all(self.occupied_co[color] == other.occupied_co[color] for color in (BLACK, WHITE))

    def __str__(self):
        board_str = BaseBoard.__str__(self)
        new_chars = []
        for char in board_str:
            new_char = char
            if char in ascii_uppercase:
                new_char = 'W'
            elif char in ascii_lowercase:
                new_char = 'b'
            new_chars.append(new_char)

        return ''.join(new_chars)

    def diff(self, board_from):
        '''
        The object inherits from `BaseBoard` an `occupied` attribute which corresponds to
        a binary mask (encoded as an integer) indicating which cases are occupied on
        the board. We compute `emptied`, `filled` and `changed` as binary masks as well.
        '''
        emptied = ~self.occupied & board_from.occupied
        filled = self.occupied & ~board_from.occupied
        changed = self.occupied_co[WHITE] & board_from.occupied_co[BLACK]
        changed |= self.occupied_co[BLACK] & board_from.occupied_co[WHITE]

        return BlindBoard.Diff(emptied, filled, changed)

    @staticmethod
    def from_dict(occupied_squares):
        '''
        Build a BlindBoard from a dictionary with the structure {square: color}
        '''
        board = BlindBoard()
        for square, color in occupied_squares.items():
            board.set_piece_at(square, Piece(PAWN, color))
        return board

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


    class Diff:
        '''Diff between two BlindBoard'''

        emptied = BB_VOID
        '''Binary mask representing the set of squares that were emptied'''

        filled  = BB_VOID
        '''Binary mask representing the set of squares that were filled'''

        changed = BB_VOID
        '''Binary mask representing the set of squares that were occuppied
        and still are, but whose piece color has changed'''

        def __init__(self, emptied, filled, changed):

            self.emptied = emptied
            self.filled = filled
            self.changed = changed

        def __eq__(self, other):
            return all(getattr(self, attr) == getattr(other, attr) for attr in ('emptied', 'filled', 'changed'))

        def __str__(self):
            return "emptied:{} filled:{} changed:{}".format(*(
                {SQUARE_NAMES[k] for k in self.get_squares_from_mask(n)}
                for n in (self.emptied, self.filled, self.changed)))

        def length(self):
            '''Get the size of the diff

            Returns: the size of the three sets `emptied`, `filled` and
               `changed`
            '''
            return tuple(bin(n).count('1') for n in (self.emptied, self.filled, self.changed))

        def get_emptied(self):
            return self.get_squares_from_mask(self.emptied)

        def get_filled(self):
            return self.get_squares_from_mask(self.filled)

        def get_changed(self):
            return self.get_squares_from_mask(self.changed)

        @staticmethod
        def get_squares_from_mask(n):
            '''
            Takes as input an integer `n` and returns the indices of the
            set bits in its binary representation on 64 bits, as a set
            '''
            pieces = set()
            k = 1 ; i = 0
            while k <= BB_H8:
                if k & n:
                    pieces.add(i)
                k <<= 1 ; i += 1
            return pieces




EMPTY_BLINDBOARD = BlindBoard()
'''Empty board representation'''

START_BLINDBOARD = BlindBoard(fen=STARTING_BOARD_FEN)
'''Board with pieces in starting position'''

