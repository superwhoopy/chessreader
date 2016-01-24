'''Pieces moves representation'''

from enum import Enum
import re

from . import board
from .. import utils

################################################################################

RE_MOVE = re.compile( r'((?P<move_count>\d+)\. )?'
                      r'(?P<black_move>\.\.\. )?'
                      r'(?P<from_square>[abcdefgh][12345678])'
                      r'(?P<to_square>[abcdefgh][12345678])')

class Castling(Move):
    '''Castling move'''

    class Side(Enum):
        '''Side of the castling move'''
        KING  = 0
        QUEEN = 1

    def __init__(self, side, move_count=None, color=None):
        '''Create a Castling move

        Args:
            side       : side of the castling move : `Side.KING` or `Side.QUEEN`
            move_count : see `Move.__init__()`
            color      : see `Move.__init__()`
        '''
        assert side in Castling.Side
        self.from_square = 'O-O'
        self.to_square   = '-O' if side == Castling.Side.QUEEN \
                                else ''

        self.move_count  = move_count
        self.color       = color


class Promote(Move):
    '''Promotion move'''

    def __init__(self, from_square, to_square, promote_to,
                 move_count=None, color=None):
        '''TODO'''
        # sanity checks
        assert promote_to in Piece
        expected_line = {
                Color.WHITE : [7],
                Color.BLACK : [0],
                None              : [0, 7],
                }
        _, line = board.square_coordinates(to_square)
        assert line in expected_line

        # call parent ctor
        super(Promote, self).__init__(from_square, to_square, move_count,
                color)

        # register the piece to promote to
        self.promote_to = promote_to

    def __eq__(self, other):
        return super(Promote, self).__eq__(other) and \
               self.promote_to == other.promote_to

    def __str__(self):
        return super(Promote, self).__str__() + \
               '={}'.format(self.promote_to.value)
