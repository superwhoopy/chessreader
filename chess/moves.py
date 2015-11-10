'''Pieces moves representation'''

from enum import Enum
import chess.board

################################################################################


class Move:
    '''Default simple move, from a square to another'''

    def __init__(self, from_square, to_square, move_count=0):
        assert from_square in chess.board.ALL_SQUARES
        assert to_square   in chess.board.ALL_SQUARES

        self.from_square = from_square
        self.to_square   = to_square
        self.move_count  = move_count

    def __str__(self):
        return '{}{}'.format(self.from_square, self.to_square)

    def __eq__(self,other):
        return self.from_square == other.from_square and \
               self.to_square   == other.to_square


class Castling(Move):
    '''Castling move'''

    class Side(Enum):
        '''Side of the castling move'''
        KING = 0
        QUEEN = 1

    def __init__(self, side):
        assert side in Castling.Side
        self.from_square = 'O-O'
        self.to_square   = '-O' if side == Castling.Side.QUEEN \
                                else ''

# TODO: PromotionMove? EnPassant?
