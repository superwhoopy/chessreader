from enum import Enum
import chess.board

################################################################################


class Move:
    def __init__(self, from_square, to_square):
        assert from_square in chess.board.ALL_SQUARES
        assert to_square   in chess.board.ALL_SQUARES
        self.from_square = from_square
        self.to_square = to_square

    def __str__(self):
        return '{}{}'.format(self.from_square, self.to_square)


class CastlingMove(Move):
    class Side(Enum):
        KING = 0
        QUEEN = 1

    def __init__(self, side):
        assert side in CastlingMove.Side
        self.from_square = 'O-O'
        self.to_square   = '-O' if side == CastlingMove.Side.QUEEN \
                                else ''

# TODO: PromotionMove?
