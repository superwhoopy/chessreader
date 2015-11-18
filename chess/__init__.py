from enum import Enum

class Color(Enum):
    '''Chess pieces color representation'''
    WHITE = 1
    BLACK = 2

    @staticmethod
    def opposite(color):
        '''Invert a `Color`: `BLACK` becomes `WHITE` and conversely'''
        assert color in Color
        return Color.WHITE if color == Color.BLACK \
               else Color.BLACK

class Piece(Enum):
    QUEEN  = 'Q'
    ROOK   = 'R'
    BISHOP = 'B'
    KNIGHT = 'N'

