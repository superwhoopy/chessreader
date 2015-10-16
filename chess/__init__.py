from enum import Enum

class PlayMode(Enum):
    ONE_PLAYER = 1
    TWO_PLAYERS = 2

class Color(Enum):
    WHITE = 1
    BLACK = 2

    @staticmethod
    def opposite(color):
        assert color in Color
        return Color.WHITE if color == Color.BLACK \
               else Color.BLACK


class ChessEngine:

    def __init__(self, playmode):
        assert playmode in PlayMode

    def play_move(self, player, move):
        pass

    def read_move(self):
        pass

