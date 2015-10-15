from enum import Enum

class PlayMode(Enum):
    ONE_PLAYER = 1
    TWO_PLAYERS = 2

class PlayerColor(Enum):
    WHITE = 1
    BLACK = 2

def opposite_color(color):
    assert color in PlayerColor
    return PlayerColor.WHITE if color == PlayerColor.BLACK \
           else PlayerColor.BLACK

class ChessEngine:

    def __init__(self, playmode):
        assert playmode in PlayMode

    def play_move(self, player, move):
        pass

    def read_move(self):
        pass

