from enum import Enum

class PlayMode(Enum):
    ONE_PLAYER = 1
    TWO_PLAYERS = 2

class Player(Enum):
    WHITE = 1
    BLACK = 2

class ChessEngine:

    def __init__(self, playmode):
        assert playmode in PlayMode
        pass

    def play_move(self, player, move):
        pass

    def read_move(self):
        pass

