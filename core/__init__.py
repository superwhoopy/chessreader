
from . import diffreader
from .. import utils, chessboard, imgprocessor, engine

from enum import Enum

class IllegalMove(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class PlayMode(Enum):
    '''TODO'''
    ONE_PLAYER = 1
    TWO_PLAYERS = 2


class Core:
    def __init__(self):
        self.capture_engine        = imgprocessor.CaptureEngine()
        self.chess_engine          = engine.GnuChess()
        self.last_valid_chessboard = chessboard.board.START_BLINDBOARD

    def run(self):
        pass

    def receive_chessboard(self, new_chessboard):
        if new_chessboard == self.last_valid_chessboard:
            pass

        board_diff = new_chessboard.diff(self.last_valid_chessboard)
        try:
            move = diffreader.read(board_diff)
        except IllegalMove as move:
            utils.log.warn(move)

    def kill(self):
        pass
