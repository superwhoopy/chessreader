from enum import Enum

from chessboard import BlindBoard

import core.diffreader
import imgprocessor
import engine
import utils

class IllegalMove(Exception):
    pass


class PlayMode(Enum):
    '''TODO'''
    ONE_PLAYER = 1
    TWO_PLAYERS = 2


class Core:
    def __init__(self):
        self.capture_engine        = imgprocessor.CaptureEngine()
        self.chess_engine          = engine.GnuChess()
        self.last_valid_chessboard = BlindBoard.get_starting_board()

    def run(self):
        pass

    def receive_chessboard(self, new_chessboard):
        if new_chessboard == self.last_valid_chessboard:
            pass

        board_diff = new_chessboard.diff(self.last_valid_chessboard)
        try:
            move = core.diffreader.read(board_diff)
        except IllegalMove as move:
            utils.log.warn(move)

    def kill(self):
        pass
