from enum import Enum

import chess

import capture, utils
import core.diffreader
from chessboard import BlindBoard


class IllegalMove(Exception):
    pass


class PlayMode(Enum):
    '''TODO'''
    ONE_PLAYER = 1
    TWO_PLAYERS = 2


class Core:

    def __init__(self, image_processor, capture_engine=None):
        self.capture_engine        = capture_engine or capture.Fswebcam()
        self.image_processor       = image_processor
        self.chess_engine          = None  # TODO
        self.last_valid_blindboard = BlindBoard.get_starting_board()
        self.last_valid_board      = chess.Board()

    def process_next_move(self):
        image_path = self.capture_engine.capture()
        self.image_processor.process(image_path)
        new_blindboard = self.image_processor.get_blindboard()
        diff = new_blindboard.diff(self.last_valid_blindboard)
        move = core.diffreader.read(diff)

        utils.log.info("{0} played: {1}".format(self.get_turn_str(), move))

        if not self.last_valid_board.is_legal(move):
            utils.log.warn("Illegal move: {0}".format(move))
            raise IllegalMove(move)

        self.last_valid_board.push(move)
        return move, self.check_game_status()

    def check_game_status(self):
        turn = self.get_turn_str()

        if self.last_valid_board.is_check():
            utils.log.info("{0} is in check!".format(turn))
        elif self.last_valid_board.is_checkmate():
            utils.log.info("{0} is checkmated. Game over!".format(turn))
            return False
        elif self.last_valid_board.is_stalemate():
            utils.log.info("{0} to move is in stalemate. Game over!".format(turn))
            return False

        return True

    def kill(self):
        pass

    def get_turn_str(self):
        return "White" if self.last_valid_board.turn else "Black"


