import imgprocessor
import chess.board
import core
import utils

class IllegalMove(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class Core:
    def __init__(self):
        self.capture_engine = imgprocessor.CaptureEngine()
        self.chess_engine   = chess.ChessEngine(chess.PlayMode.ONE_PLAYER)
        self.last_valid_chessboard = chess.board.BlindBoard()

    def run(self):
        pass

    def receive_chessboard(self, new_chessboard):
        if new_chessboard == self.last_valid_chessboard:
            pass

        board_diff = new_chessboard.diff(self.last_valid_chessboard)
        try:
            move = core.diffreader.read(board_diff)
        except IllegalMove as m:
            utils.log.warn(m)
            pass

    def kill(self):
        pass
