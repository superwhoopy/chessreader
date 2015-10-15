import imgprocessor
import chess
import utils

class IllegalMove(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def compute_move(filled_squares, emptied_squares):
    if len(filled_squares) > 1 or len(emptied_squares) > 1:
        errmsg = 'too many pieces seem to have moved: emptied[{}], filled' \
                 '[{}]'.format(emptied_squares, filled_squares)
        raise IllegalMove(errmsg)

    if len(filled_squares) == 1 and len(emptied_squares) == 0:
        errmsg = \
            'a piece seem to have appeared out of nowhere in {}'.format(
                filled_squares)
        raise IllegalMove(errmsg)

    pass


class Core:
    def __init__(self):
        self.capture_engine = imgprocessor.CaptureEngine()
        self.chess_engine   = chess.ChessEngine()
        self.last_valid_chessboard = chess.board.BlindBoard()

    def run(self):
        pass

    def receive_chessboard(self, new_chessboard):

        if new_chessboard == self.last_valid_chessboard:
            pass

        filled, emptied = new_chessboard - self.last_valid_chessboard
        try:
            move = compute_move(filled, emptied)
        except IllegalMove as m:
            utils.log.warn(m)
            pass

    def kill(self):
        pass
