import imgprocessor
import chess
import utils

class IllegalMove(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def interpret_move(blind_board_diff):
    # aliases
    emptied = blind_board_diff.emptied
    filled = blind_board_diff.filled
    changed = blind_board_diff.changed

    utils.log.debug("Diffing boards E: {} F: {} C:{}".format(emptied,
        filled, changed))

    if len(filled) > 1 or len(emptied) > 1:
        raise IllegalMove(
                 'too many pieces seem to have moved: emptied[{}], filled' \
                 '[{}]'.format(emptied, filled))

    if len(filled) == 1 and len(emptied) == 0:
        raise IllegalMove(
                'a piece seem to have appeared out of nowhere in {}'.format(
                filled))

    if len(changed) > 1:
        raise IllegalMove(
                'several pieces have changed color in: {}'.format(changed))

    # TODO: analysis of changed pieces?

    return move


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

        board_diff = new_chessboard.diff(self.last_valid_chessboard)
        try:
            move = interpret_move(board_diff)
        except IllegalMove as m:
            utils.log.warn(m)
            pass

    def kill(self):
        pass
