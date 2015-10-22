import nose.tools

import chess.board
from chess import Color as Color
import core.diffreader

def test_simple_move_diff():
    board_1 = chess.board.BlindBoard( {
                    'e2': Color.WHITE,
                    'f2': Color.BLACK,
                })
    board_2 = chess.board.BlindBoard( {
                    'e4': Color.WHITE,
                    'f2': Color.BLACK,
                })
    diff = chess.board.BlindBoard.diff_board(board_2, board_2)
    move = core.diffreader.read(diff)

    nose.tools.eq_(str(move), 'e2e4', 'Expected e2e4 got {}'.format(str(move)))
