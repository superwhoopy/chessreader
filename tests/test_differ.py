import nose.tools

import chess
from chess import Move, BLACK, WHITE

from .. import core
from ..chessboard.board import START_BLINDBOARD, BlindBoard

class TestDiffer:

    def setUp(self):

        self.bb_start = START_BLINDBOARD

    def test_simple_move_diff(self):
        '''Test simple move deduction'''
        board_1 = BlindBoard.from_dict( {
                        chess.E2: WHITE,
                        chess.F2: BLACK,
                    })
        board_2 = BlindBoard.from_dict({
                        chess.E4: WHITE,
                        chess.F2: BLACK,
                    })
        diff = BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = Move(chess.E2, chess.E4)

        nose.tools.eq_(move, expected_move)

    def test_take_move_diff(self):
        '''Test take move deduction'''
        board_1 = BlindBoard.from_dict( {
                        chess.E2: WHITE,
                        chess.F2: BLACK,
                    })
        board_2 = BlindBoard.from_dict( {
                        chess.F2: WHITE,
                    })

        diff = BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = Move(chess.E2, chess.F2)

        nose.tools.eq_(move, expected_move)

    def test_king_castling_move_diff(self):
        '''Test king castling move deduction'''
        board_1 = BlindBoard.from_dict( {
                        chess.H1: WHITE,
                        chess.E1: WHITE,
                    })
        board_2 = BlindBoard.from_dict( {
                        chess.F1: WHITE,
                        chess.G1: WHITE,
                    })

        diff = BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = Move(chess.E1, chess.G1)

        nose.tools.eq_(move, expected_move)

    def test_queen_castling_move_diff(self):
        '''Test queen castling move deduction'''
        board_1 = BlindBoard.from_dict({
                        chess.E8: BLACK,
                        chess.A8: BLACK,
                    })
        board_2 = BlindBoard.from_dict({
                        chess.D8: BLACK,
                        chess.C8: BLACK,
                    })

        diff = BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = Move(chess.E8, chess.C8)

        nose.tools.eq_(move, expected_move)

    def tearDown(self):
        pass
