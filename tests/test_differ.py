import nose.tools

import chess
from chess import COLORS, Move

from .. import chessboard, core
from ..chessboard.board import START_BLINDBOARD



class TestDiffer:

    def setUp(self):

        self.bb_start = START_BLINDBOARD

    def test_simple_move_diff(self):
        board_1 = chessboard.board.BlindBoard( {
                        chess.E2: COLORS.WHITE,
                        chess.F2: COLORS.BLACK,
                    })
        board_2 = chessboard.board.BlindBoard( {
                        chess.E4: COLORS.WHITE,
                        chess.F2: COLORS.BLACK,
                    })
        diff = chessboard.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = Move(chess.E2, chess.E4)

        nose.tools.eq_(move, expected_move)


    def test_take_move_diff(self):
        board_1 = chessboard.board.BlindBoard( {
                        chess.E2: COLORS.WHITE,
                        chess.F2: COLORS.BLACK,
                    })
        board_2 = chessboard.board.BlindBoard( {
                        chess.F2: COLORS.WHITE,
                    })

        diff = chessboard.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = Move(chess.E2, chess.F2)

        nose.tools.eq_(move, expected_move)


    def test_king_castling_move_diff(self):
        board_1 = chessboard.board.BlindBoard( {
                        chess.H1: COLORS.WHITE,
                        chess.E1: COLORS.WHITE,
                    })
        board_2 = chessboard.board.BlindBoard( {
                        chess.F1: COLORS.WHITE,
                        chess.G1: COLORS.WHITE,
                    })

        diff = chessboard.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        # TODO is this how we encode castlings?
        expected_move = Move(chess.E1, chess.G1)
        # expected_move = chessboard.moves.Castling(chessboard.moves.Castling.Side.KING)

        nose.tools.eq_(move, expected_move)

    def test_queen_castling_move_diff(self):
        board_1 = chessboard.board.BlindBoard( {
                        chess.E8: COLORS.BLACK,
                        chess.A8: COLORS.BLACK,
                    })
        board_2 = chessboard.board.BlindBoard( {
                        chess.D8: COLORS.BLACK,
                        chess.C8: COLORS.BLACK,
                    })

        diff = chessboard.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = Move(chess.E8, chess.C8)
        # expected_move = chessboard.moves.Castling(chessboard.moves.Castling.Side.QUEEN)

        nose.tools.eq_(move, expected_move)

    def tearDown(self):
        pass
