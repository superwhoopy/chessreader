import nose.tools

from .. import chess, core
from ..chess import Color


WHITE_START_SQUARES = [ '{}{}'.format(col, row)
                                for col in chess.board.COL_NAMES
                                for row in [ 1, 2 ] ]

BLACK_START_SQUARES = [ '{}{}'.format(col, row)
                                for col in chess.board.COL_NAMES
                                for row in [ 7, 8 ] ]


class TestDiffer:

    def setUp(self):
        occupied_squares = dict()
        for square in WHITE_START_SQUARES:
            occupied_squares[square] = Color.WHITE
        for square in BLACK_START_SQUARES:
            occupied_squares[square] = Color.BLACK

        self.bb_start = chess.board.BlindBoard(occupied_squares)

    def test_simple_move_diff(self):
        board_1 = chess.board.BlindBoard( {
                        'e2': Color.WHITE,
                        'f2': Color.BLACK,
                    })
        board_2 = chess.board.BlindBoard( {
                        'e4': Color.WHITE,
                        'f2': Color.BLACK,
                    })
        diff = chess.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = chess.moves.Move('e2', 'e4')

        nose.tools.eq_(move, expected_move)


    def test_take_move_diff(self):
        board_1 = chess.board.BlindBoard( {
                        'e2': Color.WHITE,
                        'f2': Color.BLACK,
                    })
        board_2 = chess.board.BlindBoard( {
                        'f2': Color.WHITE,
                    })

        diff = chess.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = chess.moves.Move('e2', 'f2')

        nose.tools.eq_(move, expected_move)


    def test_king_castling_move_diff(self):
        board_1 = chess.board.BlindBoard( {
                        'h1': Color.WHITE,
                        'e1': Color.WHITE,
                    })
        board_2 = chess.board.BlindBoard( {
                        'f1': Color.WHITE,
                        'g1': Color.WHITE,
                    })

        diff = chess.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = chess.moves.Castling(chess.moves.Castling.Side.KING)

        nose.tools.eq_(move, expected_move)

    def test_queen_castling_move_diff(self):
        board_1 = chess.board.BlindBoard( {
                        'e8': Color.BLACK,
                        'a8': Color.BLACK,
                    })
        board_2 = chess.board.BlindBoard( {
                        'd8': Color.BLACK,
                        'c8': Color.BLACK,
                    })

        diff = chess.board.BlindBoard.diff_board(board_2, board_1)
        move = core.diffreader.read(diff)
        expected_move = chess.moves.Castling(chess.moves.Castling.Side.QUEEN)

        nose.tools.eq_(move, expected_move)

    def tearDown(self):
        pass
