'''TODO
'''
import chess

import nose.tools
from nose.tools import eq_
from chessboard.board import BlindBoard
from chess import BLACK, WHITE


def setUp():
    pass


def test_BlindBoard_print():
    '''Pretty-printing of a blind board'''
    blindboard = BlindBoard(chess.STARTING_BOARD_FEN)
    expected = '''b b b b b b b b
b b b b b b b b
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
W W W W W W W W
W W W W W W W W'''
    eq_(str(blindboard), expected)


def test_BlindBoardDiff():
    '''Test the __eq__() method of BlindBoard.Diff'''
    diff1 = BlindBoard.Diff(chess.BB_A1, chess.BB_B1 | chess.BB_C1, chess.BB_VOID)
    diff2 = BlindBoard.Diff(chess.BB_A1, chess.BB_C1 | chess.BB_B1, chess.BB_VOID)
    nose.tools.ok_(diff1 == diff2)


def test_BlindBoard_moves():
    '''Diff 2 BlinBoards and check the result'''
    board_from = BlindBoard.from_dict({chess.A1: WHITE,
                                       chess.A7: BLACK,
                                       chess.A8: BLACK,
                                       chess.E2: WHITE})
    board_to = BlindBoard.from_dict({chess.A1: WHITE,
                                     chess.A7: BLACK,
                                     chess.A8: WHITE,
                                     chess.E4: WHITE})
    diff = board_to.diff(board_from)

    nose.tools.eq_(diff.emptied, chess.BB_E2)
    nose.tools.eq_(diff.filled, chess.BB_E4)
    nose.tools.eq_(diff.changed, chess.BB_A8)
    nose.tools.eq_(diff.__str__(), "emptied:{'e2'} filled:{'e4'} changed:{'a8'}")


def test_BlindBoard_identical():
    '''Diff 2 identical BlindBoards and check the result'''
    board_from = BlindBoard.from_dict({chess.A1: WHITE,
                                       chess.A7: BLACK,
                                       chess.A8: BLACK,
                                       chess.E2: WHITE})
    board_to = BlindBoard.from_dict({chess.A1: WHITE,
                                     chess.A7: BLACK,
                                     chess.A8: BLACK,
                                     chess.E2: WHITE})
    diff = board_to.diff(board_from)

    nose.tools.ok_(not diff.emptied)
    nose.tools.ok_(not diff.filled)
    nose.tools.ok_(not diff.changed)


def test_BlindBoard_changes():
    '''Test BlindBoard editing methods'''
    board = BlindBoard.from_dict({chess.A1: WHITE, chess.B2: WHITE})
    board.move_piece(chess.A1, chess.D5)
    board.change_color_at(chess.B2)
    nose.tools.eq_(board, BlindBoard.from_dict({chess.D5: WHITE,
                                                chess.B2: BLACK}))


def tearDown():
    pass
