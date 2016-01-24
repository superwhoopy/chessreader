'''TODO
'''
import chess
import nose.tools

from .. import chessboard
from chess import COLORS

def setUp():
    pass

def test_BlindBoardDiff():
    '''Test the __eq__() method of BlindBoard.Diff'''
    diff1 = chessboard.board.BlindBoard.Diff({chess.A1}, {chess.B1, chess.C1}, {})
    diff2 = chessboard.board.BlindBoard.Diff({chess.A1}, {chess.C1, chess.B1}, {})
    nose.tools.ok_(diff1 == diff2)

def test_BlindBoard_moves():
    '''Diff 2 BlinBoards and check the result'''
    board_from = chessboard.board.BlindBoard({chess.A1: COLORS.WHITE,
                                              chess.A7: COLORS.BLACK,
                                              chess.A8: COLORS.BLACK,
                                              chess.E2: COLORS.WHITE})
    board_to   = chessboard.board.BlindBoard({chess.A1: COLORS.WHITE,
                                              chess.A7: COLORS.BLACK,
                                              chess.A8: COLORS.WHITE,
                                              chess.E4: COLORS.WHITE})
    diff = board_to.diff(board_from)

    nose.tools.eq_(diff.emptied, {chess.E2})
    nose.tools.eq_(diff.filled,  {chess.E4})
    nose.tools.eq_(diff.changed, {chess.A8})

def test_BlindBoard_identical():
    '''Diff 2 identical BlindBoards and check the result'''
    board_from = chessboard.board.BlindBoard({chess.A1: COLORS.WHITE,
                                              chess.A7: COLORS.BLACK,
                                              chess.A8: COLORS.BLACK,
                                              chess.E2: COLORS.WHITE})
    board_to   = chessboard.board.BlindBoard({chess.A1: COLORS.WHITE,
                                              chess.A7: COLORS.BLACK,
                                              chess.A8: COLORS.BLACK,
                                              chess.E2: COLORS.WHITE})
    diff = board_to.diff(board_from)

    nose.tools.ok_(not diff.emptied)
    nose.tools.ok_(not diff.filled)
    nose.tools.ok_(not diff.changed)

def tearDown():
    pass
