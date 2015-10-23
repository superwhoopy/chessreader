'''TODO
'''

import nose.tools

import chess.board
from chess import Color

def setUp():
    pass

def test_BlindBoardDiff():
    diff1 = chess.board.BlindBoard.Diff({'a1'}, {'b1', 'c1'}, {})
    diff2 = chess.board.BlindBoard.Diff({'a1'}, {'c1', 'b1'}, {})
    nose.tools.ok_(diff1 == diff2)

def test_BlindBoard_moves():
    board_from = chess.board.BlindBoard({'a1': Color.WHITE,
                                         'a7': Color.BLACK,
                                         'a8': Color.BLACK,
                                         'e2': Color.WHITE})
    board_to   = chess.board.BlindBoard({'a1': Color.WHITE,
                                         'a7': Color.BLACK,
                                         'a8': Color.WHITE,
                                         'e4': Color.WHITE})
    diff = board_to.diff(board_from)

    nose.tools.eq_(diff.emptied, {'e2'})
    nose.tools.eq_(diff.filled,  {'e4'})
    nose.tools.eq_(diff.changed, {'a8'})

def test_BlindBoard_identical():
    board_from = chess.board.BlindBoard({'a1': Color.WHITE,
                                         'a7': Color.BLACK,
                                         'a8': Color.BLACK,
                                         'e2': Color.WHITE})
    board_to   = chess.board.BlindBoard({'a1': Color.WHITE,
                                         'a7': Color.BLACK,
                                         'a8': Color.BLACK,
                                         'e2': Color.WHITE})
    diff = board_to.diff(board_from)

    nose.tools.ok_(not diff.emptied)
    nose.tools.ok_(not diff.filled)
    nose.tools.ok_(not diff.changed)

def tearDown():
    pass
