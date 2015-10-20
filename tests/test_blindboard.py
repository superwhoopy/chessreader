'''TODO
'''

import nose.tools
import chess.board
from chess import Color

def test_square_name():
    squares = {
            'a1' : [0, 0],
            'h8' : [7, 7],
            'e2' : [4, 1],
            'e4' : [4, 3],
        }

    for key, value in squares.items():
        nose.tools.eq_(key,
                chess.board.square_name(value[0], value[1]))

@nose.tools.raises(chess.board.SquareOutOfBounds)
def test_square_coordinates_exception_0():
    chess.board.square_coordinates('i0')

@nose.tools.raises(chess.board.SquareOutOfBounds)
def test_square_coordinates_exception_1():
    chess.board.square_coordinates('i1')

@nose.tools.raises(chess.board.MalformedSquareName)
def test_square_coordinates_exception_2():
    chess.board.square_coordinates('a-2')

@nose.tools.raises(chess.board.SquareOutOfBounds)
def test_square_name_exception():
    chess.board.square_name(-1, 12)

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

