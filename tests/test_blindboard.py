'''TODO
'''

import nose.tools
import chess.board
from chess import PlayerColor

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


def test_BlindBoard_moves_0():
    board_from = chess.board.BlindBoard({'a1': PlayerColor.WHITE,
                                         'a7': PlayerColor.BLACK,
                                         'a8': PlayerColor.BLACK,
                                         'e2': PlayerColor.WHITE})
    board_to   = chess.board.BlindBoard({'a1': PlayerColor.WHITE,
                                         'a7': PlayerColor.BLACK,
                                         'a8': PlayerColor.WHITE,
                                         'e4': PlayerColor.WHITE})
    emptied, filled, changed = board_to - board_from

    nose.tools.eq_(emptied, ['e2'])
    nose.tools.eq_(filled,  ['e4'])
    nose.tools.eq_(changed, ['a8'])

def test_BlindBoard_identical_board():
    board_from = chess.board.BlindBoard({'a1': PlayerColor.WHITE,
                                         'a7': PlayerColor.BLACK,
                                         'a8': PlayerColor.BLACK,
                                         'e2': PlayerColor.WHITE})
    board_to   = chess.board.BlindBoard({'a1': PlayerColor.WHITE,
                                         'a7': PlayerColor.BLACK,
                                         'a8': PlayerColor.BLACK,
                                         'e2': PlayerColor.WHITE})
    emptied, filled, changed = board_to - board_from

    nose.tools.ok_(not emptied)
    nose.tools.ok_(not filled)
    nose.tools.ok_(not changed)

