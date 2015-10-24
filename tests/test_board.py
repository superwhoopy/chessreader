import nose.tools
import chess.board

def setUp():
    pass

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

@nose.tools.raises(chess.board.MalformedSquareName)
def test_square_coordinates_exception_0():
    chess.board.square_coordinates('i0')

@nose.tools.raises(chess.board.MalformedSquareName)
def test_square_coordinates_exception_1():
    chess.board.square_coordinates('i1')

@nose.tools.raises(chess.board.MalformedSquareName)
def test_square_coordinates_exception_2():
    chess.board.square_coordinates('a-2')

@nose.tools.raises(chess.board.SquareOutOfBounds)
def test_square_name_exception():
    chess.board.square_name(-1, 12)

def tearDown():
    pass
