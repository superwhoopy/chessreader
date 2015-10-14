'''TODO
'''

import unittest
import nose.tools
import chessboard.blind

class BoardTest(unittest.TestCase):

    def test_square_name(self):
        squares = {
                'a1' : [0, 0],
                'h8' : [7, 7],
                'e2' : [4, 1],
                'e4' : [4, 3],
            }

        for key in squares:
            value = squares[key]
            self.assertEqual(key,
                    chessboard.square_name(value[0], value[1]))

    @nose.tools.raises(chessboard.SquareOutOfBounds)
    def test_square_coordinates_exception_0(self):
        chessboard.square_coordinates('i0')

    @nose.tools.raises(chessboard.SquareOutOfBounds)
    def test_square_coordinates_exception_1(self):
        chessboard.square_coordinates('i1')

    @nose.tools.raises(chessboard.MalformedSquareName)
    def test_square_coordinates_exception_2(self):
        chessboard.square_coordinates('a-2')

    @nose.tools.raises(chessboard.SquareOutOfBounds)
    def test_square_name_exception(self):
        chessboard.square_name(-1, 12)

    def test_BlindBoard_moves_0(self):
        board_from = chessboard.blind.BlindChessboard({'a1', 'e2'})
        board_to   = chessboard.blind.BlindChessboard({'a1', 'e4'})
        emptied, filled = board_to - board_from

        self.assertEqual(emptied, {'e2'})
        self.assertEqual(filled,  {'e4'})

    def test_BlindBoard_moves_1(self):
        board_from = chessboard.blind.BlindChessboard({'a1', 'e2'})
        board_to   = chessboard.blind.BlindChessboard({'a1', 'e2'})
        emptied, filled = board_to - board_from

        self.assertFalse(emptied)
        self.assertFalse(filled)
