'''TODO
'''

import unittest
import utils.log
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

        return

    def test_BlindBoard_moves(self):
        board_from = chessboard.blind.BlindChessboard({'a1', 'e2'})
        board_to   = chessboard.blind.BlindChessboard({'a1', 'e4'})
        emptied, filled = board_from.find_moves(board_to)

        self.assertEqual(emptied, {'e2'})
        self.assertEqual(filled,  {'e4'})

