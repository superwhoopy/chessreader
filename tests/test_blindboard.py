'''TODO
'''

import unittest
import utils.log
import chessboard

class BlindBoardTest(unittest.TestCase):

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
