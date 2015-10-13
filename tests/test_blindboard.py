'''TODO
'''

import unittest
import log
import chessboard

class BlindBoardTest(unittest.TestCase):

    def test_square_name(self):
        testcases = {
                'a1' : [0, 0],
                'h8' : [7, 7]
            }

        for key in testcases:
            value = testcases[key]
            self.assertEqual(key,
                    chessboard.get_square_name(value[0], value[1]))

        return
