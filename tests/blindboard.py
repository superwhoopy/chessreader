import unittest
import log
import imgprocessor.chessboard

class BlindBoardTest(unittest.TestCase):

    def test_square_name(self):
        testcases =
            {
                'a1' : [0,0]
                'h8' : [7,7]
            }
        for key, value in testcases:
            self.assertEqual(key,
                    imgprocessor.chessboard.get_square_name(value[0], value[1]))
        return
