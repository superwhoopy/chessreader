import string

COL_NAMES = string.ascii_lowercase[:8]

def get_square_name(x_pos, y_pos):
    # TODO: unit tests of this function
    return "{}{}".format(COL_NAMES[x_pos], y_pos)

class BlindChessboard:

    occupied_squares = {}

    def __init__(self):
        pass

    def compare_to(self, other_board):
        pass

