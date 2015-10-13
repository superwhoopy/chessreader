import string

COL_NAMES = string.ascii_lowercase[:8]
ROW_NAMES = range(1,9)

ALL_SQUARES = [ '{}{}'.format(col, row) for col, row in COL_NAMES, ROW_NAMES ]

def get_square_name(x_pos, y_pos):
    # TODO: unit tests of this function
    _NAME = "{}{}".format(COL_NAMES[x_pos], y_pos)
    # TODO: throw an exception instead of an assert
    assert _NAME in ALL_SQUARES
    return _NAME
