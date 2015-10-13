import string

COL_NAMES = string.ascii_lowercase[:8]
ROW_NAMES = range(1, 9)

ALL_SQUARES = [ '{}{}'.format(col, row) for col in COL_NAMES \
                                        for row in ROW_NAMES ]

def square_name(x_pos, y_pos):
    # TODO: unit tests of this function
    name = "{}{}".format(COL_NAMES[x_pos], y_pos+1)
    # TODO: throw an exception instead of an assert
    # assert name in ALL_SQUARES
    return name
